import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = WORKSPACE_DIR / "data_processed"

# Load environment variables
load_dotenv(WORKSPACE_DIR / ".env")


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into words and common ASCII emoticons for BM25.
    """
    t = text.replace("<br_msg>", " ")
    return re.findall(r"(?::\)+|:\(+|:>|:<|:v|:3|;\)|;-\)|-_-|\^\^|T_T|\w+)", t.lower(), flags=re.UNICODE)


class ChatBrain:
    def __init__(self):
        self.pairs_file = DATA_PROCESSED_DIR / "clean_pairs.jsonl"
        self.memory_file = DATA_PROCESSED_DIR / "episodic_memory.json"
        self.persona_file = DATA_PROCESSED_DIR / "persona_prompt.md"
        self.metrics_file = DATA_PROCESSED_DIR / "style_metrics.json"

        self.persona_prompt: str = ""
        self.style_metrics: Dict[str, Any] = {}
        self.pairs: List[Dict[str, Any]] = []
        self.sessions: List[Dict[str, Any]] = []

        self.bm25_pairs: Optional[BM25Okapi] = None
        self.bm25_sessions: Optional[BM25Okapi] = None

        self.gemini_client = None
        self.openai_client = None

        self.load_resources()
        self.init_llm_clients()

    def load_resources(self):
        # 1. Load Persona Prompt
        if self.persona_file.exists():
            with open(self.persona_file, "r", encoding="utf-8") as f:
                self.persona_prompt = f.read()

        # 2. Load Style Metrics
        if self.metrics_file.exists():
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                self.style_metrics = json.load(f)

        # 3. Load Clean Pairs & build BM25
        self.pairs = []
        if self.pairs_file.exists():
            with open(self.pairs_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.pairs.append(json.loads(line.strip()))

        if self.pairs:
            pairs_corpus = [tokenize(p.get("user", "")) for p in self.pairs]
            self.bm25_pairs = BM25Okapi(pairs_corpus)
            print(f"[Brain] Loaded {len(self.pairs):,} few-shot pairs into BM25 index.")

        # 4. Load Episodic Memory & build BM25
        self.sessions = []
        if self.memory_file.exists():
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.sessions = json.load(f)

        if self.sessions:
            session_corpus = [tokenize(s.get("full_text", "")) for s in self.sessions]
            self.bm25_sessions = BM25Okapi(session_corpus)
            print(f"[Brain] Loaded {len(self.sessions):,} episodic sessions into BM25 index.")

    def init_llm_clients(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if gemini_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_api_key)
                print("[Brain] Khởi tạo thành công Gemini Client.")
            except Exception as e:
                print(f"[Brain] Lỗi khởi tạo Gemini Client: {e}")

        if openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                print("[Brain] Khởi tạo thành công OpenAI Client.")
            except Exception as e:
                print(f"[Brain] Lỗi khởi tạo OpenAI Client: {e}")

    def retrieve_few_shots(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Retrieves top_k conversational pairs most relevant to query using BM25.
        """
        if not self.bm25_pairs or not self.pairs:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return self.pairs[-top_k:]

        scores = self.bm25_pairs.get_scores(q_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # If best score is 0 (no common words), pick diverse sample
        if len(scores) > 0 and scores[top_indices[0]] == 0:
            import random
            return random.sample(self.pairs, min(top_k, len(self.pairs)))

        return [self.pairs[i] for i in top_indices]

    def retrieve_episodic_memory(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top_k episodic memory sessions most relevant to query.
        """
        if not self.bm25_sessions or not self.sessions:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25_sessions.get_scores(q_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Only return if there is meaningful relevance
        results = []
        for i in top_indices:
            if scores[i] > 1.0:  # Threshold for relevance
                s = self.sessions[i]
                # Extract most relevant dialogue snippet from session
                snippet = self._extract_relevant_snippet(s["full_text"], q_tokens, max_lines=12)
                results.append({
                    "session_id": s["session_id"],
                    "date": s["date"],
                    "snippet": snippet
                })
        return results

    def _extract_relevant_snippet(self, full_text: str, q_tokens: List[str], max_lines: int = 12) -> str:
        lines = full_text.split("\n")
        if len(lines) <= max_lines:
            return full_text

        best_score = -1
        best_idx = 0
        q_set = set(q_tokens)

        for idx, line in enumerate(lines):
            line_words = set(tokenize(line))
            overlap = len(q_set & line_words)
            if overlap > best_score:
                best_score = overlap
                best_idx = idx

        start = max(0, best_idx - 4)
        end = min(len(lines), start + max_lines)
        return "\n".join(lines[start:end])

    def build_system_prompt(self, memory_episodes: List[Dict[str, Any]], few_shots: List[Dict[str, Any]]) -> str:
        prompt_parts = [self.persona_prompt]

        # Style Metrics Summary
        summary = self.style_metrics.get("summary", {})
        if summary:
            metrics_text = (
                f"\n\n### CHỈ SỐ PHONG CÁCH ĐỊNH LƯỢNG BẮT BUỘC:\n"
                f"- Số bong bóng chat trung bình mỗi lượt: {summary.get('avg_bubbles_per_reply', 2.21)} (trung bình 1-2 thẻ `<br_msg>`).\n"
                f"- Độ dài trung bình mỗi bong bóng: {summary.get('avg_words_per_bubble', 4.16)} từ.\n"
                f"- Tỷ lệ viết hoa đầu câu: {summary.get('capitalization_start_ratio_pct', 1.57)}% (GÕ CHỮ THƯỜNG TOÀN BỘ).\n"
                f"- Tỷ lệ chấm câu cuối: {summary.get('ending_dot_ratio_pct', 0.22)}% (KHÔNG CHẤM CÂU CUỐI)."
            )
            prompt_parts.append(metrics_text)

        # Episodic Memory
        if memory_episodes:
            memory_section = "\n\n### KÝ ỨC CŨ LIÊN QUAN TRÍCH TỪ KHO KÝ ỨC (EPISODIC MEMORY):\n"
            for ep in memory_episodes:
                memory_section += f"--- [Phiên trò chuyện ngày {ep['date']} ({ep['session_id']})] ---\n{ep['snippet']}\n"
            prompt_parts.append(memory_section)

        # Dynamic Few-Shot Examples
        if few_shots:
            few_shot_section = "\n\n### 8 CẶP HỘI THOẠI MẪU TRONG QUÁ KHỨ CÓ NGỮ CẢNH TƯƠNG TỰ (DYNAMIC FEW-SHOT):\n"
            for i, p in enumerate(few_shots, 1):
                few_shot_section += f"Ví dụ {i}:\nKhoa: {p['user']}\nBích Hiền: {p['reply']}\n"
            prompt_parts.append(few_shot_section)

        # Mandatory Enforcement Directive
        directive = (
            "\n\n### LỆNH ÉP BUỘC TỐI CAO:\n"
            "Hãy trả lời đúng theo thói quen của TARGET_NAME (Bích Hiền): câu cực ngắn (3-6 từ/bong bóng), gõ chữ thường, không chấm câu, dùng ký hiệu `<br_msg>` để tách các tin nhắn ngắn liên tiếp. Tuyệt đối không xưng hô kiểu AI!\n"
            "Nếu Khoa gửi ảnh hoặc một chùm tin nhắn, hãy quan sát và phản ứng tự nhiên đúng bản chất của Bích Hiền (thấy ảnh dth thì khen, bựa thì cà khịa, đồ ăn thì thèm, khó hiểu thì hỏi là gì).\n"
            "Nếu muốn thả reaction icon vào tin nhắn/ảnh của Khoa (như trên Instagram: ❤️, 😂, 👌, 😒, 😮), bạn có thể thêm thẻ ở cuối dạng: `<reaction:❤️>`."
        )
        prompt_parts.append(directive)

        return "".join(prompt_parts)

    def post_process_reply(self, raw_reply: str) -> Dict[str, Any]:
        """
        Cleans up LLM output: strips prefixes, extracts reactions, ensures no trailing dots, preserves <br_msg>.
        Returns dict: {"reply": clean_text, "reaction": reaction_or_none}
        """
        if not raw_reply:
            return {"reply": "...", "reaction": None}

        text = raw_reply.strip()

        # Extract reaction if present
        reaction = None
        reaction_match = re.search(r"<reaction:\s*([^\>]+)>", text)
        if reaction_match:
            reaction = reaction_match.group(1).strip()
            text = re.sub(r"<reaction:\s*[^\>]+>", "", text).strip()

        # Remove prefixes like "Bích Hiền:", "bchhniiee_:", "Reply:", etc.
        text = re.sub(r"^(?:bích hiền|bhien|target|reply|bchhniiee_)\s*:\s*", "", text, flags=re.IGNORECASE)

        # Clean individual bubbles
        bubbles = [b.strip() for b in text.split("<br_msg>")]
        clean_bubbles = []
        for b in bubbles:
            if not b:
                continue
            # Remove trailing dots
            b_clean = re.sub(r"\.+$", "", b)
            # If start is capitalized and not an acronym, lowercase it
            if len(b_clean) > 0 and b_clean[0].isupper() and not b_clean.isupper():
                words = b_clean.split()
                if words and words[0] not in ["Khoa", "Thu", "GiaVy", "TikTok", "Winmart"]:
                    words[0] = words[0][0].lower() + words[0][1:]
                    b_clean = " ".join(words)
            clean_bubbles.append(b_clean)

        if not clean_bubbles:
            return {"reply": "hong biết nữa", "reaction": reaction}

        return {"reply": " <br_msg> ".join(clean_bubbles), "reaction": reaction}

    def generate_response(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: Optional[str] = "image/jpeg"
    ) -> Dict[str, Any]:
        """
        Full inference pipeline: RAG + Dynamic Few-Shot + Multimodal LLM call.
        """
        chat_history = chat_history or []
        effective_query = user_message.strip() if user_message else "gửi ảnh"

        # 1. Retrieve 8 Dynamic Few-Shot pairs
        few_shots = self.retrieve_few_shots(effective_query, top_k=8)

        # 2. Retrieve 2 Episodic Memory sessions
        memory_episodes = self.retrieve_episodic_memory(effective_query, top_k=2)

        # 3. Assemble System Prompt
        system_prompt = self.build_system_prompt(memory_episodes, few_shots)

        # 4. Format Recent History (last 10 turns)
        recent_history = chat_history[-10:]

        # 5. Call LLM
        # Attempt Gemini first
        if self.gemini_client:
            try:
                from google.genai import types

                # Format conversation contents for Gemini
                contents = []
                for turn in recent_history:
                    role = "user" if turn.get("role") == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": turn.get("content", "")}]})

                # Build current user turn parts
                user_parts = []
                text_to_send = user_message.strip()
                if not text_to_send and image_bytes:
                    text_to_send = "Khoa vừa gửi một bức ảnh."
                user_parts.append({"text": text_to_send})

                if image_bytes:
                    img_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/jpeg")
                    user_parts.append(img_part)

                contents.append({"role": "user", "parts": user_parts})

                candidate_models = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
                for model_name in candidate_models:
                    try:
                        response = self.gemini_client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config={
                                "system_instruction": system_prompt,
                                "temperature": 0.85,
                                "max_output_tokens": 500,
                                "thinking_config": {"thinking_budget": 0}
                            }
                        )
                        if response and response.text:
                            return self.post_process_reply(response.text)
                    except Exception as me:
                        continue
            except Exception as e:
                print(f"[Brain] Lỗi gọi Gemini API: {e}")

        # Attempt OpenAI
        if self.openai_client:
            try:
                import base64
                messages = [{"role": "system", "content": system_prompt}]
                for turn in recent_history:
                    role = "user" if turn.get("role") == "user" else "assistant"
                    messages.append({"role": role, "content": turn.get("content", "")})

                if image_bytes:
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    content_payload = [
                        {"type": "text", "text": user_message or "Khoa gửi một bức ảnh."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type or 'image/jpeg'};base64,{b64}"}}
                    ]
                    messages.append({"role": "user", "content": content_payload})
                else:
                    messages.append({"role": "user", "content": user_message})

                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.85,
                    max_tokens=150
                )
                if response.choices and response.choices[0].message.content:
                    return self.post_process_reply(response.choices[0].message.content)
            except Exception as e:
                print(f"[Brain] Lỗi gọi OpenAI API: {e}")

        # Fallback if no API key is configured or offline: intelligent few-shot synthesis
        if image_bytes:
            return {"reply": "dth v <br_msg> nhìn ghét ghê 🙂", "reaction": "❤️"}

        if few_shots:
            best_match = few_shots[0]["reply"]
            return self.post_process_reply(best_match)

        return {"reply": "hong biết nữa <br_msg> sao á", "reaction": None}

    def add_feedback_pair(self, user_msg: str, corrected_reply: str):
        """
        Feedback Loop: saves (user_msg, corrected_reply) to clean_pairs.jsonl
        and instantly updates the in-memory BM25 index.
        """
        import time
        clean_user = user_msg.strip()
        clean_reply = corrected_reply.strip()

        new_entry = {
            "user": clean_user,
            "reply": clean_reply,
            "timestamp": time.time(),
            "source": "user_feedback"
        }

        # Append to file
        with open(self.pairs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

        # Update in-memory
        self.pairs.append(new_entry)
        pairs_corpus = [tokenize(p.get("user", "")) for p in self.pairs]
        self.bm25_pairs = BM25Okapi(pairs_corpus)
        print(f"[Brain] Đã cập nhật cặp phản hồi mới và tái lập chỉ mục BM25 ({len(self.pairs)} pairs)!")
