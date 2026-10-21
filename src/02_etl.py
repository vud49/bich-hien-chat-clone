import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = WORKSPACE_DIR / "data_raw"
DATA_PROCESSED_DIR = WORKSPACE_DIR / "data_processed"


def fix_mojibake(text: str) -> str:
    """
    Standard Facebook Messenger Latin-1 Mojibake to UTF-8 fixer.
    """
    if not isinstance(text, str):
        return ""
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def load_config() -> Dict[str, Any]:
    config_path = DATA_PROCESSED_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_system_or_noise(text: str, msg_type: Optional[int] = None) -> bool:
    """
    Determines if a message is a system notification, call notification, reaction, or noise.
    """
    if not text or not text.strip():
        return True

    # Discord call type
    if msg_type == 3:
        return True

    t_clean = text.strip()
    t_lower = t_clean.lower()

    # Reactions
    if t_lower.startswith("reacted ") or " to your message" in t_lower:
        return True
    if t_lower.startswith("đã bày tỏ cảm xúc") or t_lower.startswith("đã gỡ một bày tỏ cảm xúc"):
        return True

    # Call notifications
    call_patterns = [
        r"^cuộc gọi thoại đã kết thúc",
        r"^cuộc gọi video đã kết thúc",
        r"^nhỡ cuộc gọi",
        r"^đã bắt đầu cuộc gọi",
        r"^bạn đã gọi",
        r"^missed (audio|video) call",
        r"^missed video chat",
        r"^started an? (audio|video)?\s*call",
        r"^the call ended",
        r"^you missed a call",
        r"^you called"
    ]
    for cp in call_patterns:
        if re.search(cp, t_lower):
            return True

    # Sticker or attachment placeholders with no words
    if re.fullmatch(r"\[(sticker|ảnh|tệp đính kèm|attachment|image)\]", t_lower):
        return True

    return False


def normalize_sender(raw_sender: str, user_aliases: List[str], target_aliases: List[str]) -> Optional[str]:
    """
    Maps a raw sender string to either 'USER' or 'TARGET', or None if foreign.
    """
    if not raw_sender:
        return None
    raw_s = fix_mojibake(raw_sender).strip()

    # Direct match or case-insensitive match
    for u in user_aliases:
        if raw_s == u or raw_s.lower() == u.lower():
            return "USER"
    for t in target_aliases:
        if raw_s == t or raw_s.lower() == t.lower():
            return "TARGET"

    return None


def extract_raw_messages(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reads Instagram and Discord files, extracts raw normalized messages:
    {"sender_role": "USER"|"TARGET", "sender_name": str, "content": str, "timestamp": float}
    """
    user_aliases = config["user_aliases"]
    target_aliases = config["target_aliases"]
    all_messages = []

    # 1. Instagram
    insta_file = DATA_RAW_DIR / "instagram_messages.json"
    if insta_file.exists():
        print(f"[+] Đọc dữ liệu Instagram: {insta_file.name}")
        with open(insta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = data.get("messages", [])
        for m in msgs:
            raw_sender = m.get("sender_name", "")
            role = normalize_sender(raw_sender, user_aliases, target_aliases)
            if not role:
                continue

            content = fix_mojibake(m.get("content", ""))
            if is_system_or_noise(content):
                continue

            ts_ms = m.get("timestamp_ms", 0)
            ts_sec = ts_ms / 1000.0

            all_messages.append({
                "source": "instagram",
                "sender_role": role,
                "sender_name": fix_mojibake(raw_sender),
                "content": content.strip(),
                "timestamp": ts_sec
            })

    # 2. Discord
    discord_file = DATA_RAW_DIR / "discord_messages.json"
    if discord_file.exists():
        print(f"[+] Đọc dữ liệu Discord: {discord_file.name}")
        with open(discord_file, "r", encoding="utf-8") as f:
            msgs = json.load(f)
        for m in msgs:
            author_obj = m.get("author") or {}
            raw_sender = author_obj.get("username") or m.get("userName") or ""
            role = normalize_sender(raw_sender, user_aliases, target_aliases)
            if not role:
                continue

            content = m.get("content", "")
            msg_type = m.get("type", 0)
            if is_system_or_noise(content, msg_type):
                continue

            ts_str = m.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts_str)
                ts_sec = dt.timestamp()
            except Exception:
                continue

            all_messages.append({
                "source": "discord",
                "sender_role": role,
                "sender_name": raw_sender,
                "content": content.strip(),
                "timestamp": ts_sec
            })

    # Sort strictly chronologically
    all_messages.sort(key=lambda x: x["timestamp"])
    print(f"[+] Tổng số tin nhắn thô hợp lệ sau lọc nhiễu: {len(all_messages):,}")
    return all_messages


def group_turns(messages: List[Dict[str, Any]], max_gap_seconds: float = 180.0) -> List[Dict[str, Any]]:
    """
    Groups consecutive messages from the same sender sent within max_gap_seconds into a single turn,
    separated by '<br_msg>'.
    """
    if not messages:
        return []

    turns = []
    current_turn = None

    for msg in messages:
        if current_turn is None:
            current_turn = {
                "sender_role": msg["sender_role"],
                "sender_name": msg["sender_name"],
                "source": msg["source"],
                "messages": [msg["content"]],
                "start_timestamp": msg["timestamp"],
                "last_timestamp": msg["timestamp"]
            }
        else:
            same_sender = (msg["sender_role"] == current_turn["sender_role"])
            time_gap = msg["timestamp"] - current_turn["last_timestamp"]

            if same_sender and (time_gap <= max_gap_seconds):
                # Append to current turn
                current_turn["messages"].append(msg["content"])
                current_turn["last_timestamp"] = msg["timestamp"]
            else:
                # Close current turn
                turn_text = " <br_msg> ".join(current_turn["messages"])
                turns.append({
                    "sender_role": current_turn["sender_role"],
                    "sender_name": current_turn["sender_name"],
                    "source": current_turn["source"],
                    "text": turn_text,
                    "bubble_count": len(current_turn["messages"]),
                    "timestamp": current_turn["start_timestamp"],
                    "end_timestamp": current_turn["last_timestamp"]
                })
                # Start new turn
                current_turn = {
                    "sender_role": msg["sender_role"],
                    "sender_name": msg["sender_name"],
                    "source": msg["source"],
                    "messages": [msg["content"]],
                    "start_timestamp": msg["timestamp"],
                    "last_timestamp": msg["timestamp"]
                }

    if current_turn:
        turn_text = " <br_msg> ".join(current_turn["messages"])
        turns.append({
            "sender_role": current_turn["sender_role"],
            "sender_name": current_turn["sender_name"],
            "source": current_turn["source"],
            "text": turn_text,
            "bubble_count": len(current_turn["messages"]),
            "timestamp": current_turn["start_timestamp"],
            "end_timestamp": current_turn["last_timestamp"]
        })

    print(f"[+] Gom nhịp gõ phím thành công: {len(turns):,} lượt hội thoại (turns)")
    return turns


def build_clean_pairs(turns: List[Dict[str, Any]], max_response_gap_seconds: float = 3600.0 * 12) -> List[Dict[str, Any]]:
    """
    Extracts consecutive (USER -> TARGET) turns into clean dialogue pairs.
    """
    pairs = []
    for i in range(len(turns) - 1):
        turn_curr = turns[i]
        turn_next = turns[i+1]

        if turn_curr["sender_role"] == "USER" and turn_next["sender_role"] == "TARGET":
            gap = turn_next["timestamp"] - turn_curr["end_timestamp"]
            if gap <= max_response_gap_seconds:
                pairs.append({
                    "user": turn_curr["text"],
                    "reply": turn_next["text"],
                    "timestamp": turn_next["timestamp"],
                    "time_gap_seconds": round(gap, 1),
                    "source": turn_next["source"]
                })

    print(f"[+] Trích xuất thành công: {len(pairs):,} cặp hội thoại (clean pairs)")
    return pairs


def build_episodic_memory(turns: List[Dict[str, Any]], session_silence_seconds: float = 7200.0) -> List[Dict[str, Any]]:
    """
    Splits the conversation history into episodic sessions when silence exceeds 2 hours (7200s).
    """
    if not turns:
        return []

    sessions = []
    current_session_turns = [turns[0]]

    for i in range(1, len(turns)):
        prev_turn = turns[i - 1]
        curr_turn = turns[i]
        gap = curr_turn["timestamp"] - prev_turn["end_timestamp"]

        if gap > session_silence_seconds:
            # End current session
            start_dt = datetime.fromtimestamp(current_session_turns[0]["timestamp"])
            end_dt = datetime.fromtimestamp(current_session_turns[-1]["end_timestamp"])
            duration_min = round((current_session_turns[-1]["end_timestamp"] - current_session_turns[0]["timestamp"]) / 60.0, 1)

            # Build full text for search / RAG
            text_lines = []
            for t in current_session_turns:
                speaker = "Khoa (User)" if t["sender_role"] == "USER" else "Bích Hiền (Target)"
                # Clean up <br_msg> in transcript for human readability
                readable_content = t["text"].replace(" <br_msg> ", "\n  ")
                text_lines.append(f"{speaker}:\n  {readable_content}")

            full_text = "\n".join(text_lines)

            sessions.append({
                "session_id": f"session_{len(sessions)+1:04d}",
                "date": start_dt.strftime("%Y-%m-%d"),
                "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_minutes": duration_min,
                "turn_count": len(current_session_turns),
                "full_text": full_text
            })

            current_session_turns = [curr_turn]
        else:
            current_session_turns.append(curr_turn)

    if current_session_turns:
        start_dt = datetime.fromtimestamp(current_session_turns[0]["timestamp"])
        end_dt = datetime.fromtimestamp(current_session_turns[-1]["end_timestamp"])
        duration_min = round((current_session_turns[-1]["end_timestamp"] - current_session_turns[0]["timestamp"]) / 60.0, 1)

        text_lines = []
        for t in current_session_turns:
            speaker = "Khoa (User)" if t["sender_role"] == "USER" else "Bích Hiền (Target)"
            readable_content = t["text"].replace(" <br_msg> ", "\n  ")
            text_lines.append(f"{speaker}:\n  {readable_content}")

        full_text = "\n".join(text_lines)

        sessions.append({
            "session_id": f"session_{len(sessions)+1:04d}",
            "date": start_dt.strftime("%Y-%m-%d"),
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": duration_min,
            "turn_count": len(current_session_turns),
            "full_text": full_text
        })

    print(f"[+] Phân đoạn ký ức thành công: {len(sessions):,} phiên trò chuyện (sessions)")
    return sessions


def main():
    print(">>> BƯỚC 2: LỌC NHIỄU, GOM NHỊP TIN NHẮN & TẠO KHO KÝ ỨC (ETL Pipeline)")
    config = load_config()

    # 1. Extract and clean raw messages
    messages = extract_raw_messages(config)
    if not messages:
        print("[!] Không có tin nhắn nào được trích xuất.")
        sys.exit(1)

    # 2. Group into conversational turns
    turns = group_turns(messages, max_gap_seconds=180.0)

    # 3. Build clean pairs (USER -> TARGET)
    pairs = build_clean_pairs(turns, max_response_gap_seconds=3600.0 * 12)
    pairs_file = DATA_PROCESSED_DIR / "clean_pairs.jsonl"
    with open(pairs_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[+] Đã lưu {len(pairs):,} cặp hội thoại vào: {pairs_file}")

    # 4. Build episodic memory sessions
    sessions = build_episodic_memory(turns, session_silence_seconds=7200.0)
    memory_file = DATA_PROCESSED_DIR / "episodic_memory.json"
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)
    print(f"[+] Đã lưu {len(sessions):,} phiên ký ức vào: {memory_file}")

    print("\n>>> BƯỚC 2 HOÀN TẤT THÀNH CÔNG!")


if __name__ == "__main__":
    main()
