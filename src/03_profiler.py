import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = WORKSPACE_DIR / "data_processed"

EMOJI_REGEX = re.compile(
    r'[\U0001F600-\U0001F64F]'  # emoticons
    r'|[\U0001F300-\U0001F5FF]'  # symbols & pictographs
    r'|[\U0001F680-\U0001F6FF]'  # transport & map symbols
    r'|[\U0001F1E0-\U0001F1FF]'  # flags
    r'|[\U0001F900-\U0001F9FF]'  # supplemental symbols
    r'|[\U0001FA70-\U0001FAFF]'  # symbols and pictographs extended
    r'|[\U00002600-\U000026FF]'  # misc symbols
    r'|[\U00002700-\U000027BF]'  # dingbats
, flags=re.UNICODE)


def tokenize_words(text: str) -> List[str]:
    """
    Tokenizes text into words and common ASCII emoticons.
    """
    # Replace <br_msg> with space
    t = text.replace("<br_msg>", " ")
    # Keep Vietnamese letters, numbers, common chat emotions (:))), :>, :<, ^^, etc.)
    # Match words or emoticons
    tokens = re.findall(r"(?::\)+|:\(+|:>|:<|:v|:3|;\)|;-\)|-_-|\^\^|T_T|\w+)", t.lower(), flags=re.UNICODE)
    return tokens


def analyze_style():
    pairs_file = DATA_PROCESSED_DIR / "clean_pairs.jsonl"
    if not pairs_file.exists():
        print(f"[!] Không tìm thấy file: {pairs_file}")
        sys.exit(1)

    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line.strip()))

    print(f"[+] Đọc {len(pairs):,} cặp hội thoại từ clean_pairs.jsonl")

    total_replies = len(pairs)
    total_bubbles = 0
    total_br_msg = 0
    words_per_bubble = []
    capitalized_first_char_count = 0
    alpha_first_char_count = 0
    ends_with_dot_count = 0
    ends_with_punct_count = 0

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    emojis = Counter()

    for p in pairs:
        reply = p["reply"]

        # Emojis in reply
        found_emojis = EMOJI_REGEX.findall(reply)
        emojis.update(found_emojis)

        # Count <br_msg>
        br_count = reply.count("<br_msg>")
        total_br_msg += br_count

        # Split bubbles
        raw_bubbles = [b.strip() for b in reply.split("<br_msg>")]
        bubbles = [b for b in raw_bubbles if b]
        total_bubbles += len(bubbles)

        for b in bubbles:
            words = tokenize_words(b)
            if words:
                words_per_bubble.append(len(words))
                unigrams.update(words)
                for i in range(len(words) - 1):
                    bigrams[f"{words[i]} {words[i+1]}"] += 1
                for i in range(len(words) - 2):
                    trigrams[f"{words[i]} {words[i+1]} {words[i+2]}"] += 1

            # Capitalization check
            stripped = b.lstrip()
            if stripped:
                first_char = stripped[0]
                if first_char.isalpha():
                    alpha_first_char_count += 1
                    if first_char.isupper():
                        capitalized_first_char_count += 1

                # Punctuation check
                last_char = stripped[-1]
                if last_char == '.':
                    ends_with_dot_count += 1
                if last_char in '.?!~…':
                    ends_with_punct_count += 1

    avg_bubbles_per_reply = round(total_bubbles / total_replies, 2) if total_replies else 0
    avg_br_per_reply = round(total_br_msg / total_replies, 2) if total_replies else 0
    avg_words_per_bubble = round(sum(words_per_bubble) / len(words_per_bubble), 2) if words_per_bubble else 0
    cap_ratio = round((capitalized_first_char_count / alpha_first_char_count * 100), 2) if alpha_first_char_count else 0
    dot_ratio = round((ends_with_dot_count / total_bubbles * 100), 2) if total_bubbles else 0
    punct_ratio = round((ends_with_punct_count / total_bubbles * 100), 2) if total_bubbles else 0

    metrics = {
        "summary": {
            "total_replies_analyzed": total_replies,
            "total_bubbles": total_bubbles,
            "avg_bubbles_per_reply": avg_bubbles_per_reply,
            "avg_br_msg_per_reply": avg_br_per_reply,
            "avg_words_per_bubble": avg_words_per_bubble,
            "capitalization_start_ratio_pct": cap_ratio,
            "ending_dot_ratio_pct": dot_ratio,
            "ending_any_punctuation_ratio_pct": punct_ratio
        },
        "top_100_unigrams": dict(unigrams.most_common(100)),
        "top_100_bigrams": dict(bigrams.most_common(100)),
        "top_100_trigrams": dict(trigrams.most_common(100)),
        "top_emojis": dict(emojis.most_common(50))
    }

    # Save metrics
    metrics_path = DATA_PROCESSED_DIR / "style_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[+] Đã lưu chỉ số phong cách vào: {metrics_path}")

    # Extract 150 balanced sample pairs (50 start, 50 middle, 50 end)
    n = len(pairs)
    sample_size = 50
    start_samples = pairs[:sample_size]
    mid_start = max(0, (n // 2) - (sample_size // 2))
    mid_samples = pairs[mid_start : mid_start + sample_size]
    end_samples = pairs[-sample_size:] if n >= sample_size else []

    all_150 = start_samples + mid_samples + end_samples
    samples_path = DATA_PROCESSED_DIR / "sample_150_pairs.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(all_150, f, ensure_ascii=False, indent=2)
    print(f"[+] Đã trích xuất 150 cặp mẫu phân bổ đều vào: {samples_path}")

    # Print summary to terminal
    print("\n" + "="*60)
    print("KẾT QUẢ GIẢI PHẪU PHONG CÁCH & THÓI QUEN NHẮN TIN:")
    print(f"  * Tổng số lượt trả lời (replies): {total_replies:,}")
    print(f"  * Số bong bóng chat trung bình / lượt: {avg_bubbles_per_reply} (trung bình {avg_br_per_reply} thẻ <br_msg>)")
    print(f"  * Độ dài trung bình mỗi bong bóng chat: {avg_words_per_bubble} từ")
    print(f"  * Tỷ lệ viết hoa đầu câu: {cap_ratio}% (rất hiếm viết hoa chuẩn)")
    print(f"  * Tỷ lệ kết thúc bằng dấu chấm (.): {dot_ratio}% (hầu như không bao giờ chấm câu cuối)")
    print(f"  * Tỷ lệ kết thúc bằng bất kỳ dấu câu nào: {punct_ratio}%")
    print("\nTop 20 từ đơn dùng nhiều nhất:")
    print("  ", list(dict(unigrams.most_common(20)).keys()))
    print("\nTop 15 cụm 2 từ (Bigrams) phổ biến:")
    print("  ", list(dict(bigrams.most_common(15)).keys()))
    print("\nTop 10 Emoji dùng nhiều nhất:")
    print("  ", list(dict(emojis.most_common(10)).keys()))
    print("="*60)


if __name__ == "__main__":
    analyze_style()
