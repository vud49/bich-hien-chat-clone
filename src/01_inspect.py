import os
import sys
import json
import zipfile
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout on Windows
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = WORKSPACE_DIR / "data_raw"
DATA_PROCESSED_DIR = WORKSPACE_DIR / "data_processed"


def fix_mojibake(text: str) -> str:
    """
    Standard Facebook Messenger Latin-1 Mojibake to UTF-8 fixer.
    Handles already-UTF-8 strings and edge cases safely.
    """
    if not isinstance(text, str):
        return ""
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def ensure_raw_json_files():
    """
    Checks for .json files in the workspace.
    If only .zip files exist, extracts them to data_raw/ to provide standard JSON files.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check if Instagram data is extracted
    insta_json_path = DATA_RAW_DIR / "instagram_messages.json"
    insta_zip_candidates = list(WORKSPACE_DIR.glob("*instagram*.zip"))

    if not insta_json_path.exists() and insta_zip_candidates:
        insta_zip = insta_zip_candidates[0]
        print(f"[+] Found Instagram zip: {insta_zip.name}")
        with zipfile.ZipFile(insta_zip, 'r') as z:
            # Find the largest message file (which is the target conversation)
            target_inbox_file = None
            max_size = 0
            for name in z.namelist():
                if name.endswith('.json') and 'messages/inbox/' in name:
                    info = z.getinfo(name)
                    if info.file_size > max_size:
                        max_size = info.file_size
                        target_inbox_file = name

            if target_inbox_file:
                print(f"[+] Extracting main thread '{target_inbox_file}' ({max_size/1024:.1f} KB) -> {insta_json_path.name}")
                raw_bytes = z.read(target_inbox_file)
                # Parse and save as clean JSON
                try:
                    data = json.loads(raw_bytes.decode('utf-8'))
                except Exception:
                    data = json.loads(raw_bytes.decode('latin1'))
                with open(insta_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    # 2. Check if Discord data is extracted
    discord_json_path = DATA_RAW_DIR / "discord_messages.json"
    discord_zip_candidates = list(WORKSPACE_DIR.glob("*Direct Messages*.zip"))

    if not discord_json_path.exists() and discord_zip_candidates:
        discord_zip = discord_zip_candidates[0]
        print(f"[+] Found Discord DM zip: {discord_zip.name}")
        with zipfile.ZipFile(discord_zip, 'r') as z:
            # Sort pages 1 through 6
            page_files = [n for n in z.namelist() if n.endswith('.json') and 'page_' in n]
            page_files.sort()
            all_discord_msgs = []
            for p in page_files:
                raw_bytes = z.read(p)
                try:
                    page_data = json.loads(raw_bytes.decode('utf-8'))
                except Exception:
                    page_data = json.loads(raw_bytes.decode('latin1'))
                if isinstance(page_data, list):
                    all_discord_msgs.extend(page_data)

            print(f"[+] Merged {len(page_files)} pages ({len(all_discord_msgs)} messages) -> {discord_json_path.name}")
            with open(discord_json_path, "w", encoding="utf-8") as f:
                json.dump(all_discord_msgs, f, ensure_ascii=False, indent=2)

    # Return discovered JSON files
    raw_files = list(DATA_RAW_DIR.glob("*.json"))
    return raw_files


def inspect_schema(file_path: Path):
    """
    Inspects schema and data tree structure of a JSON file without loading all into memory.
    """
    print(f"\n{'='*25} KHẢO SÁT CẤU TRÚC: {file_path.name} {'='*25}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        print(f"Loại dữ liệu gốc: Dictionary (Object)")
        print(f"Các khóa cấp 1 (Root Keys): {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, list):
                print(f"  - '{k}': Danh sách gồm {len(v)} phần tử")
                if len(v) > 0 and isinstance(v[0], dict):
                    print(f"    Cấu trúc 1 phần tử mẫu: {list(v[0].keys())}")
            elif isinstance(v, dict):
                print(f"  - '{k}': Dict các khóa {list(v.keys())[:5]}")
            else:
                print(f"  - '{k}': {str(v)[:80]}")
    elif isinstance(data, list):
        print(f"Loại dữ liệu gốc: List (Mảng)")
        print(f"Tổng số phần tử trong mảng: {len(data)}")
        if len(data) > 0 and isinstance(data[0], dict):
            print(f"Cấu trúc 1 phần tử mẫu: {list(data[0].keys())}")
            print(f"Mẫu phần tử đầu tiên:")
            for k, v in list(data[0].items())[:6]:
                print(f"  - '{k}': {str(v)[:80]}")


def analyze_senders(file_path: Path):
    """
    Counts messages per sender, applying fix_mojibake to sender names.
    """
    sender_counts = Counter()
    total_messages = 0

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Instagram format
    if isinstance(data, dict) and 'messages' in data:
        messages = data['messages']
        total_messages = len(messages)
        for m in messages:
            raw_sender = m.get('sender_name', 'Unknown')
            sender = fix_mojibake(raw_sender)
            sender_counts[sender] += 1

    # Discord format
    elif isinstance(data, list):
        total_messages = len(data)
        for m in data:
            if isinstance(m, dict):
                author_obj = m.get('author')
                if isinstance(author_obj, dict):
                    sender = author_obj.get('username') or author_obj.get('name') or 'Unknown'
                else:
                    sender = m.get('userName') or 'Unknown'
                sender_counts[sender] += 1

    print(f"\nThống kê người gửi trong {file_path.name} (Tổng cộng: {total_messages} tin nhắn):")
    for s, c in sender_counts.most_common():
        pct = (c / total_messages * 100) if total_messages else 0
        print(f"  * {s}: {c:,} tin nhắn ({pct:.1f}%)")

    return sender_counts


def main():
    print(">>> BƯỚC 1: KHẢO SÁT CẤU TRÚC & SỬA LỖI FONT (Data Discovery)")
    raw_files = ensure_raw_json_files()
    if not raw_files:
        print("[!] Không tìm thấy file JSON nào trong workspace hoặc data_raw/.")
        sys.exit(1)

    all_senders = Counter()
    for f in raw_files:
        inspect_schema(f)
        counts = analyze_senders(f)
        all_senders.update(counts)

    print("\n" + "="*70)
    print("TỔNG HỢP TOÀN BỘ NGƯỜI THAM GIA TRÊN CẢ 2 NGUỒN:")
    for s, c in all_senders.most_common():
        print(f"  - {s}: {c:,} tin nhắn")

    # Match User and Target automatically
    # User aliases: 'Vũ Đình Khoa', 'vud49'
    # Target aliases: 'ngyn ngc bich hien', 'bchhniiee_'
    config = {
        "user_aliases": ["Vũ Đình Khoa", "vud49", "vud.49"],
        "target_aliases": ["ngyn ngc bich hien", "bchhniiee_", "bchhniiee", "Nguyễn Ngọc Bích Hiền"],
        "user_primary_name": "Vũ Đình Khoa",
        "target_primary_name": "ngyn ngc bich hien",
        "raw_files": [str(f.relative_to(WORKSPACE_DIR)) for f in raw_files]
    }

    config_path = DATA_PROCESSED_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n[+] ĐÃ XÁC ĐỊNH THÀNH CÔNG VÀ LƯU CẤU HÌNH VÀO: data_processed/config.json")
    print(f"    * MY_NAME: {config['user_primary_name']} (Aliases: {config['user_aliases']})")
    print(f"    * TARGET_NAME: {config['target_primary_name']} (Aliases: {config['target_aliases']})")
    print(">>> BƯỚC 1 HOÀN TẤT THÀNH CÔNG!")


if __name__ == "__main__":
    main()
