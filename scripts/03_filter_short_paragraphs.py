import argparse
import json
from pathlib import Path

SRC_DIR = None
DST_DIR = None
SHORT_THRESHOLD = 500

file_count = 0
total_old_paras = 0
total_removed_paras = 0


def extract_paragraphs(data):
    """
    Support two input formats:
    1. {"metadata": ..., "abstract": ..., "paragraphs": [...]}
    2. [{"index": 1, "content": "..."}]
    """
    if isinstance(data, dict) and "paragraphs" in data:
        return data["paragraphs"], "document"
    elif isinstance(data, list):
        return data, "list"
    else:
        return None, "unsupported"


def clean_paragraphs(paragraphs):
    cleaned_paras = []
    removed_count = 0

    for para in paragraphs:
        if not isinstance(para, dict):
            removed_count += 1
            continue

        content = str(para.get("content", "")).strip()

        if not content:
            removed_count += 1
            continue

        if content.lower().startswith("<table>"):
            removed_count += 1
            continue

        if content.lower().startswith("keywords:"):
            removed_count += 1
            continue

        if len(content) <= SHORT_THRESHOLD:
            removed_count += 1
            continue

        cleaned_paras.append(para)

    return cleaned_paras, removed_count


def process_file(src_path):
    try:
        data = json.loads(src_path.read_text(encoding="utf-8-sig"))

        paragraphs, data_format = extract_paragraphs(data)

        if paragraphs is None:
            print(f"[WARN] {src_path.name}: Unsupported JSON format")
            return None, 0, 0

        old_paras_count = len(paragraphs)
        cleaned_paras, removed_count = clean_paragraphs(paragraphs)

        if data_format == "document":
            output_data = data
            output_data["paragraphs"] = cleaned_paras
        else:
            output_data = cleaned_paras

        return output_data, old_paras_count, removed_count

    except Exception as e:
        print(f"[ERROR] Failed to process file {src_path.name}: {e}")
        return None, 0, 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter short, low-information paragraphs from JSON paragraph files."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory for cleaned JSON files.")
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=SHORT_THRESHOLD,
        help="Paragraphs with content length less than or equal to this value will be removed.",
    )
    return parser.parse_args()


def main():
    global SRC_DIR, DST_DIR, SHORT_THRESHOLD
    global file_count, total_old_paras, total_removed_paras

    args = parse_args()
    SRC_DIR = Path(args.input_dir)
    DST_DIR = Path(args.output_dir)
    SHORT_THRESHOLD = args.short_threshold
    DST_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(SRC_DIR.glob("*.json"))

    if not json_files:
        print(f"[WARN] No JSON files found in {SRC_DIR}")
        return

    print(f"Starting to process {len(json_files)} files...")
    print("=" * 70)

    for src_path in json_files:
        file_count += 1

        output_data, old_paras, removed = process_file(src_path)

        if output_data is None:
            continue

        total_old_paras += old_paras
        total_removed_paras += removed

        output_path = DST_DIR / f"{src_path.stem}_cleaned.json"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            if old_paras > 0:
                percentage = (removed / old_paras) * 100
                remaining = old_paras - removed
                print(
                    f"[{src_path.name:40s}] Original paragraphs: {old_paras:4d} | "
                    f"Removed: {removed:4d} | "
                    f"Retained: {remaining:4d} | "
                    f"Removal rate: {percentage:6.2f}%"
                )
            else:
                print(f"[{src_path.name:40s}] No paragraph data")

        except Exception as e:
            print(f"[ERROR] Failed to write file {output_path.name}: {e}")

    print("=" * 70)
    print("Processing completed!")
    print(f"Total files processed: {file_count}")
    print(f"Total original paragraphs: {total_old_paras}")
    print(f"Total paragraphs removed: {total_removed_paras}")

    if total_old_paras > 0:
        total_percentage = (total_removed_paras / total_old_paras) * 100
        total_remaining = total_old_paras - total_removed_paras
        print(f"Total paragraphs retained: {total_remaining}")
        print(f"Total removal rate: {total_percentage:.2f}%")

    print(f"\nCleaned files saved to: {DST_DIR}")
    print("File naming convention: original_filename_cleaned.json")


if __name__ == "__main__":
    main()