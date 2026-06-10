import argparse
import json
import re
import os
from concurrent.futures import ProcessPoolExecutor

def split_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def extract_content(md_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    title = lines[0].strip().lstrip('#').strip()
    content = '\n'.join(lines[1:]).strip()
    
    paragraphs = split_paragraphs(content)
    return title, paragraphs

def process_document(md_file_path, output_dir):
    try:
        title, paragraphs = extract_content(md_file_path)
        doc = {
            "metadata": {
                "title": title,
                "doi": ""  
            },
            "abstract": "",  
            "paragraphs": [{"index": i + 1, "content": p} for i, p in enumerate(paragraphs)]
        }
        
        base_name = os.path.basename(md_file_path).replace('.md', '.json')
        output_file_path = os.path.join(output_dir, 'auto', base_name)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=4)
        print(f"Processed file saved to {output_file_path}")
    except Exception as e:
        print(f"Error processing {md_file_path}: {e}")

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    md_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    
    if not md_files:
        print(f"No '.md' files found in {input_dir}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_document, file, output_dir) for file in md_files]
        for future in futures:
            future.result()
            print(f"Completed processing")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Markdown files into paragraph-level JSON documents.")
    parser.add_argument("--input-dir", required=True, help="Directory containing Markdown files.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated paragraph JSON files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.output_dir)
