#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os, time, uuid, json, requests, pandas as pd
from typing import Dict, List

BASE_URL = os.getenv("LLM_BASE_URL", "https://llmapi.paratera.com/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "DeepSeek-V3.2-Instruct")
MIN_API_CALL_INTERVAL = 1.0

_last_api_call_time = 0

def wait_for_api_cooldown():
    global _last_api_call_time
    current_time = time.time()
    time_since_last_call = current_time - _last_api_call_time
    if time_since_last_call < MIN_API_CALL_INTERVAL:
        time.sleep(MIN_API_CALL_INTERVAL - time_since_last_call)
    _last_api_call_time = time.time()

def call_deepseek_model(prompt, history=None, model=MODEL, stream=False, timeout=180, max_retries=3):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "id": str(uuid.uuid4()),
        "stream": stream,
        "model": model,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.1
    }
    
    backoff = 5
    for attempt in range(max_retries + 1):
        try:
            wait_for_api_cooldown()
            r = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=timeout)
            
            if r.status_code == 200:
                data = r.json()
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]
                return "(Unable to parse response content)"
            
            elif r.status_code == 401:
                return "(Invalid API key)"
            elif r.status_code == 429 or r.status_code in {500, 502, 503}:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            else:
                r.raise_for_status()
                
        except Exception as e:
            if attempt == max_retries:
                return f"(API call failed)"
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
            
    return "(API call failed)"

def parse_model_response(response: str) -> Dict[str, str]:
    result = {"answer": "", "reasoning": ""}
    response = response.strip()
    
    for char in response:
        if char in 'ABCD':
            result["answer"] = char
            break
    
    if result["answer"]:
        parts = response.split(result["answer"], 1)
        result["reasoning"] = parts[1].strip() if len(parts) > 1 else response
    else:
        result["reasoning"] = response
    
    return result

def read_json_questions(input_file: str) -> List[Dict]:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'questions' in data:
        return data['questions']
    else:
        return []

def process_questions(input_file: str, output_file: str):
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.json':
        questions_data = read_json_questions(input_file)
        total_questions = len(questions_data)
        
        if total_questions == 0:
            return None
            
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
        total_questions = len(df)
        
        questions_data = []
        for idx, row in df.iterrows():
            question_data = {
                "id": idx + 1,
                "question": str(row.get('question', '')).strip(),
                "options": {
                    'A': str(row.get('A', '')).strip(),
                    'B': str(row.get('B', '')).strip(),
                    'C': str(row.get('C', '')).strip(),
                    'D': str(row.get('D', '')).strip(),
                },
                "correct_answer": str(row.get('correct_answer', '')).strip().upper()
            }
            questions_data.append(question_data)
    else:
        return None
    
    results = []
    
    for idx, question_data in enumerate(questions_data):
        question_id = idx + 1
        
        question = str(question_data.get('question', '')).strip()
        options = question_data.get('options', {})
        correct_answer = str(question_data.get('correct_answer', '')).strip().upper()
        
        question_text = f"{question}\n\n"
        for opt in ['A', 'B', 'C', 'D']:
            text = options.get(opt, "")
            if text:
                question_text += f"{opt}. {text}\n"
        
        prompt = f"{question_text}\nPlease provide the answer (only A/B/C/D) and briefly explain the reasoning."
        
        response = call_deepseek_model(prompt, history=None, model=MODEL, stream=False)
        parsed = parse_model_response(response)
        is_correct = parsed["answer"].upper() == correct_answer
        
        result = {
            "id": question_data.get('id', question_id),
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "deepseek_v3_2_model_result": {
                "answer": parsed["answer"],
                "reasoning": parsed["reasoning"],
                "is_correct": is_correct
            }
        }
        
        results.append(result)
        
        if question_id % 10 == 0:
            current_correct = sum(1 for r in results if r['deepseek_v3_2_model_result']['is_correct'])
            current_total = len(results)
            current_accuracy = current_correct / current_total * 100
            print(f"Progress: {question_id}/{total_questions} | Current accuracy: {current_accuracy:.2f}%")
            
            full_output = {
                "summary": {
                    "total_questions_processed": current_total,
                    "correct_answers_count": current_correct,
                    "incorrect_answers_count": current_total - current_correct,
                    "accuracy_rate": round(current_accuracy, 2),
                    "progress_status": f"{question_id}/{total_questions}",
                    "model_name": MODEL
                },
                "detailed_results": results
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_output, f, ensure_ascii=False, indent=2)
    
    final_correct = sum(1 for r in results if r['deepseek_v3_2_model_result']['is_correct'])
    final_total = len(results)
    final_accuracy = final_correct / final_total * 100 if final_total > 0 else 0
    
    final_output = {
        "summary": {
            "total_questions": final_total,
            "correct_answers": final_correct,
            "incorrect_answers": final_total - final_correct,
            "accuracy_percentage": round(final_accuracy, 2),
            "accuracy_fraction": f"{final_correct}/{final_total}",
            "model_used": MODEL,
            "processing_completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing completed! Accuracy: {final_accuracy:.2f}% ({final_correct}/{final_total})")
    print(f"Results saved to: {output_file}")
    
    return final_output

def parse_args():
    parser = argparse.ArgumentParser(description="Run the standalone LLM baseline on a batch of questions.")
    parser.add_argument("--input-file", required=True, help="Input question file in JSON, XLSX, or XLS format.")
    parser.add_argument("--output-file", required=True, help="Output JSON file for baseline answers and metrics.")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", BASE_URL), help="LLM API base URL.")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="LLM API key.")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", MODEL), help="LLM model name.")
    return parser.parse_args()


def main():
    global BASE_URL, API_KEY, MODEL

    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    BASE_URL = args.base_url
    API_KEY = args.api_key
    MODEL = args.model

    print("="*50)
    print("Batch Question Processing Tool")
    print(f"Model: {MODEL}")
    print("="*50)

    if not API_KEY:
        raise ValueError("No LLM API key provided. Use --api-key or LLM_API_KEY.")

    if not os.path.exists(input_file):
        print(f"File does not exist: {input_file}")
        return
    
    try:
        final_result = process_questions(input_file, output_file)
        
        if final_result:
            stats = final_result['summary']
            print("\n" + "="*50)
            print("Processing Results:")
            print(f"Total questions: {stats['total_questions']}")
            print(f"Correct answers: {stats['correct_answers']}")
            print(f"Incorrect answers: {stats['incorrect_answers']}")
            print(f"Accuracy: {stats['accuracy_percentage']}%")
            print("="*50)
        
    except KeyboardInterrupt:
        print("\nUser interrupted processing")
    except Exception as e:
        print(f"\nProcessing error: {e}")

if __name__ == "__main__":
    try:
        import pandas
    except ImportError:
        print("Please install dependencies: pip install pandas requests")
        exit(1)
    
    main()