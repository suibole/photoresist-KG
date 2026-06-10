import argparse
import os
import sys
import time
import uuid
import requests
import json
import re
import random
from neo4j import GraphDatabase
from typing import Dict, List, Any
import concurrent.futures
import pandas as pd

class Config:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "DeepSeek-V3.2-Instruct")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://llmapi.paratera.com/v1")
    LLM_API_KEYS = [key.strip() for key in os.getenv("LLM_API_KEYS", "").split(",") if key.strip()]
    MIN_API_CALL_INTERVAL = 1.0

class LLMClient:
    def __init__(self):
        self.model = Config.LLM_MODEL
        self.base_url = Config.LLM_BASE_URL
        self.api_keys = Config.LLM_API_KEYS
        self._last_api_call_time = 0
    
    def _get_key(self):
        return random.choice(self.api_keys)
    
    def _wait_for_api_cooldown(self):
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call_time
        if time_since_last_call < Config.MIN_API_CALL_INTERVAL:
            time.sleep(Config.MIN_API_CALL_INTERVAL - time_since_last_call)
        self._last_api_call_time = time.time()
    
    def chat_completion(self, prompt, history=None, max_retries=3, timeout=180):
        messages = (history or []) + [{"role": "user", "content": prompt}]
        backoff = 5
        
        for attempt in range(max_retries + 1):
            try:
                self._wait_for_api_cooldown()
                api_key = self._get_key()
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                payload = {
                    "id": str(uuid.uuid4()),
                    "stream": False,
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4000,
                    "temperature": 0.1
                }
                
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=timeout)
                
                if r.status_code == 200:
                    data = r.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            content = re.sub(r'^```(?:cypher|json)?\s*', '', content)
                            content = re.sub(r'\s*```$', '', content)
                            return content.strip()
                
                if r.status_code in {429, 500, 502, 503}:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                    
                r.raise_for_status()
                    
            except requests.exceptions.RequestException:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                return "(LLM call failed)"
        
        return "(LLM call failed)"

class SmartQueryAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        prompt = f"""
Analyze the following technical question and return JSON:
Question: "{question}"

Return JSON containing:
1. "keywords": relevant keywords (list of strings)
2. "cypher_condition": WHERE condition for Cypher query (string)
3. "query_type": type of query
4. "language": "english"

Important requirements:
1. "cypher_condition" must be a valid Cypher WHERE condition
2. Use CONTAINS for fuzzy matching

Example:
Question: "What is the resolution of photoresist?"
{{
    "keywords": ["photoresist", "resolution"],
    "cypher_condition": "toLower(n.name) CONTAINS 'photoresist' OR toLower(m.name) CONTAINS 'resolution'",
    "query_type": "attribute query",
    "language": "english"
}}
"""
        
        try:
            response = self.llm.chat_completion(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        return {
            "keywords": [],
            "cypher_condition": "1=1",
            "query_type": "semantic query",
            "language": "english"
        }

class SimpleNeo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
    
    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
        except Exception:
            return []
    
    def path1_keyword_search(self, keywords: List[str]) -> List[Dict]:
        if not keywords:
            return []
        
        keyword_conditions = []
        params = {}
        
        for i, keyword in enumerate(keywords[:5]):
            param_name = f"keyword{i}"
            params[param_name] = keyword.lower()
            condition = f"""
                toLower(n.name) CONTAINS ${param_name} OR
                toLower(m.name) CONTAINS ${param_name} OR
                toLower(r.type) CONTAINS ${param_name} OR
                ANY(attr IN n.attributes + r.attributes + m.attributes WHERE 
                    toLower(attr) CONTAINS ${param_name}
                )
            """
            keyword_conditions.append(condition)
        
        condition_text = " OR ".join(keyword_conditions) if keyword_conditions else "1=1"
        
        query = f"""
        MATCH (n:SuiboleEntity)-[r:SUIBOLE_RELATION]->(m:SuiboleEntity)
        WHERE {condition_text}
        RETURN 
            n.name as head, 
            n.attributes as head_attrs,
            r.type as relation_type,
            r.attributes as rel_attrs,
            type(r) as relation_label,
            m.name as tail,
            m.attributes as tail_attrs
        LIMIT 15
        """
        
        return self.execute_query(query, params)
    
    def _build_safe_cypher_condition(self, cypher_condition: str):
        """Build a parameterized WHERE clause from a restricted subset of LLM-generated conditions."""
        if not cypher_condition:
            return None, {}

        cypher_condition = cypher_condition.strip()
        if not cypher_condition or cypher_condition == "1=1":
            return None, {}

        cypher_condition = re.sub(r'^\s*MATCH.*WHERE\s+', '', cypher_condition, flags=re.IGNORECASE | re.DOTALL)
        cypher_condition = re.sub(r'^\s*WHERE\s+', '', cypher_condition, flags=re.IGNORECASE)

        allowed_fields = {
            ("n", "name"): "n.name",
            ("m", "name"): "m.name",
            ("r", "type"): "r.type",
        }
        pattern = re.compile(
            r"toLower\(\s*(n|m|r)\.(name|type)\s*\)\s+CONTAINS\s+(['\"])([^'\"]{1,100})\3",
            re.IGNORECASE,
        )

        conditions = []
        params = {}
        for index, match in enumerate(pattern.finditer(cypher_condition)):
            alias, field, _, value = match.groups()
            field_ref = allowed_fields.get((alias.lower(), field.lower()))
            value = value.strip().lower()
            if not field_ref or not value:
                continue

            param_name = f"safe_cypher_{index}"
            conditions.append(f"toLower({field_ref}) CONTAINS ${param_name}")
            params[param_name] = value

        if not conditions:
            return None, {}

        return " OR ".join(conditions[:8]), params

    def path2_cypher_search(self, cypher_condition: str) -> List[Dict]:
        try:
            safe_condition, params = self._build_safe_cypher_condition(cypher_condition)
            
            if not safe_condition:
                return []

            standard_query = f"""
            MATCH (n:SuiboleEntity)-[r:SUIBOLE_RELATION]->(m:SuiboleEntity)
            WHERE {safe_condition}
            RETURN 
                n.name as head, 
                n.attributes as head_attrs,
                r.type as relation_type,
                r.attributes as rel_attrs,
                type(r) as relation_label,
                m.name as tail,
                m.attributes as tail_attrs
            LIMIT 15
            """
            
            return self.execute_query(standard_query, params)
            
        except Exception:
            return []
    
    def close(self):
        self.driver.close()

class SimpleRAGEngine:
    def __init__(self):
        self.llm_client = LLMClient()
        self.neo4j_client = SimpleNeo4jClient()
        self.query_analyzer = SmartQueryAnalyzer(self.llm_client)
    
    def two_path_search(self, question: str) -> Dict[str, Any]:
        analysis = self.query_analyzer.analyze_question(question)
        keywords = analysis.get("keywords", [])
        cypher_condition = analysis.get("cypher_condition", "1=1")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self.neo4j_client.path1_keyword_search, keywords)
            future2 = executor.submit(self.neo4j_client.path2_cypher_search, cypher_condition)
            results1 = future1.result()
            results2 = future2.result()
        
        for results in [results1, results2]:
            for result in results:
                if 'relation_type' in result:
                    result['relation'] = result['relation_type']
                elif 'relation' not in result:
                    result['relation'] = result.get('relation_label', 'unknown')
        
        combined_results = self._merge_results(results1, results2)
        
        triples_counts = {
            "keyword_search_triples": len(results1),
            "cypher_search_triples": len(results2),
            "total_triples_retrieved": len(results1) + len(results2),
            "unique_triples_combined": len(combined_results),
            "retrieval_summary": f"Two-path retrieval obtained {len(results1) + len(results2)} triples, {len(combined_results)} unique after deduplication"
        }
        
        search_queries = [
            {
                "path": "keyword_search",
                "query_type": "Keyword Search",
                "keywords": keywords[:5],
                "triples_retrieved": len(results1)
            },
            {
                "path": "cypher_search",
                "query_type": "Cypher Condition Search",
                "cypher_condition": cypher_condition,
                "triples_retrieved": len(results2)
            }
        ]
        
        return {
            "analysis": analysis,
            "combined_results": combined_results,
            "keywords": keywords,
            "triples_counts": triples_counts,
            "search_queries": search_queries,
            "retrieved_triples_examples": combined_results[:5]
        }
    
    def _merge_results(self, *results_lists):
        seen = set()
        unique_results = []
        
        for results in results_lists:
            for result in results:
                key = (result.get('head', ''), result.get('relation', ''), result.get('tail', ''))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
        
        return unique_results[:20]
    
    def generate_rag_answer(self, question: str, options: Dict[str, str]) -> Dict[str, Any]:
        start_time = time.time()
        
        print(f"  [DEBUG] Question options: {options}")
        
        search_results = self.two_path_search(question)
        context = self._build_context(search_results["combined_results"])
        
        print(f"  [DEBUG] Retrieved knowledge count: {len(search_results['combined_results'])}")
        
        answer = self._generate_choice_answer(question, options, context)
        
        print(f"  [DEBUG] Model raw response: {repr(answer[:200])}...")
        
        parsed_answer = self._parse_answer_response(answer)
        total_time = time.time() - start_time
        
        return {
            "rag_result": {
                "answer": parsed_answer["answer"],
                "reasoning": parsed_answer["reasoning"],
                "context_info": {
                    "query_type": search_results["analysis"].get("query_type", "Unknown"),
                    "keywords_used": search_results["keywords"][:5],
                    "results_count": len(search_results["combined_results"]),
                    "triples_counts": search_results["triples_counts"],
                    "retrieved_triples_examples": search_results["retrieved_triples_examples"],
                    "response_time": f"{total_time:.2f}s",
                    "all_retrieved_triples": search_results["combined_results"]
                }
            }
        }
    
    def _build_context(self, results: List[Dict]) -> str:
        if not results:
            return "No relevant technical knowledge found"
        
        context_lines = ["Retrieved knowledge:"]
        for i, result in enumerate(results[:8], 1):
            head = result.get('head', 'unknown')
            relation = result.get('relation', 'related to')
            tail = result.get('tail', 'unknown')
            
            head_attrs = result.get('head_attrs', [])
            rel_attrs = result.get('rel_attrs', [])
            tail_attrs = result.get('tail_attrs', [])
            
            line = f"{i}. {head}"
            if head_attrs:
                line += f" [{', '.join(head_attrs)}]"
            line += f" → [{relation}]"
            if rel_attrs:
                line += f" ({', '.join(rel_attrs)})"
            line += f" → {tail}"
            if tail_attrs:
                line += f" [{', '.join(tail_attrs)}]"
            
            context_lines.append(line)
        
        return "\n".join(context_lines)
    
    def _generate_choice_answer(self, question: str, options: Dict[str, str], context: str) -> str:
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        prompt = f"""
You are an expert in physics, photolithography, and nonlinear optics. Answer the following multiple choice question step by step.

Question: {question}

Options:
{options_text}

Retrieved knowledge:
{context}

**IMPORTANT INSTRUCTIONS:**
1. You MUST select ONE answer from the options A, B, C, or D above
2. Your response MUST start with the letter of your choice (A, B, C, or D) on the first line
3. Then provide your detailed reasoning in the following lines
4. Do NOT include any other text before the answer letter

Example format:
C
This is correct because... [your reasoning]
"""
        
        return self.llm_client.chat_completion(prompt)
    
    def _parse_answer_response(self, response: str) -> Dict[str, str]:
        result = {"answer": "", "reasoning": ""}
        response = response.strip()
        lines = response.split('\n', 1)
        first_line = lines[0].strip()
        
        if first_line in ['A', 'B', 'C', 'D']:
            result["answer"] = first_line
            result["reasoning"] = lines[1].strip() if len(lines) > 1 else ""
            return result
        
        for char in response:
            if char in 'ABCD':
                result["answer"] = char
                parts = response.split(char, 1)
                result["reasoning"] = parts[1].strip() if len(parts) > 1 else response
                break
        
        return result
    
    def close(self):
        self.neo4j_client.close()

class BatchQuestionProcessor:
    def __init__(self):
        self.rag_engine = SimpleRAGEngine()
        self.correct_count = 0
        self.processed_count = 0
    
    def read_input_file(self, input_file: str) -> List[Dict]:
        file_ext = os.path.splitext(input_file)[1].lower()
        
        if file_ext == '.json':
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'questions' in data:
                return data['questions']
            return []
        
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
            questions_data = []
            
            for idx, row in df.iterrows():
                questions_data.append({
                    "id": idx + 1,
                    "question": str(row.get('question', '')).strip(),
                    "options": {
                        'A': str(row.get('A', '')).strip(),
                        'B': str(row.get('B', '')).strip(),
                        'C': str(row.get('C', '')).strip(),
                        'D': str(row.get('D', '')).strip(),
                    },
                    "correct_answer": str(row.get('correct_answer', '')).strip().upper()
                })
            
            return questions_data
        
        return []
    
    def _calculate_current_accuracy(self) -> float:
        if self.processed_count == 0:
            return 0.0
        return (self.correct_count / self.processed_count) * 100
    
    def _save_intermediate_results(self, output_file: str, results: List[Dict], processed_count: int, total_questions: int):
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        temp_output = output_file.replace('.json', '_temp.json')
        
        intermediate_data = {
            "intermediate_save": {
                "processed_so_far": processed_count,
                "total_questions": total_questions,
                "progress": f"{processed_count}/{total_questions}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": results
        }
        
        with open(temp_output, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, ensure_ascii=False, indent=2)
        
        current_correct = sum(1 for r in results if r['rag_result']['is_correct'])
        current_accuracy = (current_correct / processed_count) * 100 if processed_count > 0 else 0
        
        print(f"  Saved progress ({processed_count}/{total_questions}), current accuracy: {current_accuracy:.2f}%")
    
    def process_batch_questions(self, input_file: str, output_file: str):
        print(f"Reading file: {input_file}")
        print(f"Output file will be saved to: {output_file}")
        
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        questions_data = self.read_input_file(input_file)
        if not questions_data:
            print("Error: No question data found")
            return
        
        total_questions = len(questions_data)
        print(f"Found {total_questions} questions")
        
        results = []
        
        for idx, question_data in enumerate(questions_data):
            question_id = question_data.get('id', idx + 1)
            question = question_data.get('question', '').strip()
            options = question_data.get('options', {})
            correct_answer = question_data.get('correct_answer', '').strip().upper()
            
            if not question:
                print(f"Warning: Question {question_id} is empty, skipping")
                continue
            
            print(f"Processing question {question_id}: {question[:50]}...")
            
            try:
                rag_result = self.rag_engine.generate_rag_answer(question, options)
                is_correct = rag_result["rag_result"]["answer"].upper() == correct_answer
                
                self.processed_count += 1
                if is_correct:
                    self.correct_count += 1
                
                current_accuracy = self._calculate_current_accuracy()
                
                result = {
                    "id": question_id,
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer,
                    "rag_result": {
                        **rag_result["rag_result"],
                        "is_correct": is_correct
                    }
                }
                
                results.append(result)
                
                if question_id % 5 == 0:
                    self._save_intermediate_results(output_file, results, question_id, total_questions)
                
                answer_char = rag_result["rag_result"]["answer"]
                status = "✓" if is_correct else "✗"
                print(f"  Answer: {answer_char} {status}, current accuracy: {current_accuracy:.2f}%")
                
                if idx < len(questions_data) - 1:
                    time.sleep(2)
                
            except Exception as e:
                print(f"Failed to process question {question_id}: {e}")
                self.processed_count += 1
                results.append({
                    "id": question_id,
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer,
                    "rag_result": {
                        "answer": "",
                        "reasoning": f"Processing failed: {str(e)[:100]}",
                        "is_correct": False,
                        "context_info": {}
                    }
                })
        
        final_correct = sum(1 for r in results if r['rag_result']['is_correct'])
        final_total = len(results)
        final_accuracy = final_correct / final_total * 100 if final_total > 0 else 0
        
        final_output = {
            "summary": {
                "total_questions_processed": final_total,
                "correct_answers_count": final_correct,
                "accuracy_percentage": round(final_accuracy, 2),
                "accuracy_fraction": f"{final_correct}/{final_total}",
                "model_name": Config.LLM_MODEL,
                "processing_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "search_method": "Two-path search (Keyword + Cypher)",
                "system_version": "1.0 - Two-path lite version"
            },
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        temp_file = output_file.replace('.json', '_temp.json')
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print(f"\n" + "="*60)
        print(f"✓ Processing completed!")
        print(f"  Total questions: {final_total}")
        print(f"  Correct answers: {final_correct}")
        print(f"  Accuracy: {final_accuracy:.2f}% ({final_correct}/{final_total})")
        print(f"  Search method: Two-path search (Keyword + Cypher)")
        print(f"  Results saved to: {output_file}")
        print("="*60)
        
        return final_output
    
    def close(self):
        self.rag_engine.close()

def parse_api_keys(primary_key: str = "", key_list: str = "") -> List[str]:
    keys = []
    if key_list:
        keys.extend([key.strip() for key in key_list.split(",") if key.strip()])
    if primary_key:
        keys.append(primary_key.strip())
    return keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run KG-enhanced RAG question answering on a batch of questions."
    )
    parser.add_argument("--input-file", required=True, help="Input question file in JSON, XLSX, or XLS format.")
    parser.add_argument("--output-file", required=True, help="Output JSON file for RAG answers and evaluation results.")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", Config.LLM_MODEL), help="LLM model name.")
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", Config.LLM_BASE_URL), help="LLM API base URL.")
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""), help="Single LLM API key.")
    parser.add_argument(
        "--llm-api-keys",
        default=os.getenv("LLM_API_KEYS", ""),
        help="Comma-separated LLM API keys. Values from --llm-api-key and --llm-api-keys are combined.",
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", Config.NEO4J_URI), help="Neo4j Bolt URI.")
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", Config.NEO4J_USER), help="Neo4j user name.")
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", Config.NEO4J_PASSWORD), help="Neo4j password.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file

    Config.LLM_MODEL = args.llm_model
    Config.LLM_BASE_URL = args.llm_base_url
    Config.NEO4J_URI = args.neo4j_uri
    Config.NEO4J_USER = args.neo4j_user
    Config.NEO4J_PASSWORD = args.neo4j_password

    configured_keys = parse_api_keys(args.llm_api_key, args.llm_api_keys)
    if configured_keys:
        Config.LLM_API_KEYS = configured_keys
    if not any(key.strip() for key in Config.LLM_API_KEYS):
        raise ValueError("No LLM API key provided. Use --llm-api-key, --llm-api-keys, LLM_API_KEY, or LLM_API_KEYS.")

    print("="*60)
    print("Batch Question RAG Processing Tool - Two-path Lite Version")
    print(f"LLM Model: {Config.LLM_MODEL}")
    print(f"Search Method: Two-path search (Keyword Search + Cypher Condition Search)")
    print("="*60)
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    if not os.path.exists(input_file):
        print(f"Error: File does not exist - {input_file}")
        return
    
    processor = BatchQuestionProcessor()
    
    try:
        final_result = processor.process_batch_questions(input_file, output_file)
        
        if final_result:
            stats = final_result['summary']
            print("\n" + "="*60)
            print("Processing Results Summary:")
            print(f"  Total questions: {stats['total_questions_processed']}")
            print(f"  Correct answers: {stats['correct_answers_count']}")
            print(f"  Accuracy: {stats['accuracy_percentage']}% ({stats['accuracy_fraction']})")
            print(f"  LLM Model: {stats['model_name']}")
            print(f"  Search method: {stats.get('search_method', 'Unknown')}")
            print(f"  Completion time: {stats['processing_completed_at']}")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\nUser interrupted processing")
    except Exception as e:
        print(f"\nProcessing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.close()

if __name__ == "__main__":
    try:
        import pandas
        from neo4j import GraphDatabase
        import requests
    except ImportError as e:
        print(f"Please install dependencies: pip install pandas neo4j requests")
        exit(1)
    
    main()