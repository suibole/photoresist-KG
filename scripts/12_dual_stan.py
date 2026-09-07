"""Ablation 3: dual-path retrieval with standard triples.

This standalone script executes both keyword and Cypher retrieval.  It removes
all node/relationship attributes before building the answer context and before
saving the retrieved triples, so the representation shown to the answer model
is strictly head-relation-tail.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://llmapi.paratera.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "DeepSeek-V3.2-Instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MIN_API_INTERVAL = float(os.getenv("MIN_API_CALL_INTERVAL", "1.0"))
CONDITION_NAME = "dual_standard"


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"```(?:json)?|```", "", text or "", flags=re.I).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


class LLMClient:
    def __init__(self) -> None:
        if not LLM_API_KEY:
            raise RuntimeError("Please set the environment variable LLM_API_KEY first")
        self._last_call = 0.0

    def chat(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(retries + 1):
            wait = MIN_API_INTERVAL - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                response = requests.post(
                    f"{LLM_BASE_URL}/chat/completions",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
                    json={"id": str(uuid.uuid4()), "model": LLM_MODEL,
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": LLM_TEMPERATURE, "max_tokens": 4000, "stream": False},
                    timeout=180,
                )
                if response.status_code == 200:
                    return str(response.json()["choices"][0]["message"]["content"]).strip()
                if response.status_code not in {429, 500, 502, 503}:
                    response.raise_for_status()
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
                if attempt == retries:
                    break
            time.sleep(min(5 * (2**attempt), 60))
        return "(LLM call failed)"


ANALYSIS_PROMPT = """Analyze the following photoresist question and return JSON only.
Question: {question}
{{"keywords":["keyword1"],"cypher_condition":"condition expression","query_type":"...","language":"english"}}
The fixed graph pattern is (n:SuiboleEntity)-[r:SUIBOLE_RELATION]->(m:SuiboleEntity).
Nodes have only name and attributes; relationships have only type and attributes.
Never use r.name, description, property, text, or label. For descriptive content,
use ANY(a IN coalesce(n.attributes, []) WHERE toLower(toString(a)) CONTAINS 'term')
or the analogous expression for r or m. Use CONTAINS. Return only the expression
after WHERE, with no MATCH, WHERE, RETURN, LIMIT, markdown, or HERE."""


def analyze_question(llm: LLMClient, question: str) -> Dict[str, Any]:
    value = parse_json_response(llm.chat(ANALYSIS_PROMPT.format(question=question)))
    if not value:
        return {"keywords": [], "cypher_condition": "", "query_type": "fallback", "language": "english"}
    keywords = value.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = [keywords]
    return {"keywords": [str(x).strip() for x in keywords if str(x).strip()][:5],
            "cypher_condition": str(value.get("cypher_condition", "") or ""),
            "query_type": str(value.get("query_type", "dual") or "dual"),
            "language": str(value.get("language", "english") or "english")}


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_condition(condition: str) -> str:
    text = str(condition or "").replace("\\n", " ").replace("\n", " ").strip()
    text = re.sub(r"```(?:cypher)?|```", "", text, flags=re.I).strip()
    text = re.sub(r"^HERE\b", "WHERE", text, flags=re.I).strip()
    if re.match(r"^MATCH\b", text, flags=re.I):
        found = re.search(r"\bWHERE\b(.*)$", text, flags=re.I | re.S)
        text = found.group(1).strip() if found else ""
    text = re.sub(r"^WHERE\b", "", text, count=1, flags=re.I).strip()
    text = re.split(r"\bRETURN\b|\bLIMIT\b", text, maxsplit=1, flags=re.I)[0].strip()
    if (not text or text.lower() in {"1=1", "true"} or
            re.search(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD\s+CSV|UNION)\b|;", text, flags=re.I)):
        return ""
    pattern = re.compile(
        r"toLower\(\s*([nrm])\.(description|property|text|label)\s*\)\s*CONTAINS\s*(['\"][^'\"]+['\"])", flags=re.I)
    text = pattern.sub(lambda m: f"ANY(a IN coalesce({m.group(1)}.attributes, []) WHERE toLower(toString(a)) CONTAINS {m.group(3)})", text)
    return re.sub(r"toLower\(\s*r\.name\s*\)", "toLower(coalesce(r.type, ''))", text, flags=re.I)


class Neo4jClient:
    def __init__(self) -> None:
        if not NEO4J_PASSWORD:
            raise RuntimeError("Please set the environment variable NEO4J_PASSWORD first")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], str]:
        try:
            with self.driver.session() as session:
                return [dict(record) for record in session.run(query, params or {})], "ok"
        except Exception as exc:
            print(f"[Neo4j warning] {exc}")
            return [], str(exc)

    @staticmethod
    def _normalize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for row in rows:
            row["relation"] = row.get("relation_type") or row.get("relation_label") or "unknown"
        return rows

    def keyword_search(self, keywords: Iterable[str]) -> Tuple[List[Dict[str, Any]], str]:
        words = [str(x).strip().lower() for x in keywords if str(x).strip()][:5]
        if not words:
            return [], "no_keywords"
        clauses, params = [], {}
        for i, word in enumerate(words):
            key = f"kw{i}"; params[key] = word
            clauses.append(
                f"(toLower(coalesce(n.name, '')) CONTAINS ${key} OR toLower(coalesce(m.name, '')) CONTAINS ${key} OR "
                f"toLower(coalesce(r.type, '')) CONTAINS ${key} OR "
                f"ANY(a IN coalesce(n.attributes, []) WHERE toLower(toString(a)) CONTAINS ${key}) OR "
                f"ANY(a IN coalesce(r.attributes, []) WHERE toLower(toString(a)) CONTAINS ${key}) OR "
                f"ANY(a IN coalesce(m.attributes, []) WHERE toLower(toString(a)) CONTAINS ${key}))"
            )
        query = f"""MATCH (n:SuiboleEntity)-[r:SUIBOLE_RELATION]->(m:SuiboleEntity)
WHERE {' OR '.join(clauses)}
RETURN n.name AS head, n.attributes AS head_attrs, r.type AS relation_type,
       r.attributes AS rel_attrs, type(r) AS relation_label,
       m.name AS tail, m.attributes AS tail_attrs LIMIT 15"""
        rows, status = self._run(query, params)
        return self._normalize(rows), status

    def cypher_search(self, condition: str) -> Tuple[List[Dict[str, Any]], str]:
        condition = clean_condition(condition)
        if not condition:
            return [], "invalid_or_empty_condition"
        query = f"""MATCH (n:SuiboleEntity)-[r:SUIBOLE_RELATION]->(m:SuiboleEntity)
WHERE {condition}
RETURN n.name AS head, n.attributes AS head_attrs, r.type AS relation_type,
       r.attributes AS rel_attrs, type(r) AS relation_label,
       m.name AS tail, m.attributes AS tail_attrs LIMIT 15"""
        rows, status = self._run(query)
        return self._normalize(rows), status

    def close(self) -> None:
        self.driver.close()


def merge_interleaved(keyword_rows: List[Dict[str, Any]], cypher_rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Deduplicate while alternating paths so both can reach the top-8 context."""
    output, seen = [], set()
    for i in range(max(len(keyword_rows), len(cypher_rows))):
        for group in (keyword_rows, cypher_rows):
            if i >= len(group):
                continue
            row = group[i]
            key = (row.get("head", ""), row.get("relation", ""), row.get("tail", ""))
            if key not in seen:
                seen.add(key); output.append(row)
                if len(output) >= limit:
                    return output
    return output


def standardize(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"head": str(x.get("head", "unknown")),
             "relation": str(x.get("relation") or x.get("relation_type") or x.get("relation_label") or "unknown"),
             "tail": str(x.get("tail", "unknown"))} for x in rows]


def read_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file(): raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8")); raw = data if isinstance(data, list) else data.get("questions", [])
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path).to_dict(orient="records")
    else: raise ValueError("Input must be JSON, XLSX, or XLS")
    out = []
    for i, item in enumerate(raw, 1):
        options = item.get("options") or {k: item.get(k, "") for k in "ABCD"}
        out.append({"id": item.get("id", i), "question": str(item.get("question", "")).strip(),
                    "options": {k: str(options.get(k, "")).strip() for k in "ABCD"},
                    "correct_answer": str(item.get("correct_answer", item.get("answer", ""))).strip().upper()})
    return [x for x in out if x["question"]]


def context(rows: List[Dict[str, str]]) -> str:
    if not rows: return "No relevant technical knowledge found."
    return "Retrieved knowledge:\n" + "\n".join(f"{i}. {x['head']} -> [{x['relation']}] -> {x['tail']}" for i, x in enumerate(rows[:8], 1))


def parse_answer(text: str) -> str:
    lines = (text or "").strip().splitlines()
    if lines and lines[0].strip().upper() in "ABCD": return lines[0].strip().upper()
    match = re.search(r"(?:^|[\s\(])([ABCD])(?:[\).:\s]|$)", (text or "").upper())
    return match.group(1) if match else ""


def save(path: Path, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "results": records}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-path standard-triple ablation")
    ap.add_argument("--input", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--analysis-cache", default="query_analysis_cache.json"); ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()
    questions = read_questions(Path(args.input).resolve())
    cache_path = Path(args.analysis_cache).resolve(); cache = load_cache(cache_path)
    llm, graph, records, correct = LLMClient(), Neo4jClient(), [], 0
    try:
        for i, item in enumerate(questions, 1):
            q = item["question"]; print(f"[{CONDITION_NAME}] {i}/{len(questions)}: {q[:70]}")
            if q not in cache:
                cache[q] = analyze_question(llm, q); save_cache(cache_path, cache)
            info = cache[q]
            kw, kw_status = graph.keyword_search(info.get("keywords", []))
            cy, cy_status = graph.cypher_search(info.get("cypher_condition", ""))
            raw_rows = merge_interleaved(kw, cy)
            rows = standardize(raw_rows)
            options = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
            raw = llm.chat(f"""You are an expert in photoresist materials and photolithography. Answer this single-answer multiple-choice question.
Question: {q}
Options:
{options}

Retrieved knowledge:
{context(rows)}

Return one answer letter (A, B, C, or D) on the first line, followed by a concise explanation.""")
            pred = parse_answer(raw); ok = pred == item["correct_answer"]; correct += int(ok)
            records.append({**item, "result": {"answer": pred, "reasoning": raw, "is_correct": ok,
                "retrieval": "dual", "attribute_enhanced": False,
                "keyword_count": len(kw), "cypher_count": len(cy),
                "keyword_status": kw_status, "cypher_status": cy_status,
                "cypher_condition": info.get("cypher_condition", ""), "rows": rows}})
            if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
                save(Path(args.output_dir) / f"{CONDITION_NAME}_checkpoint.json", {"processed": i}, records)
    finally:
        graph.close()
    summary = {"condition": CONDITION_NAME, "total_questions": len(records), "correct_answers": correct,
               "accuracy_percentage": round(100 * correct / len(records), 2) if records else 0.0,
               "model_name": LLM_MODEL, "retrieval": "dual", "attribute_enhanced_context": False}
    save(Path(args.output_dir) / f"{CONDITION_NAME}.json", summary, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()