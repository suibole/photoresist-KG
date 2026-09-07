"""Ablation 1: keyword-only retrieval with attribute-enhanced triples.

This is a standalone script.  It does not import any other local experiment
file.  Set LLM_API_KEY and NEO4J_PASSWORD in the environment before running.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
CONDITION_NAME = "keyword_attribute_enhanced"


def _json_from_response(text: str) -> Optional[Dict[str, Any]]:
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
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {LLM_API_KEY}",
                    },
                    json={
                        "id": str(uuid.uuid4()),
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": LLM_TEMPERATURE,
                        "max_tokens": 4000,
                        "stream": False,
                    },
                    timeout=180,
                )
                if response.status_code == 200:
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"]).strip()
                if response.status_code not in {429, 500, 502, 503}:
                    response.raise_for_status()
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
                if attempt == retries:
                    break
            time.sleep(min(5 * (2**attempt), 60))
        return "(LLM call failed)"


ANALYSIS_PROMPT = """Analyze the following photoresist question and return JSON only.
Question: {question}

Return exactly these fields:
{{
  "keywords": ["keyword1", "keyword2"],
  "cypher_condition": "a condition expression",
  "query_type": "...",
  "language": "english"
}}

The Neo4j pattern is fixed as
(n:SuiboleEntity)-[r:SUIBOLE_RELATION]->(m:SuiboleEntity).
The only node properties are n.name, m.name, n.attributes, and m.attributes.
The only relationship properties are r.type and r.attributes. Never use
r.name, description, property, text, or label. Descriptive searches must use
ANY(a IN coalesce(n.attributes, []) WHERE toLower(toString(a)) CONTAINS 'term')
or the analogous expression for r or m. Use CONTAINS. Return only the
expression after WHERE; do not return MATCH, WHERE, RETURN, LIMIT, markdown,
or the prefix HERE."""


def analyze_question(llm: LLMClient, question: str) -> Dict[str, Any]:
    value = _json_from_response(llm.chat(ANALYSIS_PROMPT.format(question=question)))
    if not value:
        return {"keywords": [], "cypher_condition": "", "query_type": "fallback", "language": "english"}
    keywords = value.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = [keywords]
    keywords = [str(x).strip() for x in keywords if str(x).strip()][:5]
    return {
        "keywords": keywords,
        "cypher_condition": str(value.get("cypher_condition", "") or ""),
        "query_type": str(value.get("query_type", "keyword") or "keyword"),
        "language": str(value.get("language", "english") or "english"),
    }


def load_analysis_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_analysis_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_condition(condition: str) -> str:
    """Normalize an LLM condition without permitting write operations."""
    text = str(condition or "").replace("\\n", " ").replace("\n", " ").strip()
    text = re.sub(r"```(?:cypher)?|```", "", text, flags=re.I).strip()
    text = re.sub(r"^HERE\b", "WHERE", text, flags=re.I).strip()
    if re.match(r"^MATCH\b", text, flags=re.I):
        found = re.search(r"\bWHERE\b(.*)$", text, flags=re.I | re.S)
        text = found.group(1).strip() if found else ""
    text = re.sub(r"^WHERE\b", "", text, count=1, flags=re.I).strip()
    text = re.split(r"\bRETURN\b|\bLIMIT\b", text, maxsplit=1, flags=re.I)[0].strip()
    if not text or re.search(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD\s+CSV|UNION)\b|;", text, flags=re.I):
        return ""

    attr_pattern = re.compile(
        r"toLower\(\s*([nrm])\.(description|property|text|label)\s*\)\s*"
        r"CONTAINS\s*(['\"][^'\"]+['\"])",
        flags=re.I,
    )
    text = attr_pattern.sub(
        lambda m: (
            f"ANY(a IN coalesce({m.group(1)}.attributes, []) WHERE "
            f"toLower(toString(a)) CONTAINS {m.group(3)})"
        ),
        text,
    )
    text = re.sub(r"toLower\(\s*r\.name\s*\)", "toLower(coalesce(r.type, ''))", text, flags=re.I)
    return text


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

    def keyword_search(self, keywords: List[str]) -> Tuple[List[Dict[str, Any]], str]:
        keywords = [str(x).strip().lower() for x in keywords if str(x).strip()][:5]
        if not keywords:
            return [], "no_keywords"
        conditions, params = [], {}
        for i, keyword in enumerate(keywords):
            key = f"kw{i}"
            params[key] = keyword
            conditions.append(
                f"(toLower(coalesce(n.name, '')) CONTAINS ${key} OR "
                f"toLower(coalesce(m.name, '')) CONTAINS ${key} OR "
                f"toLower(coalesce(r.type, '')) CONTAINS ${key} OR "
                f"ANY(a IN coalesce(n.attributes, []) WHERE toLower(toString(a)) CONTAINS ${key}) OR "
                f"ANY(a IN coalesce(r.attributes, []) WHERE toLower(toString(a)) CONTAINS ${key}) OR "
                f"ANY(a IN coalesce(m.attributes, []) WHERE toLower(toString(a)) CONTAINS ${key}))"
            )
        query = f"""MATCH (n:SuiboleEntity)-[r:SUIBOLE_RELATION]->(m:SuiboleEntity)
WHERE {' OR '.join(conditions)}
RETURN n.name AS head, n.attributes AS head_attrs,
       r.type AS relation_type, r.attributes AS rel_attrs,
       type(r) AS relation_label, m.name AS tail, m.attributes AS tail_attrs
LIMIT 15"""
        rows, status = self._run(query, params)
        return self._normalize(rows), status

    def close(self) -> None:
        self.driver.close()


def read_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data if isinstance(data, list) else data.get("questions", [])
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path).to_dict(orient="records")
    else:
        raise ValueError("Input must be JSON, XLSX, or XLS")
    questions = []
    for i, item in enumerate(raw, 1):
        options = item.get("options") or {k: item.get(k, "") for k in "ABCD"}
        questions.append({
            "id": item.get("id", i),
            "question": str(item.get("question", "")).strip(),
            "options": {k: str(options.get(k, "")).strip() for k in "ABCD"},
            "correct_answer": str(item.get("correct_answer", item.get("answer", ""))).strip().upper(),
        })
    return [x for x in questions if x["question"]]


def format_attributes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def build_context(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No relevant technical knowledge found."
    lines = ["Retrieved knowledge:"]
    for i, row in enumerate(rows[:8], 1):
        line = f"{i}. {row.get('head', 'unknown')}"
        head_attrs = format_attributes(row.get("head_attrs"))
        rel_attrs = format_attributes(row.get("rel_attrs"))
        tail_attrs = format_attributes(row.get("tail_attrs"))
        if head_attrs:
            line += f" [{head_attrs}]"
        line += f" -> [{row.get('relation', 'unknown')}"
        if rel_attrs:
            line += f" ({rel_attrs})"
        line += f"] -> {row.get('tail', 'unknown')}"
        if tail_attrs:
            line += f" [{tail_attrs}]"
        lines.append(line)
    return "\n".join(lines)


def parse_answer(text: str) -> str:
    first = (text or "").strip().splitlines()
    if first and first[0].strip().upper() in {"A", "B", "C", "D"}:
        return first[0].strip().upper()
    match = re.search(r"(?:^|[\s\(])([ABCD])(?:[\).:\s]|$)", (text or "").upper())
    return match.group(1) if match else ""


def answer_question(llm: LLMClient, question: str, options: Dict[str, str], rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    option_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    prompt = f"""You are an expert in photoresist materials and photolithography. Answer this single-answer multiple-choice question.
Question: {question}
Options:
{option_text}

Retrieved knowledge:
{build_context(rows)}

Return one answer letter (A, B, C, or D) on the first line, followed by a concise explanation."""
    raw = llm.chat(prompt)
    return parse_answer(raw), raw


def save_results(path: Path, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "results": records}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword-only attribute-enhanced ablation")
    parser.add_argument("--input", required=True, help="150-question JSON/XLSX file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--analysis-cache", default="query_analysis_cache.json", help="Shared analysis cache")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    questions = read_questions(Path(args.input).resolve())
    cache_path = Path(args.analysis_cache).resolve()
    analysis_cache = load_analysis_cache(cache_path)
    llm, graph = LLMClient(), Neo4jClient()
    records, correct = [], 0
    try:
        for index, item in enumerate(questions, 1):
            question = item["question"]
            print(f"[{CONDITION_NAME}] {index}/{len(questions)}: {question[:70]}")
            if question not in analysis_cache:
                analysis_cache[question] = analyze_question(llm, question)
                save_analysis_cache(cache_path, analysis_cache)
            analysis = analysis_cache[question]
            rows, status = graph.keyword_search(analysis.get("keywords", []))
            predicted, raw = answer_question(llm, question, item["options"], rows)
            is_correct = predicted == item["correct_answer"]
            correct += int(is_correct)
            records.append({**item, "result": {
                "answer": predicted, "reasoning": raw, "is_correct": is_correct,
                "retrieval": "keyword_only", "attribute_enhanced": True,
                "keyword_count": len(rows), "keyword_status": status,
                "keywords": analysis.get("keywords", []), "rows": rows,
            }})
            if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
                save_results(Path(args.output_dir) / f"{CONDITION_NAME}_checkpoint.json", {"processed": index}, records)
    finally:
        graph.close()

    summary = {
        "condition": CONDITION_NAME, "total_questions": len(records),
        "correct_answers": correct,
        "accuracy_percentage": round(100 * correct / len(records), 2) if records else 0.0,
        "model_name": LLM_MODEL, "retrieval": "keyword_only",
        "attribute_enhanced_context": True,
    }
    save_results(Path(args.output_dir) / f"{CONDITION_NAME}.json", summary, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()