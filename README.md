# Photoresist Knowledge Graph and RAG Pipeline

This repository contains research code for constructing a photoresist-domain knowledge graph from scientific literature and evaluating retrieval-augmented question answering over that graph.

The workflow covers literature conversion, paragraph scoring, attribute-aware triple extraction, semantic normalization, Neo4j graph import, and comparison between an LLM-only baseline and a knowledge-graph RAG pipeline.

## Pipeline Overview

```text
PDF literature
  -> Markdown conversion with MinerU
  -> paragraph-level JSON construction
  -> paragraph filtering
  -> LLM-assisted paragraph value scoring
  -> DeBERTa paragraph scoring model training or inference
  -> attribute-aware knowledge triple extraction
  -> semantic entity and relation normalization
  -> Neo4j graph import
  -> LLM-only and KG-RAG question answering evaluation
```

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- scripts/
|   |-- 01_pdf_to_markdown_mineru.sh
|   |-- 02_markdown_to_paragraph_json.py
|   |-- 03_filter_short_paragraphs.py
|   |-- 04_score_paragraphs_with_llm.py
|   |-- 05_extract_attribute_triples.py
|   |-- 06_semantic_normalize_triples.py
|   |-- 07_import_triples_to_neo4j.py
|   |-- 08_run_llm_baseline_qa.py
|   `-- 09_run_kg_rag_qa.py
`-- DeBERTa_train/
    `-- DeBERTa_train/
        |-- README.md
        |-- configs/base.yaml
        |-- requirements.txt
        `-- src/
```

## Main Components

### Corpus Processing

- `scripts/01_pdf_to_markdown_mineru.sh` converts PDF files to Markdown using MinerU.
- `scripts/02_markdown_to_paragraph_json.py` converts Markdown documents into paragraph-level JSON records.
- `scripts/03_filter_short_paragraphs.py` removes short, low-information, keyword-only, and table-like paragraphs.

### Paragraph Scoring

- `scripts/04_score_paragraphs_with_llm.py` uses an LLM to assign weak labels for paragraph value dimensions.
- `DeBERTa_train/DeBERTa_train/` contains the DeBERTa-based regression model used to predict paragraph value scores.

### Knowledge Graph Construction

- `scripts/05_extract_attribute_triples.py` extracts attribute-aware triples from selected paragraphs.
- `scripts/06_semantic_normalize_triples.py` normalizes entity and relation surface forms with sentence embeddings.
- `scripts/07_import_triples_to_neo4j.py` imports normalized triples into Neo4j.

### QA Evaluation

- `scripts/08_run_llm_baseline_qa.py` evaluates a direct LLM-only multiple-choice QA baseline.
- `scripts/09_run_kg_rag_qa.py` evaluates a two-path KG-RAG workflow using keyword retrieval and generated Cypher conditions.

## Environment Setup

Use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The DeBERTa training module has additional development and training dependencies:

```bash
pip install -r DeBERTa_train/DeBERTa_train/requirements.txt
```

Additional runtime services are required for the full pipeline:

- MinerU for PDF-to-Markdown conversion.
- GNU Parallel for parallel PDF conversion in `01_pdf_to_markdown_mineru.sh`.
- Neo4j 5.x for graph storage and retrieval.
- Access to an OpenAI-compatible LLM API for paragraph scoring, triple extraction, and QA.

## Configuration

Do not commit credentials, private corpus files, generated model checkpoints, or Neo4j database dumps.

Create a local `.env` file from `.env.example` or set equivalent environment variables in your shell:

```bash
cp .env.example .env
```

The research scripts use command-line arguments for input and output paths. API credentials and service endpoints can be supplied either through command-line arguments or environment variables. Keep local credentials in `.env` or your shell environment, and do not commit `.env` to Git.

Common settings include:

- `LLM_BASE_URL`: base URL for the OpenAI-compatible API.
- `LLM_API_KEY` or `LLM_API_KEYS`: API key or comma-separated key list.
- `LLM_MODEL`: model name used by the API.
- `NEO4J_URI`: Neo4j Bolt URI, for example `bolt://localhost:7687`.
- `NEO4J_USER`: Neo4j user name.
- `NEO4J_PASSWORD`: Neo4j password.

## Expected Data Formats

### Paragraph JSON

Most downstream scripts expect paragraph records in one of the following forms:

```json
[
  {
    "index": 1,
    "content": "Paragraph text ..."
  }
]
```

or:

```json
{
  "metadata": {
    "title": "Paper title",
    "doi": "10.xxxx/example"
  },
  "abstract": "Abstract text ...",
  "paragraphs": [
    {
      "index": 1,
      "content": "Paragraph text ..."
    }
  ]
}
```

### Paragraph Scoring Labels

The DeBERTa scoring model expects supervised records with continuous labels:

```json
[
  {
    "paragraph": "Paragraph text ...",
    "labels": {
      "relevance": 0.85,
      "info_density": 0.75,
      "structure": 0.70,
      "factual": 0.90
    }
  }
]
```

### Attribute-Aware Triple JSON

Triple extraction and graph import use the following schema:

```json
[
  {
    "start_node": "photoresist film",
    "sn_attribute": ["thickness: 1 um"],
    "relationship": "requires_prebake_condition",
    "re_attribute": ["temperature: 110 C", "time: 60 s"],
    "end_node": "thermal stabilization",
    "en_attribute": []
  }
]
```

## Typical Usage

Run each stage with explicit input and output paths. Replace the example paths with your local data locations.

```bash
bash scripts/01_pdf_to_markdown_mineru.sh \
  --input-base-dir data/pdfs \
  --output-base-dir outputs/markdown

python scripts/02_markdown_to_paragraph_json.py \
  --input-dir outputs/markdown \
  --output-dir outputs/paragraph_json

python scripts/03_filter_short_paragraphs.py \
  --input-dir outputs/paragraph_json/auto \
  --output-dir outputs/filtered_paragraphs

python scripts/04_score_paragraphs_with_llm.py \
  --input-file outputs/filtered_paragraphs/example_cleaned.json \
  --output-file outputs/scored_paragraphs.json \
  --api-key "$LLM_API_KEY"

python scripts/05_extract_attribute_triples.py \
  --input-dir outputs/filtered_paragraphs \
  --output-dir outputs/triples \
  --api-key "$LLM_API_KEY"

python scripts/06_semantic_normalize_triples.py \
  --input-dir outputs/triples \
  --output-dir outputs/normalized_triples

python scripts/07_import_triples_to_neo4j.py \
  --triples-path outputs/normalized_triples \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"

python scripts/08_run_llm_baseline_qa.py \
  --input-file data/questions.json \
  --output-file outputs/llm_baseline_answers.json \
  --api-key "$LLM_API_KEY"

python scripts/09_run_kg_rag_qa.py \
  --input-file data/questions.json \
  --output-file outputs/kg_rag_answers.json \
  --llm-api-key "$LLM_API_KEY" \
  --neo4j-password "$NEO4J_PASSWORD"
```

Train the DeBERTa scoring model:

```bash
cd DeBERTa_train/DeBERTa_train
python -m src.training.train --config configs/base.yaml
```

Generate predictions with a trained checkpoint:

```bash
python -m src.cli.run_predict \
  --config configs/base.yaml \
  --checkpoint outputs/base_retrain/best \
  --input path/to/paragraph_json \
  --output outputs/predictions
```

## Data and Model Availability

The full-text literature corpus is not included because publisher licenses may restrict redistribution. Large intermediate files, generated outputs, DeBERTa checkpoints, and Neo4j database files should be archived separately, for example on Zenodo, Hugging Face, or an institutional repository.

This repository is intended to provide the executable research pipeline and expected data schemas. To reproduce reported results, users must provide the literature corpus, local API credentials, Neo4j instance, and any released model checkpoints.

## Git Hygiene

Before publishing this repository, verify that the following files are not committed:

- `.env` or any file containing API keys, passwords, cookies, or tokens.
- Full-text PDFs and publisher-controlled literature.
- Large generated JSON outputs, model checkpoints, embeddings, and Neo4j dumps.
- Python bytecode caches such as `__pycache__/` and `*.pyc`.
- Local logs, temporary conversion outputs, and failed-PDF copies.

The included `.gitignore` provides conservative defaults for these categories.

## License and Citation

This repository is provided for academic peer review, manuscript evaluation, and reproducibility inspection only. See `LICENSE` for details.

If this code supports a manuscript, add the manuscript citation here once it is available.
