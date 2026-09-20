# Photoresist Knowledge Graph and RAG Pipeline

Welcome to the Photoresist Knowledge Graph project, a domain-specific knowledge graph and retrieval-augmented generation (RAG) pipeline developed for organizing and querying knowledge from photoresist literature.

## Project Overview

This project uses large language models (LLMs), a DeBERTa-based paragraph-value assessment model, semantic normalization, and Neo4j to transform unstructured scientific literature into an attribute-enhanced knowledge graph.

The workflow was applied to nearly 12,000 full-text photoresist publications. The resulting knowledge graph reported in the associated manuscript contains approximately:

- 320,000 entities;
- 178,000 relations;
- 1,172,000 attribute entries;
- 600,000 valid attribute-enhanced triples.

The knowledge graph was integrated with an LLM through a dual-path RAG strategy combining keyword retrieval and Cypher-based graph querying. On a benchmark of 150 expert-designed multiple-choice questions, the KG-enhanced system achieved an overall accuracy of 95.33%, compared with 76.67% for the LLM-only baseline.

## Key Features

### Paragraph-Value Assessment

An LLM-guided weak-supervision strategy assigns paragraph-level scores across four dimensions:

- topic relevance;
- information density;
- logical structure;
- factual accuracy.

A DeBERTa regression model is then trained to identify high-value paragraphs before knowledge extraction.

### Attribute-Enhanced Triple Extraction

The extraction pipeline represents knowledge using an attribute-enhanced triple structure. In addition to subject, relation, and object fields, the representation preserves numerical values, units, experimental conditions, processing parameters, and performance metrics.

Example:

```json
{
  "start_node": "photoresist film",
  "sn_attribute": ["thickness: 1 um"],
  "relationship": "requires_prebake_condition",
  "re_attribute": ["temperature: 110 C", "time: 60 s"],
  "end_node": "thermal stabilization",
  "en_attribute": [],
  "doi": "10.xxxx/example",
  "title": "Source article title"
}
```

### Semantic Normalization

Entity and relation expressions are embedded using `paraphrase-mpnet-base-v2` and normalized through cosine-similarity-based clustering. The released configuration uses a similarity threshold of 0.9.

### Knowledge-Graph Question Answering

The RAG system combines two retrieval paths:

- keyword-based retrieval for broad semantic coverage;
- dynamically generated Cypher queries for graph-structured retrieval.

Retrieved triples are converted into evidence context for the LLM to support traceable, knowledge-grounded answers.

### Ablation Experiments

## Ablation Experiments

Five separately executable ablation scripts are provided to evaluate the effects of retrieval pathway and triple representation. Together with the complete dual-path attribute-enhanced RAG system, these experiments form a 3 × 2 configuration matrix combining three retrieval modes with two evidence representations.

| Script | Retrieval pathway | Evidence representation |
|---|---|---|
| `10_keyword_only.py` | Keyword retrieval only | Attribute-enhanced triples |
| `11_cypher_only.py` | Cypher retrieval only | Attribute-enhanced triples |
| `12_dual_stan.py` | Keyword + Cypher retrieval | Standard triples |
| `13_keyword_only_standard.py` | Keyword retrieval only | Standard triples |
| `14_cypher_only_standard.py` | Cypher retrieval only | Standard triples |

The complete dual-path attribute-enhanced configuration is implemented in `09_run_kg_rag_qa.py`. The LLM-only baseline is implemented in `08_run_llm_baseline_qa.py`.

All configurations use the same 150-question benchmark, answering LLM, prompt format, and evaluation protocol. In the standard-triple conditions, node and relation attributes are removed from the evidence supplied to the answering LLM, leaving only head–relation–tail triples.


### Ablation Results

| Experimental setting | Accuracy | Improvement over LLM-only (pp) | Reduction from dual-path attribute-enhanced RAG (pp) |
|---|---:|---:|---:|
| LLM-only baseline | 76.67% | — | 18.66 |
| Keyword-only + attribute-enhanced triples | 92.00% | 15.33 | 3.33 |
| Keyword-only + standard triples | 91.33% | 14.66 | 4.00 |
| Cypher-only + attribute-enhanced triples | 89.33% | 12.66 | 6.00 |
| Cypher-only + standard triples | 90.67% | 14.00 | 4.66 |
| Dual-path + standard triples | 89.33% | 12.66 | 6.00 |
| Dual-path attribute-enhanced RAG | 95.33% | 18.66 | — |

Here, “pp” denotes percentage points. The dual-path attribute-enhanced RAG system achieved the highest accuracy. Attribute enhancement produced its largest benefit under dual-path retrieval, whereas the representation effects in the two single-path settings were smaller and nonuniform. The results therefore indicate that retrieval pathway and evidence representation interact, with the strongest performance obtained when dual-path retrieval is combined with attribute-enhanced triples.

## Repository Layout

```text
.
|-- README.md
|-- LICENSE
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
|   |-- 09_run_kg_rag_qa.py
|   |-- 10_keyword_only.py
|   |-- 11_cypher_only.py
|   |-- 12_dual_stan.py
|   |-- keyword_only_standard.py
|   `-- cypher_only_standard.py
|-- DeBERTa_train/DeBERTa_train/
|   |-- README.md
|   |-- configs/base.yaml
|   |-- requirements.txt
|   `-- src/
|-- example_extracted_triples_v1.0.zip
|-- photoresist_paragraph_value_dataset_v1.0.zip
|-- photoresist_qa_benchmark_v1.0.zip
`-- Q&A.zip



## Released Data

### 1. Paragraph-Value Assessment Dataset

`photoresist_paragraph_value_dataset_v1.0.zip` contains 10,000 weakly labeled paragraph records used to train and evaluate the paragraph-value assessment model.

The predefined split is:

| Split | Number of records | Percentage |
|---|---:|---:|
| Training | 8,000 | 80% |
| Validation | 1,000 | 10% |
| Test | 1,000 | 10% |

Each record contains paragraph text and four continuous labels:

```json
{
  "paragraph": "Paragraph text ...",
  "labels": {
    "relevance": 0.85,
    "info_density": 0.75,
    "structure": 0.70,
    "factual": 0.90
  }
}
```

The released training, validation, and test files should be used without repartitioning when reproducing the reported paragraph-model evaluation.

### 2. Extracted Triple Examples

`example_extracted_triples_v1.0.zip` contains representative attribute-enhanced triples generated by the extraction pipeline. Records include source identifiers such as DOI and article title when available.

This archive is a representative example subset. It is not the complete photoresist knowledge graph and does not contain the complete set of approximately 600,000 triples reported in the manuscript.

### 3. Photoresist QA Benchmark

`photoresist_qa_benchmark_v1.0.zip` contains the 150-question multiple-choice benchmark used in the manuscript.

The benchmark covers four categories:

- material formulations;
- process control;
- lithography technology;
- integrated applications.

Each benchmark record contains:

- a stable question ID;
- a category label;
- the question text;
- four answer options;
- the expert-assigned correct option.

Example:

```json
{
  "question_id": "Q001",
  "category": "process_control",
  "question": "Question text ...",
  "options": {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  },
  "correct_option": "B"
}
```

If model-response files are released, their exact filenames and contents should be listed here. Do not claim that LLM-only or KG-RAG predictions are included unless those files are present in the repository.

## Environment Requirements

Python 3.10 or later is recommended.

Install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r DeBERTa_train/DeBERTa_train/requirements.txt
```

On Windows PowerShell, activate the environment using:

```powershell
.venv\Scripts\Activate.ps1
```

Additional software and services required for the complete pipeline include:

- MinerU for PDF-to-Markdown conversion;
- GNU Parallel for parallel conversion with the supplied shell script;
- Neo4j 5.x for graph storage and retrieval;
- an OpenAI-compatible LLM API for weak labeling, triple extraction, and QA.

## Configuration

The current pipeline scripts use configuration placeholders defined inside the scripts. They do not currently expose a unified command-line interface, and most scripts do not automatically read a `.env` file.

Before running the pipeline, provide the required paths and service parameters in the corresponding files:

| Component | Settings to configure |
|---|---|
| `01_pdf_to_markdown_mineru.sh` | `INPUT_BASE_DIR`, `OUTPUT_BASE_DIR`, and storage-directory layout |
| `02_markdown_to_paragraph_json.py` | Input and output directories |
| `03_filter_short_paragraphs.py` | `SRC_DIR`, `DST_DIR`, and `SHORT_THRESHOLD` |
| `04_score_paragraphs_with_llm.py` | API endpoint, API key, model name, and input/output paths |
| `05_extract_attribute_triples.py` | API configuration, `JSON_DIR`, `OUTPUT_DIR`, and `NO_ATTRIBUTES_DIR` |
| `06_semantic_normalize_triples.py` | Embedding configuration, `INPUT_DIR`, and `OUTPUT_DIR` |
| `07_import_triples_to_neo4j.py` | Input path and Neo4j connection settings |
| `08_run_llm_baseline_qa.py` | API endpoint, API key, model name, and QA input/output paths |
| `09_run_kg_rag_qa.py` | API configuration, Neo4j connection, and QA input/output paths |
| `10_keyword_only.py` | Keyword-only retrieval with attribute-enhanced triples |
| `11_cypher_only.py` | Cypher-only retrieval with attribute-enhanced triples |
| `12_dual_stan.py` | Dual-path retrieval with standard triples |
| `13_keyword_only_standard.py` | Keyword-only retrieval with standard triples |
| `14_cypher_only_standard.py` | Cypher-only retrieval with standard triples |

Do not commit real API keys, Neo4j passwords, access tokens, publisher-controlled PDF files, or other credentials.

## Running the Pipeline

After configuring each script, execute the stages sequentially:

```text
01  PDF-to-Markdown conversion
02  Markdown-to-paragraph JSON conversion
03  paragraph filtering
04  LLM-assisted weak labeling
05  attribute-enhanced triple extraction
06  semantic normalization
07  Neo4j graph import
08  LLM-only QA evaluation
09  KG-enhanced RAG evaluation
```

For example:

```bash
bash scripts/01_pdf_to_markdown_mineru.sh
python scripts/02_markdown_to_paragraph_json.py
python scripts/03_filter_short_paragraphs.py
python scripts/04_score_paragraphs_with_llm.py
python scripts/05_extract_attribute_triples.py
python scripts/06_semantic_normalize_triples.py
python scripts/07_import_triples_to_neo4j.py
python scripts/08_run_llm_baseline_qa.py
python scripts/09_run_kg_rag_qa.py
```

These commands assume that the required path and service placeholders have already been configured.

### Running the Ablation Experiments

The five ablation experiments can be run independently:

```bash
python scripts/10_keyword_only.py \
  --input path/to/qa_benchmark.json \
  --output-dir outputs/keyword_only \
  --analysis-cache outputs/shared/query_analysis_cache.json

python scripts/11_cypher_only.py \
  --input path/to/qa_benchmark.json \
  --output-dir outputs/cypher_only \
  --analysis-cache outputs/shared/query_analysis_cache.json

python scripts/12_dual_stan.py \
  --input path/to/qa_benchmark.json \
  --output-dir outputs/dual_standard \
  --analysis-cache outputs/shared/query_analysis_cache.json

python scripts/13_keyword_only_standard.py \
  --input path/to/qa_benchmark.json \
  --output-dir outputs/keyword_only_standard \
  --analysis-cache outputs/shared/query_analysis_cache.json

python scripts/14_cypher_only_standard.py \
  --input path/to/qa_benchmark.json \
  --output-dir outputs/cypher_only_standard \
  --analysis-cache outputs/shared/query_analysis_cache.json

## Training the Paragraph-Value Model

The released training module is located in `DeBERTa_train/DeBERTa_train/`.

Before training, edit `configs/base.yaml` and replace the original machine-specific absolute paths with valid local or relative paths:

```yaml
model_name: microsoft/deberta-v3-base
seed: 42

train_file: data/paragraph_value_train.json
val_file: data/paragraph_value_validation.json
test_file: data/paragraph_value_test.json

training:
  output_dir: outputs/base_retrain
  num_epochs: 25
  train_batch_size: 8
  eval_batch_size: 8
  learning_rate: 1.2e-5
```

Then run:

```bash
cd DeBERTa_train/DeBERTa_train
python -m src.training.train --config configs/base.yaml
```

Generate predictions using a trained checkpoint:

```bash
python -m src.cli.run_predict \
  --config configs/base.yaml \
  --checkpoint outputs/base_retrain/best \
  --input path/to/paragraph_json \
  --output outputs/predictions
```

The model identifier in `configs/base.yaml`, the associated manuscript, and any released checkpoint must refer to the same DeBERTa variant. The current released configuration specifies `microsoft/deberta-v3-base`; update either the configuration or the manuscript if another variant was used in the reported experiments.

## Data and Software Availability

The source code and released datasets described above are available in this GitHub repository:

<https://github.com/suibole/photoresist-KG>

The full-text literature corpus is not redistributed because publisher licences may restrict redistribution. Users must obtain source articles through lawful open-access or institution-authorized routes.

The public release contains the paragraph-value assessment dataset, the photoresist QA benchmark, and representative extracted triple examples. The representative triple archive is not the complete knowledge graph. Consequently, this repository supports inspection and reproduction of the released components but does not independently contain the complete set of approximately 600,000 triples reported in the manuscript.Five ablation scripts used to compare retrieval pathways and triple representations are included in the scripts/ directory. The corresponding per-question outputs and summary results are provided in Q&A.zip. The archive covers the LLM-only baseline, the complete dual-path attribute-enhanced RAG system, and the five ablated configurations reported above.

Any paragraph text released in the training dataset should be limited to content that the authors are permitted to redistribute. Source identifiers and provenance metadata should be retained whenever possible.

## Security and Repository Hygiene

Before publishing or updating the repository, verify that it does not contain:

- API keys, passwords, tokens, cookies, or populated `.env` files;
- publisher-controlled full-text PDF files;
- machine-specific absolute paths that expose local usernames or server directories;
- Neo4j credentials or database dumps containing restricted content;
- `__pycache__/`, `.pyc`, test caches, logs, or temporary output files.

## License and Citation

See `LICENSE` for the repository licence.

If you use this repository or the released datasets, please cite the associated manuscript and reference this GitHub repository:

```text
Code and released data: https://github.com/suibole/photoresist-KG
```

## Contributions

Issues and pull requests are welcome. When reporting a problem, please include the affected script, software environment, relevant configuration, and a minimal reproducible example without credentials or publisher-restricted content.
