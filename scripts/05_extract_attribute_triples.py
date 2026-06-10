import argparse
import os
import time
import json
import glob
import re
from typing import List, Dict, Any
from collections import OrderedDict
from openai import OpenAI
import concurrent.futures

PARALLEL_API_CONFIG_1 = {
    "name": "Parallel-API-1",
    "base_url": os.getenv("LLM_BASE_URL_1", os.getenv("LLM_BASE_URL", "https://llmapi.paratera.com/v1")),
    "api_key": os.getenv("LLM_API_KEY_1", os.getenv("LLM_API_KEY", "")),
    "model": os.getenv("LLM_MODEL_1", os.getenv("LLM_MODEL", "Qwen3-Next-80B-A3B-Instruct"))
}

PARALLEL_API_CONFIG_2 = {
    "name": "Parallel-API-2",
    "base_url": os.getenv("LLM_BASE_URL_2", os.getenv("LLM_BASE_URL", "https://llmapi.paratera.com/v1")),
    "api_key": os.getenv("LLM_API_KEY_2", os.getenv("LLM_API_KEY", "")),
    "model": os.getenv("LLM_MODEL_2", os.getenv("LLM_MODEL", "Qwen3-Next-80B-A3B-Instruct"))
}

JSON_DIR = None
OUTPUT_DIR = None
NO_ATTRIBUTES_DIR = None

SYSTEM_PROMPT = """You are an expert in knowledge graph construction. Extract precise triples with STRONG CONSTRAINT ATTRIBUTES from research literature.

CRITICAL REQUIREMENTS:
1. **Critical Format Requirements**
   - Output MUST be a single, complete JSON array.  
   - Forbidden to use any field names other than:  
     "start_node", "sn_attribute", "relationship", "re_attribute", "end_node", "en_attribute".  
   - If you use "name", "value", "title", "context", or any other field, the response will be rejected.  
   - Every opening bracket/quotes must have its closing counterpart; truncate content if necessary to stay within 1000 tokens.

2. **STRONG CONSTRAINT ATTRIBUTES ONLY**:+++++++++
   - Extract ONLY attributes that define ESSENTIAL CHARACTERISTICS
   - SKIP descriptive/supplier/location attributes
   - INCLUDE: dimensions, thickness, performance metrics, physical properties
   - EXCLUDE: manufacturer names, supplier locations, generic descriptions

3. **NUMERICAL ACCURACY**:
   - Extract ALL numerical values EXPLICITLY mentioned in source text
   - NEVER infer or calculate values
   - ALWAYS include units with numerical values (nm, μm, ms, °C, etc.)
   - PRESERVE exact values, ranges, and approximation terms (e.g., "≈50", ">1 μm", "<100 nm")

4. **SPECIFIC RELATIONSHIP TYPES**:
   - Use precise, mechanism-driven relationship verbs
   - AVOID generic "compared_with", "has_component"
   - Examples: "enhances_detection_sensitivity", "requires_thickness_constraint", "increases_surface_area_for"

5. **DEDUPLICATION AWARENESS**:
   - Merge similar technical information into comprehensive triples
   - Avoid extracting redundant entity-relationship pairs
   - Combine related parameter information

6. **TECHNICAL SPECIFICITY**:
   - Focus on parameters that define system behavior
   - Include: geometric dimensions, process conditions, performance limits
   - Preserve exact technical specifications

LEARNING EXAMPLES:

【Example 1 - Strong constraint attributes】
Source Text: "the nanopillars were cone shaped, about 500 nm wide at the base and 1 μm high in an irregular lattice"
Poor Triple:
{
  "start_node": "nanostructures",
  "sn_attribute": [],
  "relationship": "compared_with", 
  "re_attribute": [],
  "end_node": "planar sample plates",
  "en_attribute": []
}
Good Triple:
{
  "start_node": "nanopillar_geometry",
  "sn_attribute": ["base_diameter: 500 nm", "height: 1 μm", "packing_density: irregular"],
  "relationship": "enhances_SALDI_signal_compared_to", 
  "re_attribute": ["signal_performance: stronger"],
  "end_node": "planar_surfaces",
  "en_attribute": []
}

【Example 2 - Specific relationships】
Source Text: "The nominal thickness of each coating was between 40 nm and 60 nm"
Poor Triple:
{
  "start_node": "amorphous silicon", 
  "sn_attribute": ["thin film"],
  "relationship": "coated_on",
  "re_attribute": [],
  "end_node": "nonporous hybrid nanopillars",
  "en_attribute": []
}
Good Triple:
{
  "start_node": "amorphous_silicon_coating",
  "sn_attribute": ["thickness_range: 40-60 nm"],
  "relationship": "requires_thickness_constraint_for",
  "re_attribute": ["purpose: SALDI_activity_optimization"],
  "end_node": "nonporous_nanostructured_substrates", 
  "en_attribute": []
}

OUTPUT FORMAT:
[
  {
    "start_node": "...",
    "sn_attribute": ["..."],
    "relationship": "...",
    "re_attribute": ["..."], 
    "end_node": "...",
    "en_attribute": ["..."]
  }
]
DO NOT add markdown code blocks (```).  
DO NOT include any explanation or comments outside the JSON array.  
Output ONLY the JSON array.  """

class StandardFormatSystem:
    def __init__(self, client, model_name, platform_name):
        self.client = client
        self.model_name = model_name
        self.platform_name = platform_name
        self._last_call_time = 0
        self.MIN_API_CALL_INTERVAL = 1.0
    
    def wait_for_cooldown(self):
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < self.MIN_API_CALL_INTERVAL:
            sleep_time = self.MIN_API_CALL_INTERVAL - time_since_last_call
            time.sleep(sleep_time)
        self._last_call_time = time.time()
    
    def extract_triples_from_paragraph(self, paragraph_text: str, abstract: str = "", 
                                      metadata: Dict = None, paragraph_idx: int = 0) -> List[Dict]:
        print(f"      [{self.platform_name}] Processing paragraph {paragraph_idx+1}...")
        
        if abstract and len(abstract.strip()) > 30:
            user_prompt = f"""Quick overview (Abstract):
{abstract}

Target paragraph:
{paragraph_text}

EXTRACTION REQUIREMENTS:
1. **STRONG CONSTRAINT ATTRIBUTES REQUIRED**: 
   - MUST extract numerical parameters, dimensions, performance metrics
   - Include: thickness, temperature, pressure, dimensions, concentrations
   - Preserve exact values with units: nm, μm, °C, %, MPa, etc.

2. **FIELD NAMES MUST BE EXACT**:
   - Use ONLY: "start_node", "sn_attribute", "relationship", "re_attribute", "end_node", "en_attribute"
   - NEVER use: "subject", "object", "name", "value"

3. **ATTRIBUTE FORMAT**:
   - "sn_attribute", "re_attribute", "en_attribute" must be arrays of strings
   - Include numerical details in attributes

Now extract triples with detailed attributes from the target paragraph:"""
        else:
            user_prompt = f"""TEXT TO ANALYZE:
{paragraph_text}

EXTRACTION REQUIREMENTS:
1. **EXTRACT NUMERICAL ATTRIBUTES**: thickness, dimensions, temperatures, pressures, concentrations
2. **PRESERVE UNITS**: nm, μm, °C, %, MPa, mJ/cm², etc.
3. **USE EXACT FIELD NAMES**: "start_node", "sn_attribute", "relationship", "re_attribute", "end_node", "en_attribute"
4. **ATTRIBUTES AS ARRAYS**: Include numerical details in attribute arrays

Extract triples with detailed numerical attributes:"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(3): 
            try:
                self.wait_for_cooldown()
                
                print(f"      [{self.platform_name}] API call attempt {attempt+1}/3...")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.1,
                    timeout=60
                )
                
                response_text = response.choices[0].message.content
                
                if "401 Unauthorized" in response_text or "Invalid API key" in response_text:
                    print(f"      [{self.platform_name}] ❌ Authentication failed")
                    return []
       
                triples = self._safe_json_with_attributes(response_text)
                
                if triples and self._is_valid_triple_json(triples):
                    print(f"      [{self.platform_name}] ✅ Extracted {len(triples)} triples")
                    
                    if triples:
                        print(f"      📋 Extraction results for this paragraph (showing first 3):")
                        display_count = min(3, len(triples))
                        for i in range(display_count):
                            triple = triples[i]
                            start_node = triple.get("start_node", "").strip()
                            relationship = triple.get("relationship", "").strip()
                            end_node = triple.get("end_node", "").strip()
                            
                            print(f"        Triple {i+1}:")
                            print(f"          Start node: {start_node}")
                            print(f"          Relationship: {relationship}")
                            print(f"          End node: {end_node}")
                            
                            sn_attrs = triple.get("sn_attribute", [])
                            if sn_attrs:
                                attrs_str = ", ".join([str(attr).strip() for attr in sn_attrs[:2]])
                                if len(sn_attrs) > 2:
                                    attrs_str += f" ... and {len(sn_attrs)-2} more attributes"
                                print(f"          Start node attributes: {attrs_str}")
                            
                            re_attrs = triple.get("re_attribute", [])
                            if re_attrs:
                                attrs_str = ", ".join([str(attr).strip() for attr in re_attrs[:2]])
                                if len(re_attrs) > 2:
                                    attrs_str += f" ... and {len(re_attrs)-2} more attributes"
                                print(f"          Relationship attributes: {attrs_str}")
                            
                            en_attrs = triple.get("en_attribute", [])
                            if en_attrs:
                                attrs_str = ", ".join([str(attr).strip() for attr in en_attrs[:2]])
                                if len(en_attrs) > 2:
                                    attrs_str += f" ... and {len(en_attrs)-2} more attributes"
                                print(f"          End node attributes: {attrs_str}")
                            
                            print()  
                        if len(triples) > 3:
                            print(f"        ... and {len(triples)-3} more triples not shown")
                    
                    processed_triples = self._post_process_triples(triples)
                    enhanced_triples = []
                    
                    for triple in processed_triples:
                        enhanced_triple = OrderedDict([
                            ("start_node", triple.get("start_node", "")),
                            ("sn_attribute", triple.get("sn_attribute", [])),
                            ("relationship", triple.get("relationship", "")),
                            ("re_attribute", triple.get("re_attribute", [])),
                            ("end_node", triple.get("end_node", "")),
                            ("en_attribute", triple.get("en_attribute", [])),
                            ("doi", metadata.get("doi", "") if metadata else ""),
                            ("title", metadata.get("title", "") if metadata else ""),
                            ("source_file", metadata.get("source_file", "") if metadata else ""),
                            ("processed_by", self.platform_name),
                            ("paragraph_idx", paragraph_idx)
                        ])
                        enhanced_triples.append(enhanced_triple)
                    
                    return enhanced_triples
                else:
                    if attempt < 2:
                        print(f"      [{self.platform_name}] ⚠️ Parsing failed, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"      [{self.platform_name}] ❌ Parsing failed")
                        return []
                        
            except Exception as e:
                error_msg = str(e)
                if attempt < 2:
                    print(f"      [{self.platform_name}] ⚠️ Exception: {error_msg[:100]}, retrying...")
                    time.sleep(2 ** attempt) 
                    continue
                else:
                    print(f"      [{self.platform_name}] ❌ Final failure: {error_msg[:100]}")
                    return []
        
        return []
    
    def _is_valid_triple_json(self, data: Any) -> bool:
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, dict):
                return False
            must_keys = ["start_node", "relationship", "end_node"]
            if any(k not in item for k in must_keys):
                return False
            if "name" in item or "value" in item:
                return False
        return True
    
    def _safe_json_with_attributes(self, raw: str) -> List[Dict]:
        raw = str(raw).strip()
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'\s*```', '', raw)

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                normalized = []
                for item in data:
                    triple = self._normalize_single_triple(item)
                    if triple:
                        normalized.append(triple)
                if normalized:
                    return normalized
        except json.JSONDecodeError:
            pass

        try:
            arr_match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
            if arr_match:
                data = json.loads(arr_match.group())
                if isinstance(data, list):
                    normalized = []
                    for item in data:
                        triple = self._normalize_single_triple(item)
                        if triple:
                            normalized.append(triple)
                    if normalized:
                        return normalized
        except json.JSONDecodeError:
            pass

        triples = []
        for m in re.finditer(r'\{[^{}]*\}(?=\s*[,]?\s*(?:\{|\]))', raw, re.DOTALL):
            try:
                triple_data = json.loads(m.group())
                normalized = self._normalize_single_triple(triple_data)
                if normalized:
                    triples.append(normalized)
            except json.JSONDecodeError:
                continue
        
        return triples
    
    def _normalize_single_triple(self, triple_data: Dict) -> Dict:
        if not isinstance(triple_data, dict):
            return None

        start_node = None
        for key in ['start_node', 'subject', 'head', 'from', 'source']:
            if key in triple_data and triple_data[key]:
                start_node = str(triple_data[key]).strip()
                break
        
        relationship = None
        for key in ['relationship', 'relation', 'predicate', 'edge', 'property']:
            if key in triple_data and triple_data[key]:
                relationship = str(triple_data[key]).strip()
                break
        
        end_node = None
        for key in ['end_node', 'object', 'tail', 'to', 'target']:
            if key in triple_data and triple_data[key]:
                end_node = str(triple_data[key]).strip()
                break
        
        if not (start_node and relationship and end_node):
            return None
        
        def extract_attributes_with_mapping(data, field_names):
            for field in field_names:
                if field in data and data[field]:
                    if isinstance(data[field], list):
                        attrs = [str(attr).strip() for attr in data[field] if attr and str(attr).strip()]
                        if attrs:
                            return attrs
                    else:
                        attr_str = str(data[field]).strip()
                        if attr_str:
                            return [attr_str]
            return []

        sn_attributes = extract_attributes_with_mapping(triple_data, 
            ['sn_attribute', 'subject_attribute', 'head_attribute', 'source_attr'])
        
        re_attributes = extract_attributes_with_mapping(triple_data,
            ['re_attribute', 'relation_attribute', 'predicate_attr', 'edge_attr'])
        
        en_attributes = extract_attributes_with_mapping(triple_data,
            ['en_attribute', 'object_attribute', 'tail_attribute', 'target_attr'])
        
        result = OrderedDict([
            ("start_node", start_node),
            ("sn_attribute", sn_attributes),
            ("relationship", relationship),
            ("re_attribute", re_attributes),
            ("end_node", end_node),
            ("en_attribute", en_attributes)
        ])
        
        return result
    
    def _post_process_triples(self, triples: List[Dict]) -> List[Dict]:
        processed = []
        
        for triple in triples:
            if not (triple.get("start_node") and triple.get("relationship") and triple.get("end_node")):
                continue
            processed.append(triple)
        
        return processed

def extract_abstract_from_json(json_data: Dict) -> str:
    if isinstance(json_data, dict) and "metadata" in json_data and "abstract" in json_data["metadata"]:
        abstract = json_data["metadata"]["abstract"].strip()
        if abstract:
            return abstract
    
    if isinstance(json_data, dict) and "paragraphs" in json_data and len(json_data["paragraphs"]) > 0:
        first_para = json_data["paragraphs"][0].get("content", "").strip()
        if first_para.lower().startswith("abstract:"):
            return first_para[9:].strip()
    
    return ""

def split_paragraphs_for_parallel(paragraphs, num_workers=2):
    chunks = []
    total = len(paragraphs)
    
    chunk_size = total // num_workers
    remainder = total % num_workers
    
    start = 0
    for i in range(num_workers):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(list(enumerate(paragraphs[start:end], start=start)))
        start = end
    
    return chunks

def process_paragraphs_parallel(paragraph_chunk, abstract, metadata, client_config, file_idx, chunk_idx):
    platform_name = client_config["name"]
    print(f"    🚀 Starting {platform_name} processing chunk {chunk_idx+1} ({len(paragraph_chunk)} paragraphs)...")
    
    client = OpenAI(
        base_url=client_config["base_url"],
        api_key=client_config["api_key"]
    )
    
    extraction_system = StandardFormatSystem(
        client=client,
        model_name=client_config["model"],
        platform_name=platform_name
    )
    
    all_triples = []
    
    for para_idx, paragraph in paragraph_chunk:
        if isinstance(paragraph, dict):
            content = paragraph.get("content", "").strip()
        else:
            content = str(paragraph).strip()
        
        if not content:
            continue
        
        global_para_idx = para_idx
        
        triples = extraction_system.extract_triples_from_paragraph(
            content, abstract, metadata, global_para_idx
        )
        
        if triples:
            all_triples.extend(triples)
    
    print(f"    ✅ {platform_name} completed chunk {chunk_idx+1}: {len(all_triples)} triples")
    return all_triples

def process_single_json_file_parallel(json_path: str) -> List[Dict]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"  ❌ Failed to read JSON file: {e}")
        return []

    if isinstance(json_data, list):
        paragraphs = json_data
        metadata = {
            "doi": "",
            "title": os.path.basename(json_path),
            "source_file": os.path.basename(json_path)
        }
        abstract = ""
    elif isinstance(json_data, dict):
        metadata = {
            "doi": json_data.get("doi", ""),
            "title": json_data.get("metadata", {}).get("title", ""),
            "source_file": os.path.basename(json_path)
        }
        abstract = extract_abstract_from_json(json_data)
        paragraphs = json_data.get("paragraphs", [])
    else:
        print(f"  ❌ Unknown JSON structure type: {type(json_data)}")
        return []

    if not paragraphs:
        print(f"  ⚠️ File has no paragraph content")
        return []

    paragraph_contents = []
    for para in paragraphs:
        if isinstance(para, dict):
            content = para.get("content", "").strip()
        else:
            content = str(para).strip()
        if content:
            paragraph_contents.append(content)
    
    if abstract:
        print(f"    📄 Found abstract, length: {len(abstract)} characters")
    
    print(f"    📊 Total paragraphs: {len(paragraph_contents)}")
    
    paragraph_chunks = split_paragraphs_for_parallel(paragraph_contents, num_workers=2)
    
    print(f"    ⚡ Starting 2 parallel API processing...")
    print(f"      Parallel-API-1: {len(paragraph_chunks[0])} paragraphs")
    print(f"      Parallel-API-2: {len(paragraph_chunks[1])} paragraphs")
    
    tasks = [
        (paragraph_chunks[0], abstract, metadata, PARALLEL_API_CONFIG_1, 0, 0),
        (paragraph_chunks[1], abstract, metadata, PARALLEL_API_CONFIG_2, 0, 1)
    ]
    
    all_triples = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_task = {}
        for task_args in tasks:
            future = executor.submit(process_paragraphs_parallel, *task_args)
            future_to_task[future] = task_args[3]["name"]  
        
        for future in concurrent.futures.as_completed(future_to_task):
            platform_name = future_to_task[future]
            try:
                triples = future.result()
                all_triples.extend(triples)
                print(f"    ✅ {platform_name} processing completed")
            except Exception as e:
                print(f"    ❌ {platform_name} processing failed: {str(e)[:100]}")
    
    print(f"    🔄 Total extracted {len(all_triples)} triples")
    all_triples.sort(key=lambda x: x.get("paragraph_idx", 0))
    
    return all_triples

def extract_triples_from_json_files(json_dir: str, output_dir: str, resume: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(NO_ATTRIBUTES_DIR, exist_ok=True)

    json_pattern = os.path.join(json_dir, "*.json")
    json_files = glob.glob(json_pattern)
    if not json_files:
        print(f"❌ No JSON files found in directory {json_dir}")
        return {}

    processed_files = set()
    if resume:
        triples_pattern = os.path.join(output_dir, "triples_*.json")
        for triples_file in glob.glob(triples_pattern):
            original_name = os.path.basename(triples_file)[8:] 
            processed_files.add(original_name)
        print(f"🔍 Resume mode: Found {len(processed_files)} already processed files")

    files_to_process = [f for f in json_files if os.path.basename(f) not in processed_files]
    print(f"🎯 Files to process: {len(files_to_process)}/{len(json_files)}")

    all_results = {}
    for i, json_file in enumerate(files_to_process, 1):
        filename = os.path.basename(json_file)
        print(f"\n【{i:>3}/{len(files_to_process)}】Processing {filename}")
        print("=" * 60)

        triples = process_single_json_file_parallel(json_file)
        all_results[filename] = triples

        output_filename = f"triples_{filename}"
        has_attributes = any(triple.get('sn_attribute') or triple.get('re_attribute') or triple.get('en_attribute')
                             for triple in triples) if triples else False
        save_dir = output_dir if has_attributes else NO_ATTRIBUTES_DIR
        status = "✅ Has attributes" if has_attributes else "❌ No attributes"

        with open(os.path.join(save_dir, output_filename), 'w', encoding='utf-8') as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)
        print(f"    💾 {status} -> {output_filename}")

        if i % 2 == 0 and i < len(files_to_process):
            print("🔄 Processed 2 files, resting for 15 seconds...")
            time.sleep(15)

    return all_results

def parse_args():
    parser = argparse.ArgumentParser(description="Extract attribute-enhanced triples from paragraph JSON files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing paragraph JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted triples with attributes.")
    parser.add_argument("--no-attributes-dir", default=None, help="Directory for files whose extracted triples contain no attributes.")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://llmapi.paratera.com/v1"), help="Default LLM API base URL.")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="Default LLM API key for both parallel clients.")
    parser.add_argument("--api-key-1", default=os.getenv("LLM_API_KEY_1", ""), help="LLM API key for Parallel-API-1.")
    parser.add_argument("--api-key-2", default=os.getenv("LLM_API_KEY_2", ""), help="LLM API key for Parallel-API-2.")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "Qwen3-Next-80B-A3B-Instruct"), help="Default LLM model for both parallel clients.")
    parser.add_argument("--model-1", default=os.getenv("LLM_MODEL_1", ""), help="LLM model for Parallel-API-1.")
    parser.add_argument("--model-2", default=os.getenv("LLM_MODEL_2", ""), help="LLM model for Parallel-API-2.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing triples_*.json outputs.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resume mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    JSON_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    NO_ATTRIBUTES_DIR = args.no_attributes_dir or os.path.join(args.output_dir, "no_attributes")

    PARALLEL_API_CONFIG_1["base_url"] = os.getenv("LLM_BASE_URL_1", args.base_url)
    PARALLEL_API_CONFIG_2["base_url"] = os.getenv("LLM_BASE_URL_2", args.base_url)
    PARALLEL_API_CONFIG_1["api_key"] = args.api_key_1 or args.api_key
    PARALLEL_API_CONFIG_2["api_key"] = args.api_key_2 or args.api_key
    PARALLEL_API_CONFIG_1["model"] = args.model_1 or args.model
    PARALLEL_API_CONFIG_2["model"] = args.model_2 or args.model

    if not PARALLEL_API_CONFIG_1["api_key"] or not PARALLEL_API_CONFIG_2["api_key"]:
        raise ValueError("No LLM API key provided. Use --api-key, --api-key-1/--api-key-2, LLM_API_KEY, or LLM_API_KEY_1/LLM_API_KEY_2.")

    if not os.path.exists(JSON_DIR):
        print(f"鉂?Input directory does not exist: {JSON_DIR}")
        exit(1)
    
    print(f"Input directory: {JSON_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("General Knowledge Extraction System: Enabled")
    print(f"馃 Using model: {PARALLEL_API_CONFIG_1['model']}")
    print(f"鈿?Parallel mode: Two parallel APIs evenly distributing paragraphs")
    print(f"馃攽 API-1: {PARALLEL_API_CONFIG_1['name']}")
    print(f"馃攽 API-2: {PARALLEL_API_CONFIG_2['name']}")
    print("=" * 50)
    
    start_time = time.time()
    results = extract_triples_from_json_files(JSON_DIR, OUTPUT_DIR, resume=args.resume)
    end_time = time.time()
    
    print(f"\n鉁?General knowledge extraction completed!")
    print(f"馃搱 Total time: {end_time - start_time:.2f} seconds")
    print(f"馃搧 Processed {len(results)} files this run")