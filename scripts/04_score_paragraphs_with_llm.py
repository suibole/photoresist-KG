import argparse
import os
import time
import uuid
import requests
import json
from typing import List, Dict, Any
import random

BASE_URL = "https://uni-api.cstcloud.cn/v1/chat/completions"
API_KEYS = [
    ""
]
MODEL_NAME = "deepseek-v3:671b"

def get_random_api_key():
    """Randomly select an API key."""
    return random.choice(API_KEYS)

def s1_chat(prompt, history=None, model="deepseek-v3:671b", stream=False, timeout=120, max_retries=5):
    """
    Enhanced API call function with better error handling and retry mechanism.
    """
    messages = list(history or [])
    messages.append({"role": "user", "content": prompt})

    payload = {
        "id": str(uuid.uuid4()),
        "stream": stream,
        "model": model,
        "messages": messages,
        "max_tokens": 500,  
        "temperature": 0.1,   
    }

    for attempt in range(max_retries):
        api_key = get_random_api_key()  
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            print(f"Attempt {attempt + 1}, using configured API key")

            resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
            
            if resp.status_code == 200:
                data = resp.json()
                print("Request successful, parsing response...")

                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]
                    elif "delta" in choice and "content" in choice["delta"]:
                        return choice["delta"]["content"]
                return str(data)
                
            elif resp.status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited, waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
                continue
                
            elif resp.status_code >= 500:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Server error {resp.status_code}, waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
                continue
                
            else:
                print(f"HTTP error {resp.status_code}: {resp.text[:200]}")
                break
                
        except requests.exceptions.Timeout:
            print(f"Request timeout, retry {attempt + 1}...")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            continue
            
        except requests.exceptions.ConnectionError:
            print(f"Connection error, retry {attempt + 1}...")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            continue
            
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}, retry {attempt + 1}...")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            continue

    return None 

def create_scoring_prompt(paragraph_content: str) -> str:
    prompt = f"""You are an expert in semiconductor materials.
Please score the following paragraph according to the four dimensions below and return **only** a JSON object with the exact keys: `relevance`, `info_density`, `structure`, `factual`.
Use the scoring rules and the illustrative examples that follow.

Scoring rules
1. relevance (0-1):
   - 0.0: no photoresist-related content at all.
   - 0.5-0.7: incidental or background mention of photoresist.
   - 0.8-1.0: photoresist is the main topic.
   (Any paragraph containing the word "photoresist", "SU-8", "edge-bead", "PAG", "resist", etc. must receive ≥0.5.)

2. info_density (0-1):
   - 0.0-0.3: empty or purely qualitative fluff.
   - 0.4-0.6: ordinary technical description, no numbers.
   - 0.7-1.0: concrete data, quantities, or verifiable details.

3. structure (0-1):
   - 0.0-0.3: chaotic, impossible to extract triplets.
   - 0.4-0.6: narrative, requires effort to locate facts.
   - 0.7-1.0: clear cause-effect or parameter-value-unit presentation.

4. factual (0-1):
   - 0.0-0.3: opinions, predictions, unreferenced claims.
   - 0.4-0.6: reasonable inferences.
   - 0.7-1.0: experimental data, cited references, or physical laws.

Illustrative examples (use them as calibration anchors, not as literals to copy):

Example 1:
"To prepare the substrates, first of all, microscope glass slides were cut into pieces of an appropriate size. The glass slides were put for sonication in acetone for 15 min. After that the slides were put for sonication in ethanol and then deionized(DI) water for 15 min each. At the end the slides were thoroughly rinsed with DI water and put in the oven for drying. After cooling the slides to room temperature, positive photoresist BP212 was spin coated at a speed of 5000 r/min for a thickness of about 1 μm. After the spin coating the substrates were prebaked on a hot plate at 110 °C for 60 s. Before exposure, the substrates were allowed to cool down to the room temperature."
→ {{"relevance":1.00, "info_density":0.85, "structure":0.90, "factual":0.90}}

Example 2:
"Two test masks are fabricated and equipped with thermocouples. Based on this verification, the numerical and experimental results are used to model thermal deformations under exposure conditions to be expected at SyLMAND for practical applications. As a result, thermal deformations associated with polyimide mask membranes should become predictable. Future intensity chopper settings could be determined to limit the deformations to a level deemed acceptable for a given application."
→ {{"relevance":0.10, "info_density":0.50, "structure":0.75, "factual":0.75}}

Example 3:
"In this work, we proposed a LEE enhancement strategy to rough the surface of the Si3N4 passivation layer by dry etching using a self-masking technology with carbonized photoresist. The self-masking technology involves the development of mask patterns naturally, followed by their transfer to the target layer. When the ion bombardment energy is too high or too long, the photoresist (PR) is prone to wrinkling and carbonization. Previously, wrinkling and carbonization were considered PR defects that should be avoided in general chip processes. However, every question has two sides. The heavily carbonized PR always has a rough nano/micro-mountain texture. As a result, we attempted for the first time to use this phenomenon to achieve the nano-scale roughened surface of LEDs without nano-scale photolithography."
→ {{"relevance":0.90, "info_density":0.85, "structure":0.75, "factual":0.80}}

Example 4:
"A second effect is associated with the self-focusing of light in a medium whose refractive index increases with photopolymerization. This effect has not been studied much.7,8 The self-focusing effect was observed earlier in Refs. 5 and 9 when a monomeric composite was photopolymerized by light issuing from the end of an optical fiber immersed in a liquid composite with a positive sign of the variation of the refractive index during polymerization."
→ {{"relevance":0.60, "info_density":0.50, "structure":0.70, "factual":0.75}}

Example 5:
"The loss per length of the waveguides was extracted from the slope of the transmission loss as a function of waveguide length, as shown in Fig. 11. Averaging both sets of data for each frequency, values of 1.79, 1.54, and 1.29 dB/mm were calculated for 2.56, 2.84, and 3.11 THz, respectively, as shown in Fig. 12. The estimated accuracy of these calculations is ±0.1 dB/mm. These values are consistent with the trends, as shown in Fig. 7, and the computational results published by Zhou and Lucyszyn [17], but the loss is roughly double the expected values. There are several possible sources for the higher loss, including nonideal gold conductivity, surface roughness, photoresist residues on the waveguide walls, and residual Ti on the waveguide walls. When the loss is normalized per wavelength, the measured losses were approximately 0.20, 0.16, and 0.12 dB/λ, comparing favorably to previously reported measurements of 0.2 dB/λ at 100 GHz [4] and 0.6 dB/λ [5] at 300 GHz."
→ {{"relevance":0.15, "info_density":0.9, "structure":0.80, "factual":0.85}}

Example 6:
"Fig. 14.LER/LWR average values in percentage. The values were obtained from a Power Spectral Density (PSD) analysis done on 2.7 μm length lines using 76 square scanned CD-SEM images."
→ {{"relevance":0.4, "info_density":0.35, "structure":0.6, "factual":0.70}}

Example 7:
"In this paper we report on fabrication of a nanocomposite based on CdSe quantum dots mixed with commercial photoresist ORMOCOMP and proved its high structurability by direct laser writing. The distribution of quantum dots was visualised by transmission electron microscopy and the quality and geometrical parameters of the structures were studied by optical and atomic force microscopy. We manufactured a novel photonic device for Bloch surface electromagnetic waves in photonic crystals and thoroughly studied their propagation by both leakage microscopy and back focal plane imaging methods. By z-scan method we measured the nonlinear Kerr coefficient of quantum dots. Its high value makes the manufactured photonic device promising for all-optical switching applications."
→ {{"relevance":0.8, "info_density":0.6, "structure":0.7, "factual":0.70}}

Example 8:
"Positive photoresist BP-212(Beijing Institute of Chemical Reagents, China) was used for making the grating structures. After the exposure to the two-beam interference, the substrates were developed in 2% aqueous solution of NaOH (Beijing Chemical Works, China). Rhodamine 6G(R6G, Shanghai Chemical Reagent Company, China) was used as analyte to test the performance of SERS active substrates."
→ {{"relevance":0.9, "info_density":0.6, "structure":0.8, "factual":0.80}}

Example 9:
"Spectral and beam power calculations have been performed in part by using the LEX-D simulation tool developed by Stewart K. Griffiths and colleagues at Sandia National Laboratories, Livermore."
→ {{"relevance":0.00, "info_density":0.20, "structure":0.20, "factual":0.65}}

Example 10:
"Supplementary data associated with this article can be found, in the online version, at doi:10.1016/j.electacta.2009.12.024."
→ {{"relevance":0.00, "info_density":0.00, "structure":0.00, "factual":0.00}}

Paragraph to score (do not truncate or paraphrase):
{paragraph_content}

Output (JSON only, no extra words):
{{"relevance":<score>, "info_density":<score>, "structure":<score>, "factual":<score>}}
Do not include any explanation or markdown code block.
"""
    return prompt

def parse_model_response(response: str) -> Dict[str, float]:
    """Enhanced response parsing function."""
    if not response:
        return None
    
    try:
        response = response.strip()
        if response.startswith('{') and response.endswith('}'):
            result = json.loads(response)
            scores = {}
            for key in ['relevance', 'info_density', 'structure', 'factual']:
                if key in result:
                    score = float(result[key])
                    scores[key] = max(0.0, min(1.0, score))
                else:
                    print(f"Missing score field in response: {key}")
                    return None
            return scores
    except json.JSONDecodeError:
        pass
    import re
    numbers = re.findall(r"0\.\d+", response)
    if len(numbers) >= 4:
        try:
            return {
                "relevance": float(numbers[0]),
                "info_density": float(numbers[1]),
                "structure": float(numbers[2]),
                "factual": float(numbers[3])
            }
        except:
            pass
    
    print(f"Unable to parse response, marking paragraph as failed. Response: {response[:100]}...")
    return None

def score_paragraphs(paragraphs: List[Dict], output_file: str, start_index: int = 0):
    """Batch score paragraphs with breakpoint resume support."""
    results = []
    failed_file = output_file.replace('.json', '_failed.json') if output_file.endswith('.json') else f"{output_file}_failed.json"
    failed_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing scoring results")
        except:
            print("Unable to load existing results file, will restart")
    if os.path.exists(failed_file):
        try:
            with open(failed_file, 'r', encoding='utf-8') as f:
                failed_results = json.load(f)
            print(f"Loaded {len(failed_results)} existing failed samples")
        except:
            print("Unable to load existing failed samples file, will restart")
    
    for i in range(start_index, len(paragraphs)):
        paragraph = paragraphs[i]
        print(f"Processing paragraph {i+1}/{len(paragraphs)}...")
        
        content = paragraph.get('content', '')
        if not content.strip():
            print(f"Paragraph {i+1} is empty, skipping...")
            results.append({
                "paragraph": content,
                "labels": {"relevance": 0.0, "info_density": 0.0, "structure": 0.0, "factual": 0.0}
            })
            continue
        if i < len(results):
            print(f"Paragraph {i+1} already processed, skipping...")
            continue

        prompt = create_scoring_prompt(content)
        
        try:
            response = s1_chat(prompt, model=MODEL_NAME, timeout=120, max_retries=3)
            
            if response is None:
                print(f"Paragraph {i+1} request failed, saving to failed samples")
                failed_results.append({
                    "paragraph": content,
                    "status": "failed",
                    "reason": "llm_request_failed"
                })
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_results, f, ensure_ascii=False, indent=2)
                continue
            else:
                scores = parse_model_response(response)
                if scores is None:
                    print(f"Paragraph {i+1} response parsing failed, saving to failed samples")
                    failed_results.append({
                        "paragraph": content,
                        "status": "failed",
                        "reason": "response_parse_failed",
                        "raw_response": response
                    })
                    with open(failed_file, 'w', encoding='utf-8') as f:
                        json.dump(failed_results, f, ensure_ascii=False, indent=2)
                    continue
            results.append({
                "paragraph": content,
                "labels": scores
            })
            
            print(f"Paragraph {i+1} scoring completed: {scores}")
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1} paragraphs, saving progress...")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print("Progress saved")
            
            delay = random.uniform(2, 5)
            time.sleep(delay)
                
        except Exception as e:
            print(f"Error processing paragraph {i+1}: {e}")
            failed_results.append({
                "paragraph": content,
                "status": "failed",
                "reason": "processing_exception",
                "error": str(e)
            })
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_results, f, ensure_ascii=False, indent=2)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Scoring completed! Results saved to: {output_file}")
    return results

def parse_api_keys(primary_key: str = "", key_list: str = "") -> List[str]:
    keys = []
    if key_list:
        keys.extend([key.strip() for key in key_list.split(",") if key.strip()])
    if primary_key:
        keys.append(primary_key.strip())
    return keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score literature paragraphs with an LLM for DeBERTa training labels."
    )
    parser.add_argument("--input-file", required=True, help="Input JSON file containing paragraphs.")
    parser.add_argument("--output-file", required=True, help="Output JSON file for scored paragraphs.")
    parser.add_argument("--start-index", type=int, default=0, help="Paragraph index to start from when resuming.")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", MODEL_NAME), help="LLM model name.")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", BASE_URL), help="LLM chat completions endpoint.")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="Single LLM API key.")
    parser.add_argument(
        "--api-keys",
        default=os.getenv("LLM_API_KEYS", ""),
        help="Comma-separated LLM API keys. Values from --api-key and --api-keys are combined.",
    )
    return parser.parse_args()


def main():
    global BASE_URL, API_KEYS, MODEL_NAME

    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    start_index = args.start_index
    BASE_URL = args.base_url
    MODEL_NAME = args.model

    configured_keys = parse_api_keys(args.api_key, args.api_keys)
    if configured_keys:
        API_KEYS = configured_keys
    if not any(key.strip() for key in API_KEYS):
        raise ValueError("No LLM API key provided. Use --api-key, --api-keys, LLM_API_KEY, or LLM_API_KEYS.")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            paragraphs = json.load(f)
        
        print(f"Successfully read {len(paragraphs)} paragraphs")
        print(f"Starting from paragraph {start_index + 1}")
        results = score_paragraphs(paragraphs, output_file, start_index)
        if results:
            total_paragraphs = len(results)
            avg_scores = {
                "relevance": sum(r["labels"]["relevance"] for r in results) / total_paragraphs,
                "info_density": sum(r["labels"]["info_density"] for r in results) / total_paragraphs,
                "structure": sum(r["labels"]["structure"] for r in results) / total_paragraphs,
                "factual": sum(r["labels"]["factual"] for r in results) / total_paragraphs
            }
            
            print("\nScoring statistics:")
            for dimension, score in avg_scores.items():
                print(f"{dimension}: {score:.3f}")
                
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except json.JSONDecodeError:
        print(f"JSON file format error: {input_file}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()