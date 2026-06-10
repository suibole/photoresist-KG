import argparse
import os
import json
import glob
import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any


CONFIG = {
    "model_path": "paraphrase-mpnet-base-v2",
    "similarity_threshold": 0.9,
    "global_threshold": 0.9,
    "batch_size": 256,
    "max_length": 256
}

INPUT_DIR = None
OUTPUT_DIR = None
CHUNK_SIZE = 5000

class StatsCollector:

    def __init__(self):
        self.stats = {
            'time': {},
            'entity': {
                'before': 0,
                'after': 0,
                'reduction': 0,
                'reduction_rate': 0.0,
                'unique_before': set(),
                'unique_after': set(),
            },
            'relation': {
                'before': 0,
                'after': 0,
                'reduction': 0,
                'reduction_rate': 0.0,
                'unique_before': set(),
                'unique_after': set(),
            },
            'triplet': {
                'total': 0,
                'entity_changes': 0,
                'relation_changes': 0,
                'unchanged': 0,
                'entity_change_rate': 0.0,
                'relation_change_rate': 0.0
            },
            'files': {
                'total': 0,
                'success': 0,
                'failed': 0,
                'processed_files': []
            }
        }

    def start_timer(self, stage: str):
        self.stats['time'][stage] = {'start': time.time()}

    def end_timer(self, stage: str):
        if stage in self.stats['time']:
            self.stats['time'][stage]['end'] = time.time()
            self.stats['time'][stage]['duration'] = (
                self.stats['time'][stage]['end'] - self.stats['time'][stage]['start']
            )


class TripletProcessor:

    def __init__(self, stats_collector: StatsCollector):
        self.stats = stats_collector

    def load_triplets_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            else:
                return [data]
        except Exception as e:
            print(f"  ❌ Failed to load file {os.path.basename(file_path)}: {e}")
            return []

    def load_all_triplets_from_dir(self, input_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        all_triplets = []
        file_paths = []

        json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
        json_files.extend(glob.glob(os.path.join(input_dir, "*.json")))

        json_files = list(set(json_files))

        if not json_files:
            print(f"❌ No JSON files found in directory {input_dir}")
            return [], []

        print(f"📁 Found {len(json_files)} JSON files")

        for json_file in json_files:
            triplets = self.load_triplets_from_file(json_file)
            if triplets:
                all_triplets.extend(triplets)
                file_paths.append(json_file)

        return all_triplets, file_paths

    def extract_terms_from_triplets(self, triplets: List[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
        entities = set()
        relations = set()

        for triple in triplets:
            start_entity = self._get_value(triple, ['start_node', 'start', 'subject', 'head'])
            if start_entity:
                entities.add(start_entity.strip())
            end_entity = self._get_value(triple, ['end_node', 'end', 'object', 'tail'])
            if end_entity:
                entities.add(end_entity.strip())
            relation = self._get_value(triple, ['relationship', 'relation', 'predicate', 'type'])
            if relation:
                relations.add(relation.strip())

        return entities, relations

    @staticmethod
    def _get_value(triple: Dict[str, Any], keys: List[str]) -> str:
        for key in keys:
            if key in triple and triple[key]:
                value = triple[key]
                if isinstance(value, list) and value:
                    for item in value:
                        if item:
                            return str(item).strip()
                elif isinstance(value, dict):
                    return str(value)
                else:
                    return str(value).strip()
        return ""


class SemanticEncoder:

    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config['model_path'])

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        batch_size = self.config['batch_size']
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


class IncrementalStandardizer:
    def __init__(self, model_config):
        self.encoder = SemanticEncoder(model_config)
        self.similarity_threshold = model_config['similarity_threshold']
        self.global_threshold = model_config['global_threshold']

    def standardize_terms(self, terms: Set[str], term_type: str) -> Dict[str, str]:
        if not terms:
            return {}

        print(f"\n{'='*60}")
        print(f"Starting {term_type} standardization ({len(terms)} terms)")
        print(f"{'='*60}")

        term_list = list(terms)

        num_chunks = (len(term_list) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"Processing in {num_chunks} chunks, max {CHUNK_SIZE} terms per chunk")

        chunk_results = []
        for i in range(num_chunks):
            start_idx = i * CHUNK_SIZE
            end_idx = min((i + 1) * CHUNK_SIZE, len(term_list))
            chunk_terms = term_list[start_idx:end_idx]

            print(f"\nProcessing {term_type} chunk {i+1}/{num_chunks} ({len(chunk_terms)} terms)")
            chunk_result = self._standardize_single_chunk(chunk_terms)
            chunk_results.append(chunk_result)

        print(f"\nStarting global integration for {term_type}...")
        final_result = self._integrate_chunks_memory_safe(chunk_results, term_type)

        print(f"{term_type} standardization complete: {len(final_result)} mappings")
        return final_result

    def _standardize_single_chunk(self, terms: List[str]) -> Dict[str, str]:
        if len(terms) <= 1:
            return {term: term for term in terms}

        embeddings = self.encoder.encode_texts_batch(terms)

        n = len(terms)
        similarity_dict = {}

        batch_size = 5000
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)

            if torch.cuda.is_available():
                batch_emb = torch.tensor(embeddings[batch_start:batch_end]).cuda()
                all_emb = torch.tensor(embeddings).cuda()
                sim_batch = torch.mm(batch_emb, all_emb.T)
                sim_batch_np = sim_batch.cpu().numpy()

                del batch_emb, all_emb, sim_batch
                torch.cuda.empty_cache()
            else:
                sim_batch_np = np.dot(embeddings[batch_start:batch_end], embeddings.T)

            for k in range(batch_end - batch_start):
                i = batch_start + k
                for j in range(n):
                    if j > i and sim_batch_np[k, j] >= self.similarity_threshold:
                        similarity_dict[(i, j)] = sim_batch_np[k, j]

        clusters = self._cluster_terms(terms, similarity_dict)

        return self._create_standardization_dict(terms, clusters)

    def _integrate_chunks_memory_safe(self, chunk_results: List[Dict[str, str]], term_type: str) -> Dict[str, str]:
        if not chunk_results:
            return {}

        all_mappings = {}
        for chunk_dict in chunk_results:
            all_mappings.update(chunk_dict)

        all_standards = list(set(all_mappings.values()))

        if len(all_standards) <= 1:
            return all_mappings

        print(f"Integrating {len(all_standards)} standard terms...")

        print("Vectorizing standard terms...")
        embeddings = self.encoder.encode_texts_batch(all_standards)

        clusters = self._incremental_clustering(all_standards, embeddings)

        print("Selecting final standards...")
        standard_mapping = {}
        for cluster_indices in clusters:
            cluster_terms = [all_standards[i] for i in cluster_indices]
            final_std = self._select_final_standard(cluster_terms)
            for term in cluster_terms:
                standard_mapping[term] = final_std

        final_dict = {}
        for original, intermediate_std in all_mappings.items():
            final_dict[original] = standard_mapping.get(intermediate_std, intermediate_std)

        return final_dict

    def _incremental_clustering(self, terms: List[str], embeddings: np.ndarray) -> List[List[int]]:
        n = len(terms)
        if n == 0:
            return []

        clusters = []
        assigned = [False] * n

        batch_size = 1000

        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)
            batch_size_actual = batch_end - i

            for batch_idx in range(batch_size_actual):
                idx = i + batch_idx
                if assigned[idx]:
                    continue

                cluster = [idx]
                assigned[idx] = True

                for cluster_idx, existing_cluster in enumerate(clusters):
                    if not existing_cluster:
                        continue

                    center_idx = existing_cluster[0]

                    if torch.cuda.is_available():
                        vec1 = torch.tensor(embeddings[idx:idx+1]).cuda()
                        vec2 = torch.tensor(embeddings[center_idx:center_idx+1]).cuda()
                        sim = torch.mm(vec1, vec2.T).cpu().numpy()[0, 0]

                        del vec1, vec2
                        torch.cuda.empty_cache()
                    else:
                        sim = np.dot(embeddings[idx:idx+1], embeddings[center_idx:center_idx+1].T)[0, 0]

                    if sim >= self.global_threshold:
                        cluster = existing_cluster + [idx]
                        clusters[cluster_idx] = cluster
                        assigned[idx] = True
                        break

                if not assigned[idx]:
                    clusters.append([idx])
                    assigned[idx] = True

        return clusters

    def _cluster_terms(self, terms: List[str], similarity_dict: Dict[tuple, float]) -> List[List[int]]:
        n = len(terms)
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)
            stack = [i]

            while stack:
                current = stack.pop()
                for j in range(n):
                    if j not in visited and ((current, j) in similarity_dict or (j, current) in similarity_dict):
                        cluster.append(j)
                        visited.add(j)
                        stack.append(j)

            clusters.append(cluster)

        return clusters

    def _create_standardization_dict(self, terms: List[str], clusters: List[List[int]]) -> Dict[str, str]:
        standardization_dict = {}

        for cluster in clusters:
            if not cluster:
                continue

            cluster_terms = [terms[idx] for idx in cluster]
            standard_term = self._select_standard_term(cluster_terms)
            for term in cluster_terms:
                standardization_dict[term] = standard_term

        return standardization_dict

    def _select_standard_term(self, group: List[str]) -> str:
        if not group:
            return ""

        clean_terms = [term for term in group if '_' not in term and '-' not in term and '(' not in term]

        if clean_terms:
            full_word_terms = [term for term in clean_terms if not self._is_abbreviation(term)]
            if full_word_terms:
                return max(full_word_terms, key=lambda x: (len(x), x))
            else:
                return max(clean_terms, key=lambda x: (len(x), x))

        return group[0]

    def _select_final_standard(self, group: List[str]) -> str:
        return self._select_standard_term(group)

    def _is_abbreviation(self, term: str) -> bool:
        return (term.isupper() and len(term) < 5) or ('.' in term and term.replace('.', '').isupper())


class TripletCleaner:

    def __init__(self, entity_std_dict: Dict[str, str], relation_std_dict: Dict[str, str]):
        self.entity_std_dict = entity_std_dict
        self.relation_std_dict = relation_std_dict

    def clean_triplets(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned_triplets = []

        for triple in triplets:
            cleaned_triple = triple.copy()
            if 'start_node' in cleaned_triple:
                original = cleaned_triple['start_node']
                if original in self.entity_std_dict:
                    cleaned_triple['start_node'] = self.entity_std_dict[original]

            if 'relationship' in cleaned_triple:
                original = cleaned_triple['relationship']
                if original in self.relation_std_dict:
                    cleaned_triple['relationship'] = self.relation_std_dict[original]

            if 'end_node' in cleaned_triple:
                original = cleaned_triple['end_node']
                if original in self.entity_std_dict:
                    cleaned_triple['end_node'] = self.entity_std_dict[original]

            cleaned_triplets.append(cleaned_triple)

        return cleaned_triplets


def parse_args():
    parser = argparse.ArgumentParser(description="Semantically normalize extracted KG triples.")
    parser.add_argument("--input-dir", required=True, help="Directory containing extracted triple JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized triple JSON files.")
    parser.add_argument("--model-path", default=CONFIG["model_path"], help="SentenceTransformer model name or path.")
    parser.add_argument("--similarity-threshold", type=float, default=CONFIG["similarity_threshold"], help="Within-chunk similarity threshold.")
    parser.add_argument("--global-threshold", type=float, default=CONFIG["global_threshold"], help="Cross-chunk similarity threshold.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Maximum number of terms per standardization chunk.")
    return parser.parse_args()


def main():
    global INPUT_DIR, OUTPUT_DIR, CHUNK_SIZE

    args = parse_args()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    CHUNK_SIZE = args.chunk_size
    CONFIG["model_path"] = args.model_path
    CONFIG["similarity_threshold"] = args.similarity_threshold
    CONFIG["global_threshold"] = args.global_threshold

    print("="*80)
    print("Photoresist Domain Triplet Batch Cleaning System")
    print("Maintaining same logic as standardization dictionary with optimized memory usage")
    print("="*80)

    start_time = time.time()

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory does not exist: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        print("\n[1/4] Loading data...")
        stats = StatsCollector()
        processor = TripletProcessor(stats)
        all_triplets, file_paths = processor.load_all_triplets_from_dir(INPUT_DIR)

        if not all_triplets:
            print("❌ No triplet data loaded")
            return

        print(f"Loading complete: {len(all_triplets)} triplets from {len(file_paths)} files")

        print("\n[2/4] Extracting terms...")
        entities, relations = processor.extract_terms_from_triplets(all_triplets)

        print(f"Extraction complete: {len(entities)} entities, {len(relations)} relations")

        print("\n[3/4] Standardizing terms...")
        standardizer = IncrementalStandardizer(CONFIG)

        entity_std_dict = {}
        if entities:
            entity_std_dict = standardizer.standardize_terms(entities, "Entity")

        relation_std_dict = {}
        if relations:
            relation_std_dict = standardizer.standardize_terms(relations, "Relation")

        print(f"\nStandardization complete:")
        print(f"  Entity mappings: {len(entity_std_dict)}")
        print(f"  Relation mappings: {len(relation_std_dict)}")

        print("\n[4/4] Cleaning files...")
        cleaner = TripletCleaner(entity_std_dict, relation_std_dict)

        processed_count = 0
        failed_files = []

        for file_path in file_paths:
            try:
                triplets = processor.load_triplets_from_file(file_path)
                if not triplets:
                    continue

                cleaned_triplets = cleaner.clean_triplets(triplets)

                rel_path = os.path.relpath(file_path, INPUT_DIR)
                output_file = os.path.join(OUTPUT_DIR, rel_path)

                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_triplets, f, ensure_ascii=False, indent=2)

                processed_count += 1
                print(f"  ✅ {os.path.basename(file_path)}: {len(triplets)} → {len(cleaned_triplets)}")

            except Exception as e:
                print(f"  ❌ {os.path.basename(file_path)} failed: {e}")
                failed_files.append(os.path.basename(file_path))

        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("\n" + "="*80)
        print("🎉 Batch cleaning completed!")
        print("="*80)
        print(f"⏱️  Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        print(f"📁 Files processed: {processed_count}/{len(file_paths)}")
        print(f"📊 Original triplets: {len(all_triplets)}")
        print(f"💾 Output directory: {OUTPUT_DIR}")

        if failed_files:
            print(f"\n⚠️ Failed files ({len(failed_files)}):")
            for failed_file in failed_files[:10]:
                print(f"  - {failed_file}")

        print("="*80)

    except Exception as e:
        print(f"\n❌ Program execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program execution failed: {e}")
        import traceback
        traceback.print_exc()