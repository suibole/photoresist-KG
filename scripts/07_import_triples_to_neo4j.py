import argparse
import json
import os
import glob
from neo4j import GraphDatabase
import time

class TripleUploaderSuiboleKG:
    def __init__(self, uri="bolt://localhost:7687", user="", password=""):
        self.project_name = "suibole_kg"
        self.entity_label = "SuiboleEntity"
        self.relation_label = "SUIBOLE_RELATION"
        
        try:
            if user and password:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                print(f"🔗 Connecting with authentication: {uri}")
            else:
                self.driver = GraphDatabase.driver(uri)
                print(f"🔗 Connecting without authentication: {uri}")
            
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection successful' AS message")
                print(f"✅ {result.single()['message']}")
                
            self._create_constraints()
            print(f"🎯 Project name: {self.project_name}")
            print(f"🏷️  Entity label: {self.entity_label}")
            print(f"🏷️  Relation label: {self.relation_label}")
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise

    def _create_constraints(self):
        constraints = [
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:{self.entity_label}) REQUIRE e.name IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (p:SuibolePaper) REQUIRE p.doi IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:SuiboleChemical) REQUIRE c.name IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (m:SuiboleMethod) REQUIRE m.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"✅ Created constraint: {constraint}")
                except Exception as e:
                    print(f"⚠️  Constraint creation failed: {e}")

    def auto_clear_existing_data(self, force=False):
        print(f"\n🧹 Auto clearing existing {self.project_name} project data...")
        
        with self.driver.session() as session:
            try:
                result = session.run(f"MATCH (n:{self.entity_label}) RETURN count(n) as nodes")
                node_count = result.single()["nodes"]
                
                if node_count == 0:
                    print("✅ Database is already empty, no need to clear")
                    return True
                
                if not force:
                    print(f"⚠️  Detected {node_count} existing nodes")
                    confirmation = input(f"Delete existing data? Enter 'YES_DELETE_{self.project_name.upper()}' to confirm: ")
                    
                    if confirmation != f"YES_DELETE_{self.project_name.upper()}":
                        print("❌ Operation cancelled")
                        return False
                
                print("🗑️  Clearing data...")
                result = session.run(f"MATCH (n:{self.entity_label}) DETACH DELETE n")
                summary = result.consume()
                nodes_deleted = summary.counters.nodes_deleted or 0
                relationships_deleted = summary.counters.relationships_deleted or 0
                print(f"✅ Clear complete: Deleted {nodes_deleted} nodes, {relationships_deleted} relationships")
                
                return True
                
            except Exception as e:
                print(f"❌ Clear failed: {e}")
                return False

    def load_triples_from_file(self, file_path):
        all_triples = []
        
        if not os.path.exists(file_path):
            print(f"❌ File does not exist: {file_path}")
            return all_triples
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                triples = json.load(f)
                for triple in triples:
                    triple['project'] = self.project_name
                    triple['source_file'] = os.path.basename(file_path)
                all_triples.extend(triples)
                print(f"✅ Loaded file {os.path.basename(file_path)}: {len(triples)} triples")
        except Exception as e:
            print(f"❌ Failed to load file {file_path}: {e}")
        
        print(f"📊 Total loaded {len(all_triples)} triples")
        return all_triples

    def load_triples_from_directory(self, directory_path):
        all_triples = []
        
        if os.path.isfile(directory_path):
            return self.load_triples_from_file(directory_path)
        
        json_files = glob.glob(os.path.join(directory_path, "*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in directory {directory_path}")
            return all_triples
        
        print(f"📁 Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    triples = json.load(f)
                    for triple in triples:
                        triple['project'] = self.project_name
                        triple['source_file'] = os.path.basename(json_file)
                    all_triples.extend(triples)
                    print(f"✅ Loaded {os.path.basename(json_file)}: {len(triples)} triples")
            except Exception as e:
                print(f"❌ Failed to load file {json_file}: {e}")
        
        print(f"📊 Total loaded {len(all_triples)} triples")
        return all_triples

    def import_triples(self, triples, batch_size=100):
        if not triples:
            print("❌ No triple data to import")
            return 0
        
        total_triples = len(triples)
        imported_count = 0
        errors = []
        
        print(f"🚀 Starting import of {total_triples} triples to project: {self.project_name}...")
        start_time = time.time()
        
        for i in range(0, total_triples, batch_size):
            batch = triples[i:i + batch_size]
            batch_imported = 0
            
            for triple in batch:
                try:
                    success = self._import_single_triple(triple)
                    if success:
                        batch_imported += 1
                        imported_count += 1
                    else:
                        errors.append(triple)
                except Exception as e:
                    errors.append((triple, str(e)))
            
            progress = (i + len(batch)) / total_triples * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (progress / 100) if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time
            
            print(f"📦 Batch {i//batch_size + 1}: Imported {batch_imported}/{len(batch)} triples, Progress: {progress:.1f}%")
            print(f"   ⏱️  Elapsed: {elapsed_time:.1f}s, Remaining: {remaining_time:.1f}s")
            
            if i + batch_size < total_triples:
                time.sleep(0.1)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Import completed!")
        print(f"✅ Successfully imported: {imported_count}/{total_triples} triples")
        print(f"❌ Failed imports: {len(errors)}")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📈 Average speed: {imported_count/total_time:.2f} triples/second" if total_time > 0 else "📈 Speed: Very fast")
        
        if errors:
            print(f"\n⚠️  First 5 error examples:")
            for i, error in enumerate(errors[:5]):
                if isinstance(error, tuple):
                    triple, error_msg = error
                    print(f"   {i+1}. {triple.get('start_node', '')} -> {triple.get('end_node', '')} | Error: {error_msg}")
                else:
                    print(f"   {i+1}. {error.get('start_node', '')} -> {error.get('end_node', '')}")
        
        return imported_count

    def _import_single_triple(self, triple):
        try:
            cypher = f"""
            MERGE (s:{self.entity_label} {{name: $start_node}})
            ON CREATE SET
                s.doi = $doi,
                s.title = $title,
                s.source_file = $source_file,
                s.project = $project,
                s.import_time = datetime(),
                s.project_name = $project_name
            MERGE (e:{self.entity_label} {{name: $end_node}})
            ON CREATE SET
                e.doi = $doi,
                e.title = $title,
                e.source_file = $source_file,
                e.project = $project,
                e.import_time = datetime(),
                e.project_name = $project_name
            MERGE (s)-[r:{self.relation_label} {{type: $relationship}}]->(e)
            ON CREATE SET
                r.source_file = $source_file,
                r.project = $project,
                r.import_time = datetime(),
                r.project_name = $project_name
            SET s.attributes = reduce(unique_attrs = [], attr IN coalesce(s.attributes, []) + $sn_attributes |
                    CASE WHEN attr IN unique_attrs THEN unique_attrs ELSE unique_attrs + attr END),
                r.attributes = reduce(unique_attrs = [], attr IN coalesce(r.attributes, []) + $re_attributes |
                    CASE WHEN attr IN unique_attrs THEN unique_attrs ELSE unique_attrs + attr END),
                e.attributes = reduce(unique_attrs = [], attr IN coalesce(e.attributes, []) + $en_attributes |
                    CASE WHEN attr IN unique_attrs THEN unique_attrs ELSE unique_attrs + attr END),
                s.source_files = reduce(unique_sources = [], src IN coalesce(s.source_files, []) + [$source_file] |
                    CASE WHEN src IN unique_sources OR src = '' THEN unique_sources ELSE unique_sources + src END),
                r.source_files = reduce(unique_sources = [], src IN coalesce(r.source_files, []) + [$source_file] |
                    CASE WHEN src IN unique_sources OR src = '' THEN unique_sources ELSE unique_sources + src END),
                e.source_files = reduce(unique_sources = [], src IN coalesce(e.source_files, []) + [$source_file] |
                    CASE WHEN src IN unique_sources OR src = '' THEN unique_sources ELSE unique_sources + src END)
            """
            
            parameters = {
                'start_node': triple.get('start_node', ''),
                'end_node': triple.get('end_node', ''),
                'relationship': triple.get('relationship', ''),
                'sn_attributes': triple.get('sn_attribute', []) or triple.get('sn_attributes', []),
                're_attributes': triple.get('re_attribute', []) or triple.get('re_attributes', []),
                'en_attributes': triple.get('en_attribute', []) or triple.get('en_attributes', []),
                'doi': triple.get('doi', ''),
                'title': triple.get('title', ''),
                'source_file': triple.get('source_file', ''),
                'project': self.project_name,
                'project_name': self.project_name
            }
            
            with self.driver.session() as session:
                result = session.run(cypher, parameters)
                list(result)
            
            return True
            
        except Exception as e:
            print(f"❌ Import failed: {triple.get('start_node', '')} -> {triple.get('end_node', '')} | Error: {e}")
            return False

    def get_statistics(self):
        print(f"\n📊 Getting statistics for project '{self.project_name}'...")
        
        stats_queries = {
            "Total nodes": f"MATCH (n:{self.entity_label}) RETURN count(n) as count",
            "Total relationships": f"MATCH ()-[r:{self.relation_label}]->() RETURN count(r) as count",
            "Unique entities": f"MATCH (n:{self.entity_label}) RETURN count(DISTINCT n.name) as count",
            "Relationship type distribution": f"MATCH ()-[r:{self.relation_label}]->() RETURN r.type as type, count(*) as count ORDER BY count DESC LIMIT 10",
            "Data source files": f"MATCH (n:{self.entity_label}) RETURN DISTINCT n.source_file as file, count(*) as count ORDER BY count DESC LIMIT 10"
        }
        
        stats = {}
        with self.driver.session() as session:
            for key, query in stats_queries.items():
                try:
                    if key in ["Total nodes", "Total relationships", "Unique entities"]:
                        result = session.run(query)
                        record = result.single()
                        if record:
                            stats[key] = record[0]
                            print(f"   {key}: {record[0]}")
                    else:
                        print(f"\n   {key}:")
                        result = session.run(query)
                        for record in result:
                            print(f"     - {record['type'] if 'type' in record else record['file']}: {record['count']}")
                except Exception as e:
                    print(f"❌ Statistics query failed for {key}: {e}")
        
        return stats

    def clear_project_data(self):
        print(f"\n⚠️  Warning: About to clear all data for project '{self.project_name}'!")
        confirmation = input("   Are you sure you want to continue? Enter 'DELETE_SUIBOLE_KG' to confirm: ")
        
        if confirmation != "DELETE_SUIBOLE_KG":
            print("❌ Operation cancelled")
            return False
        
        print("🧹 Clearing data...")
        
        queries = [
            f"MATCH (n:{self.entity_label}) DETACH DELETE n",
            f"MATCH ()-[r:{self.relation_label}]-() DELETE r"
        ]
        
        total_deleted = 0
        with self.driver.session() as session:
            for query in queries:
                try:
                    result = session.run(query)
                    summary = result.consume()
                    nodes_deleted = summary.counters.nodes_deleted or 0
                    relationships_deleted = summary.counters.relationships_deleted or 0
                    total_deleted += nodes_deleted + relationships_deleted
                    print(f"✅ Clear operation: {query}")
                    print(f"   Nodes deleted: {nodes_deleted}, Relationships deleted: {relationships_deleted}")
                except Exception as e:
                    print(f"❌ Clear operation failed: {e}")
        
        print(f"🎉 Data clearing completed! Total deleted {total_deleted} elements")
        return True

    def test_queries(self):
        test_queries = [
            {
                "name": "Random sample nodes",
                "query": f"MATCH (n:{self.entity_label}) RETURN n.name as name, n.source_file as source, n.project_name as project LIMIT 5"
            },
            {
                "name": "Random sample relationships",
                "query": f"MATCH (s:{self.entity_label})-[r:{self.relation_label}]->(e:{self.entity_label}) RETURN s.name as start, r.type as relation, e.name as end LIMIT 5"
            },
            {
                "name": "Relationship type statistics",
                "query": f"MATCH ()-[r:{self.relation_label}]->() RETURN r.type as type, count(*) as count ORDER BY count DESC LIMIT 5"
            }
        ]
        
        print(f"\n🧪 Testing queries for project '{self.project_name}':")
        with self.driver.session() as session:
            for test in test_queries:
                print(f"\n   {test['name']}:")
                try:
                    result = session.run(test['query'])
                    for i, record in enumerate(result, 1):
                        if 'start' in record:
                            print(f"     {i}. {record['start']} --[{record['relation']}]--> {record['end']}")
                        elif 'type' in record:
                            print(f"     {i}. {record['type']}: {record['count']} times")
                        else:
                            print(f"     {i}. {record['name']} (Source: {record['source']}, Project: {record['project']})")
                except Exception as e:
                    print(f"     ❌ Query failed: {e}")

    def close(self):
        if hasattr(self, 'driver'):
            self.driver.close()
            print(f"\n🔌 Closed Neo4j connection")

def parse_args():
    parser = argparse.ArgumentParser(description="Import extracted KG triples into Neo4j.")
    parser.add_argument("--triples-path", required=True, help="Directory containing triple JSON files, or a single JSON file.")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j Bolt URI.")
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", ""), help="Neo4j user name. Leave empty for no authentication.")
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""), help="Neo4j password. Leave empty for no authentication.")
    return parser.parse_args()


def main():
    args = parse_args()
    TRIPLES_PATH = args.triples_path
    neo4j_user = args.neo4j_user or None
    neo4j_password = args.neo4j_password or None
    
    print("=" * 70)
    print("suibole_kg Knowledge Graph Data Upload Tool - Friendly Label Version")
    print("Batch processing 10,000 JSON files")
    print("=" * 70)
    
    if not os.path.exists(TRIPLES_PATH):
        print(f"❌ Path does not exist: {TRIPLES_PATH}")
        return
    
    if os.path.isfile(TRIPLES_PATH):
        json_files = [TRIPLES_PATH]
    else:
        json_files = glob.glob(os.path.join(TRIPLES_PATH, "*.json"))
    print(f"📁 Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print("❌ No JSON files found")
        return
    
    uploader = TripleUploaderSuiboleKG(
        uri=args.neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )
    
    try:
        print(f"\n🗑️  Step 1: Clearing existing {uploader.project_name} project data")
        if not uploader.auto_clear_existing_data(force=False):
            print("❌ User cancelled clear operation, exiting program")
            return
        
        total_imported = 0
        batch_size = 100
        file_batches = [json_files[i:i+batch_size] for i in range(0, len(json_files), batch_size)]
        
        for batch_num, file_batch in enumerate(file_batches, 1):
            print(f"\n{'='*60}")
            print(f"📦 Processing batch {batch_num}/{len(file_batches)}")
            print(f"📄 Files in this batch: {len(file_batch)}")
            print(f"{'='*60}")
            
            batch_triples = []
            for json_file in file_batch:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        triples = json.load(f)
                        for triple in triples:
                            triple['project'] = uploader.project_name
                            triple['source_file'] = os.path.basename(json_file)
                        batch_triples.extend(triples)
                        print(f"✅ Loaded {os.path.basename(json_file)}: {len(triples)} triples")
                except Exception as e:
                    print(f"❌ Failed to load file {os.path.basename(json_file)}: {e}")
            
            if not batch_triples:
                print("⚠️  No data to import in this batch")
                continue
            
            print(f"\n🚀 Importing batch data: {len(batch_triples)} triples")
            imported_count = uploader.import_triples(batch_triples, batch_size=50)
            total_imported += imported_count
            
            completed_files = min(batch_num * batch_size, len(json_files))
            progress_percent = completed_files / len(json_files) * 100
            print(f"📊 Overall progress: {completed_files}/{len(json_files)} files ({progress_percent:.1f}%)")
            print(f"📈 Cumulative imported: {total_imported} triples")
            
            if batch_num < len(file_batches):
                time.sleep(1)
        
        print(f"\n{'='*70}")
        print(f"🎉 Batch processing completed!")
        print(f"📄 Total files processed: {len(json_files)}")
        print(f"✅ Total triples imported: {total_imported}")
        
        if total_imported > 0:
            stats = uploader.get_statistics()
            uploader.test_queries()
            
            print(f"\n💡 Query in Neo4j Browser:")
            print(f"   // View all nodes")
            print(f"   MATCH (n:{uploader.entity_label}) RETURN n LIMIT 25")
            print(f"   \n   // View all relationships")
            print(f"   MATCH (n:{uploader.entity_label})-[r:{uploader.relation_label}]->(m) RETURN n,r,m LIMIT 25")
            print(f"   \n   // Group by source file")
            print(f"   MATCH (n:{uploader.entity_label}) RETURN n.source_file, count(*) ORDER BY count(*) DESC LIMIT 20")
            
    except Exception as e:
        print(f"❌ Upload process error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        uploader.close()

if __name__ == "__main__":
    main()