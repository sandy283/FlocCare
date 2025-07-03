#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

import chromadb
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class AdvancedRAGQuery:
    def __init__(self, 
                 db_path: str = "./rag_db",
                 collection_name: str = "medical_docs",
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_deployment=os.getenv('AZURE_DEPLOYMENT'),
            api_version=os.getenv('API_VERSION'),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
            max_tokens=None,
            timeout=None,
            max_retries=int(os.getenv('LLM_MAX_RETRIES', '1')),
        )
        
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_collection(collection_name)
        except Exception as e:
            raise Exception(f"Database not found. Run build_rag.py first. Error: {e}")
    
    def multi_stage_retrieval(self, query: str, n_results: int = None) -> List[Dict]:
        """Simplified retrieval with semantic search and re-ranking"""
        if n_results is None:
            n_results = int(os.getenv('DEFAULT_RETRIEVAL_RESULTS', '20'))
        
        # Stage 1: Initial semantic search
        query_embedding = self.embedding_model.encode(query).tolist()
        
        initial_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not initial_results["documents"][0]:
            return []
        
        # Stage 2: Simple scoring
        results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            initial_results["documents"][0],
            initial_results["metadatas"][0], 
            initial_results["distances"][0]
        )):
            results.append({
                "document": doc,
                "metadata": metadata,
                "relevance_score": 1.0 - distance,
                "semantic_distance": distance
            })
        
        # Stage 3: Re-ranking using LLM
        rerank_limit = int(os.getenv('RERANK_LIMIT', '10'))
        reranked_results = self._llm_rerank(query, results[:rerank_limit])
        
        return reranked_results
    
    def _llm_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Use LLM to re-rank results based on query relevance"""
        try:
            # Prepare context for LLM
            context_items = []
            for i, result in enumerate(results):
                context_items.append(f"""
Result {i+1}:
Category: {result['metadata'].get('category', 'Unknown')}
Summary: {result['metadata'].get('summary', 'No summary')}
Content: {result['document'][:300]}...
Current Score: {result['relevance_score']:.3f}
""")
            
            context = "\n".join(context_items)
            
            prompt = f"""Rerank these search results for the query: "{query}"

{context}

Return JSON array with result indices (1-based) in order of relevance to the query:
{{"rankings": [1, 3, 2, ...]}}

Consider:
- Direct relevance to query
- Content quality and completeness
- Regulatory importance
- Contextual relevance

Return only valid JSON."""
            
            response = self.llm.invoke(prompt)
            rankings_data = json.loads(response.content.strip())
            rankings = rankings_data.get("rankings", list(range(1, len(results) + 1)))
            
            # Apply LLM rankings
            reranked = []
            for rank_idx in rankings:
                if 1 <= rank_idx <= len(results):
                    result = results[rank_idx - 1].copy()
                    result["llm_rank"] = len(rankings) - rankings.index(rank_idx)
                    result["final_score"] = result["relevance_score"] * (1 + result["llm_rank"] * 0.1)
                    reranked.append(result)
            
            return sorted(reranked, key=lambda x: x["final_score"], reverse=True)
            
        except Exception as e:
            print(f"LLM reranking failed: {e}")
            return sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Main search interface"""
        results = self.multi_stage_retrieval(query, n_results * 2)
        return results[:n_results]
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            count = self.collection.count()
            sample = self.collection.get(limit=min(100, count), include=["metadatas"])
            
            stats = {
                "total_chunks": count,
                "categories": {},
                "content_types": {},
                "importance_distribution": {},
                "file_sources": set()
            }
            
            for metadata in sample["metadatas"]:
                # Category distribution
                category = metadata.get("category", "Unknown")
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                
                # Content type distribution
                content_type = metadata.get("content_type", "Unknown")
                stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
                
                # Importance distribution
                importance = metadata.get("importance_score", 5)
                stats["importance_distribution"][importance] = stats["importance_distribution"].get(importance, 0) + 1
                
                # File sources
                source_file = metadata.get("source_file", "")
                if source_file:
                    stats["file_sources"].add(source_file.split("/")[0] if "/" in source_file else source_file)
            
            stats["file_sources"] = list(stats["file_sources"])
            return stats
            
        except Exception as e:
            return {"error": str(e)}

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Advanced RAG Query System")
        print("Usage:")
        print("  python query_rag.py search \"your query\"")
        print("  python query_rag.py stats")
        return
    
    command = sys.argv[1].lower()
    
    try:
        query_system = AdvancedRAGQuery()
        
        if command == "search":
            if len(sys.argv) < 3:
                print("Please provide a search query")
                return
            
            query = sys.argv[2]
            results = query_system.search(query)
            
            print(f"Query: {query}")
            print(f"Found {len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                print(f"{i}. {metadata.get('filename', 'Unknown')}")
                print(f"   Category: {metadata.get('category', 'Unknown')}")
                print(f"   Score: {result.get('final_score', result['relevance_score']):.3f}")
                print(f"   Summary: {metadata.get('summary', 'No summary')}")
                print(f"   Content: {result['document'][:200]}...")
                print()
        
        elif command == "stats":
            stats = query_system.get_database_stats()
            print("Database Statistics:")
            print(f"Total chunks: {stats.get('total_chunks', 'N/A')}")
            print(f"Categories: {stats.get('categories', {})}")
            print(f"Content types: {stats.get('content_types', {})}")
            print(f"Source folders: {stats.get('file_sources', [])}")
        
        else:
            print(f"Unknown command: {command}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
