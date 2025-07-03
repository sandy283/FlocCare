# SQLite compatibility fix for ChromaDB on cloud platforms
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import json
from pydantic import BaseModel, Field
from typing import List

client = chromadb.PersistentClient(path="./rag_db")

# Handle collection creation/access gracefully
try:
    collection = client.get_collection("medical_docs")
    print("✅ Found existing RAG collection")
except Exception as e:
    print(f"⚠️ RAG collection not found: {e}")
    try:
        # Try to list collections to see what's available
        collections = client.list_collections()
        if collections:
            collection = collections[0]  # Use the first available collection
            print(f"✅ Using available collection: {collection.name}")
        else:
            # Create a dummy collection as fallback
            collection = client.create_collection("medical_docs_fallback")
            print("✅ Created fallback collection")
    except Exception as create_error:
        print(f"❌ Could not create/access any collection: {create_error}")
        collection = None

model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryExpansion(BaseModel):
    variations: List[str] = Field(
        description="List of 3 query variations with different wording and terminology",
        min_items=3,
        max_items=3
    )

class RerankingEvaluation(BaseModel):
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of the relevance score")

def clean_llm_response(content):
    content = content.strip()
    
    if '<think>' in content and '</think>' in content:
        think_end = content.rfind('</think>')
        if think_end != -1:
            content = content[think_end + 8:].strip()
    
    content = content.replace('<think>', '').replace('</think>', '').strip()
    if content.startswith('```json'):
        content = content[7:].strip()
    elif content.startswith('```'):
        content = content[3:].strip()
    
    if content.endswith('```'):
        content = content[:-3].strip()
    
    json_start = content.find('{')
    if json_start == -1:
        if '[' in content and ']' in content and '"' in content:
            array_start = content.find('[')
            array_end = content.rfind(']')
            if array_start != -1 and array_end != -1:
                array_content = content[array_start:array_end + 1]
                if 'variations' in content.lower():
                    content = f'{{"variations": {array_content}}}'
                    json_start = 0
        
        if json_start == -1:
            raise ValueError(f"No JSON object found in response. Content: {content[:200]}...")
    
    brace_count = 0
    json_end = -1
    for i in range(json_start, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i
                break
    
    if json_end == -1:
        raise ValueError(f"No matching closing brace found. Content from start: {content[json_start:json_start+200]}...")
    
    json_content = content[json_start:json_end + 1]
    
    try:
        json.loads(json_content)
        return json_content.strip()
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON structure: {e}. JSON content: {json_content}")

def expand_query(original_query):
    base_prompt = f"""Given this query: "{original_query}"

Generate EXACTLY 3 alternative versions (no more, no less) that capture the same intent but use different approaches:

Return ONLY this JSON format with EXACTLY 3 variations:

{{
    "variations": [
        "variation 1",
        "variation 2", 
        "variation 3"
    ]
}}

CRITICAL: You must provide exactly 3 variations, not 4 or more. Do not add explanations or extra content."""
    
    for attempt in range(3):
        try:
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = f"""{base_prompt}

IMPORTANT: Previous attempt failed with error. Please ensure you:
- Return EXACTLY 3 variations (not more, not less)
- Use only simple strings in the variations array
- Return valid JSON format
- No additional explanations or fields"""
            
            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )
            
            content = clean_llm_response(response['message']['content'])
            
            import json
            json_data = json.loads(content)
            if 'variations' in json_data and len(json_data['variations']) > 3:
                json_data['variations'] = json_data['variations'][:3]
                content = json.dumps(json_data)
            
            parsed_response = QueryExpansion.model_validate_json(content)
            return [original_query] + parsed_response.variations
            
        except Exception as e:
            if attempt == 2:
                raise Exception(f"Query expansion failed after 3 attempts. Last error: {e}")
            continue

def search_multiple_queries(queries, n_results=5):
    all_results = []
    seen_docs = set()
    
    # Handle case where collection is not available
    if collection is None:
        print("⚠️ No RAG collection available, returning empty results")
        return all_results
    
    for query in queries:
        try:
            query_embedding = model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                doc_id = doc[:100]
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    all_results.append({
                        'document': doc,
                        'metadata': meta,
                        'distance': dist,
                        'relevance_score': 1 - dist,
                        'query': query
                    })
        except Exception as e:
            print(f"⚠️ Error querying collection: {e}")
            continue
    
    return all_results

def rerank_with_llm(results, original_query):
    reranked_results = []
    
    for result in results:
        meta = result['metadata']
        content_sample = result['document'][:800]
        
        base_prompt = f"""Evaluate how relevant this document is to the query: "{original_query}"

Document Source: {meta.get('source_file', 'Unknown')}
Document Content: {content_sample}

Rate the relevance on a scale from 0.0 to 1.0 where:
- 1.0 = Directly answers the query with specific information
- 0.7-0.9 = Highly relevant, contains related information
- 0.4-0.6 = Moderately relevant, tangentially related
- 0.1-0.3 = Slightly relevant, mentions related topics
- 0.0 = Not relevant at all

Respond in this exact JSON format:
{{
    "relevance_score": 0.85,
    "reasoning": "brief explanation of why this score was given"
}}"""

        for attempt in range(3):
            try:
                if attempt == 0:
                    prompt = base_prompt
                else:
                    prompt = f"""{base_prompt}

IMPORTANT: Previous attempt failed. Please ensure you:
- Return valid JSON format with curly braces
- Include both "relevance_score" (number 0.0-1.0) and "reasoning" (string) fields
- No additional text outside the JSON object"""

                response = ollama.chat(
                    model='gemma3:4b',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.1}
                )
                
                content = clean_llm_response(response['message']['content'])
                evaluation = RerankingEvaluation.model_validate_json(content)
                
                semantic_score = result['relevance_score']
                llm_score = evaluation.relevance_score
                
                final_score = (llm_score * 0.6) + (semantic_score * 0.4)
                
                result['llm_relevance'] = llm_score
                result['llm_reasoning'] = evaluation.reasoning
                result['final_score'] = final_score
                
                reranked_results.append(result)
                break
                
            except Exception as e:
                if attempt == 2:
                    result['llm_relevance'] = 0.0
                    result['llm_reasoning'] = f"LLM evaluation failed after 3 attempts: {e}"
                    result['final_score'] = result['relevance_score']
                    reranked_results.append(result)
                continue
    
    return sorted(reranked_results, key=lambda x: x['final_score'], reverse=True)

def rerank_results(results, original_query):
    return rerank_with_llm(results, original_query)

class ContextSummary(BaseModel):
    key_requirements: List[str] = Field(description="List of key regulatory requirements")
    procedures: List[str] = Field(description="List of required procedures or steps")
    compliance_notes: List[str] = Field(description="Important compliance considerations")
    summary: str = Field(description="Concise overall summary")

def distill_context(top_results, original_query):
    context = "\n\n".join([f"Source: {r['metadata'].get('source_file', 'Unknown')}\nContent: {r['document']}" 
                          for r in top_results[:3]])
    
    base_prompt = f"""Analyze these documents to answer: "{original_query}"

{context}

Extract and structure the most relevant information into these categories:

Respond in this exact JSON format:
{{
    "key_requirements": ["list of main requirements or rules"],
    "procedures": ["list of processes or steps mentioned"],
    "compliance_notes": ["important considerations or warnings"],
    "summary": "brief overall answer to the query"
}}"""
    
    for attempt in range(3):
        try:
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = f"""{base_prompt}

IMPORTANT: Previous attempt failed. Please ensure you:
- Return valid JSON format with curly braces
- Include all 4 required fields: "key_requirements", "procedures", "compliance_notes", "summary"
- Use arrays for the first 3 fields and string for summary
- No additional text outside the JSON object"""

            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.2}
            )
            
            content = clean_llm_response(response['message']['content'])
            parsed_response = ContextSummary.model_validate_json(content)
            
            return f"""
Key Requirements:
{chr(10).join([f"• {req}" for req in parsed_response.key_requirements])}

Required Procedures:
{chr(10).join([f"• {proc}" for proc in parsed_response.procedures])}

Compliance Notes:
{chr(10).join([f"• {note}" for note in parsed_response.compliance_notes])}

Summary:
{parsed_response.summary}
"""
            
        except Exception as e:
            if attempt == 2:
                return f"""
Context distillation failed after 3 attempts. Last error: {e}

Raw context available but could not be structured properly.
"""
            continue

def advanced_rag_query(query_text, n_results=10):
    print(f"Original Query: {query_text}")
    
    # Check if RAG system is available
    if collection is None:
        print("⚠️ RAG database not available. Providing general response.")
        return f"""
RAG Database Status: Not Available

The regulatory document database is currently not accessible. 
For query: "{query_text}"

General Guidance:
• Consult official regulatory authority websites (FDA, EMA, HSA)
• Review current Good Manufacturing Practice (GMP) guidelines
• Check product labeling and advertising requirements
• Verify clinical trial and approval requirements
• Consider regional regulatory differences

Note: For specific regulatory guidance, please consult the original regulatory documents 
and seek advice from qualified regulatory professionals.
"""
    
    print("\nExpanding query...")
    expanded_queries = expand_query(query_text)
    for i, q in enumerate(expanded_queries):
        print(f"   {i+1}. {q}")
    
    print(f"\nSearching with {len(expanded_queries)} queries...")
    all_results = search_multiple_queries(expanded_queries, n_results//2)
    print(f"   Found {len(all_results)} unique documents")
    
    # Handle case where no results are found
    if not all_results:
        print("⚠️ No documents found in RAG database for this query.")
        return f"""
No Relevant Documents Found

Query: "{query_text}"

The RAG system searched through available regulatory documents but found no relevant matches. 
This could mean:
• The query requires information not available in the current database
• Try rephrasing the query with different terminology
• Consider broader or more specific search terms

General Recommendation:
Consult official regulatory sources directly for the most up-to-date and comprehensive guidance.
"""
    
    print("\nReranking results using LLM evaluation...")
    reranked_results = rerank_results(all_results, query_text)
    
    print("\nDistilling key information...")
    key_info = distill_context(reranked_results[:5], query_text)
    print("\n" + "="*80)
    print("KEY INFORMATION")
    print("="*80)
    print(key_info)
    
    print("\n" + "="*80)
    print("TOP RANKED SOURCES")
    print("="*80)
    for i, result in enumerate(reranked_results[:3]):
        print(f"\n{i+1}. Final Score: {result['final_score']:.3f}")
        print(f"   LLM Relevance: {result['llm_relevance']:.3f}")
        print(f"   LLM Reasoning: {result['llm_reasoning']}")
        print(f"   Source: {result['metadata'].get('source_file', 'Unknown')}")
        print(f"   Semantic Score: {result['relevance_score']:.3f}")
        print(f"   Matched Query: {result['query']}")
        print(f"   Content Preview: {result['document'][:200]}...")
    
    return key_info

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ")
    
    advanced_rag_query(query) 