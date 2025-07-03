#!/usr/bin/env python3
import os
import re
import json
import logging
from pathlib import Path
from typing import Literal, Optional, List, Dict
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import time

import chromadb
import google.generativeai as genai
import tiktoken
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure enhanced logging with immediate flush
class ImmediateFileHandler(logging.FileHandler):
    """Custom FileHandler that flushes after every emit"""
    def emit(self, record):
        super().emit(record)
        self.flush()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        ImmediateFileHandler('rag_builder_detailed.log'),
        logging.StreamHandler()
    ]
)

# Create separate loggers for different components
logger = logging.getLogger(__name__)
chunk_logger = logging.getLogger('chunk_processor')
metadata_logger = logging.getLogger('metadata_generator')
db_logger = logging.getLogger('chromadb_operations')

# Configure Google AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../oval-sunset-404409-7164f1dbd579.json"
genai.configure(api_key=os.getenv('GOOGLE_API_KEY', ''))

class ProcessingStats:
    """Track processing statistics"""
    def __init__(self):
        self.start_time = time.time()
        self.files_processed = 0
        self.chunks_created = 0
        self.chunks_stored = 0
        self.errors = 0
        self.llm_calls = 0
        self.llm_failures = 0
    
    def log_summary(self):
        duration = time.time() - self.start_time
        logger.info("="*80)
        logger.info("PROCESSING SUMMARY:")
        logger.info(f"Total Duration: {duration:.2f} seconds")
        logger.info(f"Files Processed: {self.files_processed}")
        logger.info(f"Chunks Created: {self.chunks_created}")
        logger.info(f"Chunks Successfully Stored: {self.chunks_stored}")
        logger.info(f"Errors: {self.errors}")
        logger.info(f"LLM Calls: {self.llm_calls}")
        logger.info(f"LLM Failures: {self.llm_failures}")
        logger.info(f"Success Rate: {(self.chunks_stored/self.chunks_created*100) if self.chunks_created > 0 else 0:.2f}%")
        logger.info("="*80)

class DocumentMetadata(BaseModel):
    summary: Optional[str] = Field(default=None, description="2-3 sentence summary")
    key_topics: Optional[List[str]] = Field(default=None, description="3-5 main topics")
    entities: Optional[Dict[str, List[str]]] = Field(default=None, description="Extracted entities")
    category: Optional[Literal["Drug Safety", "Medical Devices", "Clinical Trials", "Regulatory Guidance", "Warning Letter", "Compliance", "Quality Control", "Manufacturing", "Adverse Events", "Pharmacovigilance"]] = Field(default=None, description="Medical category")
    importance_score: Optional[int] = Field(default=None, ge=1, le=10, description="Relevance score 1-10")
    semantic_tags: Optional[List[str]] = Field(default=None, description="Searchable tags")
    content_type: Optional[Literal["guidance", "warning", "regulation", "policy", "notice", "report", "document", "letter", "standard", "procedure"]] = Field(default=None, description="Document type")
    regulatory_focus: Optional[List[str]] = Field(default=None, description="Main regulatory focus areas")
    
    # System metadata
    source_file: str = Field(default="")
    filename: str = Field(default="")
    section_idx: int = Field(default=0)
    chunk_idx: int = Field(default=0)
    token_count: int = Field(default=0)
    char_count: int = Field(default=0)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    file_size: int = Field(default=0)
    processing_time: float = Field(default=0.0)

class RAGBuilder:
    def __init__(self, 
                 db_path: str = "./rag_db",
                 workspace_path: str = "../",
                 collection_name: str = "medical_docs",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 800,
                 overlap_size: int = 160,
                 min_text_length: int = 100,
                 min_section_length: int = 50):
        
        logger.info("="*80)
        logger.info("INITIALIZING RAG BUILDER")
        logger.info(f"Database Path: {db_path}")
        logger.info(f"Collection Name: {collection_name}")
        logger.info(f"Chunk Size: {chunk_size} tokens")
        logger.info(f"Overlap Size: {overlap_size} tokens")
        logger.info("="*80)
        
        self.workspace_path = Path(workspace_path)
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_text_length = min_text_length
        self.min_section_length = min_section_length
        self.stats = ProcessingStats()
        
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("✓ Embedding model loaded successfully")
        
        # Initialize Gemini model - UPDATED TO 1.5-FLASH
        try:
            logger.info("Initializing Gemini 1.5 Flash model...")
            self.llm = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("✓ Gemini 1.5 Flash model initialized successfully")
        except Exception as e:
            logger.error(f"✗ Gemini model initialization failed: {e}")
            self.llm = None
        
        # Test LLM connection
        self._test_llm_connection()
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        existing_count = self.collection.count()
        logger.info(f"✓ ChromaDB initialized with {existing_count} existing documents")
        print(f"\nRAG Builder initialized successfully with {existing_count} existing documents")
    
    def _test_llm_connection(self):
        """Test LLM connection and functionality"""
        if self.llm is None:
            logger.warning("✗ LLM not initialized")
            return
            
        try:
            logger.info("Testing LLM connection...")
            test_response = self.llm.generate_content("Return the word 'test' as JSON: {'result': 'test'}")
            if not test_response or not test_response.text:
                raise ValueError("Empty response from LLM")
            logger.info("✓ LLM connection test successful")
            logger.info(f"Test response: {test_response.text[:100]}...")
        except Exception as e:
            logger.error(f"✗ LLM connection test failed: {e}")
            logger.warning("Continuing without LLM-based features...")
            self.llm = None
    
    def semantic_chunk_text(self, text: str, file_name: str, chunk_size: int = None, overlap: int = None) -> List[Dict]:
        """Advanced semantic chunking with hierarchical structure"""
        chunk_logger.info(f"[{file_name}] Starting semantic chunking...")
        
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap_size
            
        # Structure-aware preprocessing
        text = self._preserve_structure(text)
        
        # Split at natural boundaries
        sections = re.split(r'\n\s*(?:===|---|\*\*\*)\s*\n|\n{3,}', text)
        chunk_logger.info(f"[{file_name}] Split into {len(sections)} sections")
        
        all_chunks = []
        
        for section_idx, section in enumerate(sections):
            if len(section.strip()) < self.min_section_length:
                chunk_logger.debug(f"[{file_name}] Skipping section {section_idx} - too short")
                continue
                
            chunk_logger.debug(f"[{file_name}] Processing section {section_idx} ({len(section)} chars)")
            
            # Sentence-level chunking
            sentences = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = ""
            current_tokens = 0
            chunk_idx = 0
            
            for sentence in sentences:
                sentence_tokens = len(self.tokenizer.encode(sentence))
                
                if current_tokens + sentence_tokens > chunk_size and current_chunk:
                    # Create chunk with overlap
                    chunk_data = self._create_chunk_data(current_chunk, section_idx, chunk_idx)
                    all_chunks.append(chunk_data)
                    chunk_logger.debug(f"[{file_name}] Created chunk {len(all_chunks)} (section {section_idx}, chunk {chunk_idx})")
                    
                    # Add overlap from end of current chunk
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = len(self.tokenizer.encode(current_chunk))
                    chunk_idx += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_tokens += sentence_tokens
            
            if current_chunk.strip():
                chunk_data = self._create_chunk_data(current_chunk, section_idx, chunk_idx)
                all_chunks.append(chunk_data)
                chunk_logger.debug(f"[{file_name}] Created final chunk {len(all_chunks)} (section {section_idx}, chunk {chunk_idx})")
        
        chunk_logger.info(f"[{file_name}] Chunking complete: {len(all_chunks)} chunks created")
        self.stats.chunks_created += len(all_chunks)
        return all_chunks
    
    def _preserve_structure(self, text: str) -> str:
        """Preserve document structure markers"""
        # Normalize whitespace but preserve structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n\n', text)  # Preserve section breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        return text
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlap text from end of chunk"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        overlap_tokens_list = tokens[-overlap_tokens:]
        return self.tokenizer.decode(overlap_tokens_list)
    
    def _create_chunk_data(self, text: str, section_idx: int, chunk_idx: int) -> Dict:
        """Create structured chunk data"""
        return {
            "text": text.strip(),
            "section_idx": section_idx,
            "chunk_idx": chunk_idx,
            "token_count": len(self.tokenizer.encode(text)),
            "char_count": len(text)
        }
    
    def generate_advanced_metadata(self, chunk_text: str, file_path: Path, chunk_info: Dict) -> Dict:
        """Generate comprehensive metadata using LLM with Pydantic validation"""
        metadata_logger.info(f"[{file_path.name}] Generating metadata for chunk {chunk_info['section_idx']}-{chunk_info['chunk_idx']}")
        start_time = time.time()
        
        max_text_length = int(os.getenv('METADATA_MAX_TEXT_LENGTH', '1000'))
        truncated_text = chunk_text[:max_text_length] + ("..." if len(chunk_text) > max_text_length else "")
        
        prompt = f"""Analyze this medical/regulatory text and extract metadata as JSON:

Text: {truncated_text}

Return JSON with:
- summary: 2-3 sentence summary
- key_topics: array of 3-5 main topics
- entities: {{people: [], organizations: [], locations: [], dates: []}}
- category: One of ["Drug Safety", "Medical Devices", "Clinical Trials", "Regulatory Guidance", "Warning Letter", "Compliance", "Quality Control", "Manufacturing", "Adverse Events", "Pharmacovigilance"]
- importance_score: 1-10 relevance score
- semantic_tags: array of searchable tags
- content_type: One of ["guidance", "warning", "regulation", "policy", "notice", "report", "document", "letter", "standard", "procedure"]
- regulatory_focus: main regulatory focus areas

Return only valid JSON."""
        
        # Initialize with empty metadata structure
        raw_metadata = {
            "summary": None,
            "key_topics": None,
            "entities": None,
            "category": None,
            "importance_score": None,
            "semantic_tags": None,
            "content_type": None,
            "regulatory_focus": None
        }
        
        # Try to get LLM-generated metadata
        if self.llm is not None:
            try:
                metadata_logger.debug(f"[{file_path.name}] Calling Gemini 1.5 Flash...")
                self.stats.llm_calls += 1
                
                response = self.llm.generate_content(prompt)
                response_text = response.text.strip() if response and response.text else ""
                
                if response_text:
                    metadata_logger.debug(f"[{file_path.name}] LLM response received, parsing JSON...")
                    
                    # Clean the response to extract JSON
                    if response_text.startswith("```json"):
                        response_text = response_text[7:-3]
                    elif response_text.startswith("```"):
                        response_text = response_text[3:-3]
                    
                    # Parse JSON and update metadata
                    llm_metadata = json.loads(response_text)
                    
                    # Normalize LLM response - ensure proper types
                    if "regulatory_focus" in llm_metadata:
                        if isinstance(llm_metadata["regulatory_focus"], str):
                            llm_metadata["regulatory_focus"] = [llm_metadata["regulatory_focus"]]
                    
                    if "key_topics" in llm_metadata:
                        if isinstance(llm_metadata["key_topics"], str):
                            llm_metadata["key_topics"] = [llm_metadata["key_topics"]]
                    
                    if "semantic_tags" in llm_metadata:
                        if isinstance(llm_metadata["semantic_tags"], str):
                            llm_metadata["semantic_tags"] = [llm_metadata["semantic_tags"]]
                    
                    raw_metadata.update(llm_metadata)
                    metadata_logger.info(f"[{file_path.name}] ✓ Metadata generated successfully")
                    
            except Exception as e:
                self.stats.llm_failures += 1
                metadata_logger.warning(f"[{file_path.name}] ✗ LLM processing failed: {e}")
        
        # Add system metadata (always present)
        processing_time = time.time() - start_time
        raw_metadata.update({
            "source_file": str(file_path.relative_to(self.workspace_path)),
            "filename": file_path.name,
            "section_idx": chunk_info["section_idx"],
            "chunk_idx": chunk_info["chunk_idx"],
            "token_count": chunk_info["token_count"],
            "char_count": chunk_info["char_count"],
            "created_at": datetime.now().isoformat(),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "processing_time": processing_time
        })
        
        metadata_logger.info(f"[{file_path.name}] Metadata generation took {processing_time:.2f}s")
        
        # Try Pydantic validation
        try:
            validated_metadata = DocumentMetadata(**raw_metadata)
            metadata_logger.debug(f"[{file_path.name}] ✓ Pydantic validation successful")
            return self._convert_metadata_for_chromadb(validated_metadata.model_dump())
        except Exception as e:
            metadata_logger.warning(f"[{file_path.name}] Pydantic validation failed: {e}")
            return self._convert_metadata_for_chromadb(raw_metadata)
    
    def _convert_metadata_for_chromadb(self, metadata: Dict) -> Dict:
        """Convert metadata to ChromaDB-compatible format"""
        chromadb_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                chromadb_metadata[key] = ""
            elif isinstance(value, list):
                chromadb_metadata[key] = ", ".join(str(item) for item in value) if value else ""
            elif isinstance(value, dict):
                chromadb_metadata[key] = json.dumps(value) if value else "{}"
            elif isinstance(value, (str, int, float, bool)):
                chromadb_metadata[key] = value
            else:
                chromadb_metadata[key] = str(value)
        
        return chromadb_metadata
    
    def improve_text(self, text: str, file_name: str) -> str:
        """Improve text readability using LLM"""
        if os.getenv('ENABLE_TEXT_IMPROVEMENT', 'true').lower() != 'true':
            return text
            
        if self.llm is None:
            return text
            
        chunk_logger.debug(f"[{file_name}] Improving text readability...")
        
        prompt = f"""Improve readability of this medical text while preserving all technical accuracy and meaning:

{text}

Requirements:
- Keep all medical/regulatory terminology
- Improve sentence structure and flow
- Fix grammar and clarity issues
- Return only the improved text"""
        
        try:
            response = self.llm.generate_content(prompt)
            improved_text = response.text.strip() if response and response.text else ""
            if improved_text:
                chunk_logger.debug(f"[{file_name}] ✓ Text improved successfully")
                return improved_text
            else:
                return text
        except Exception as e:
            chunk_logger.debug(f"[{file_name}] Text improvement failed: {e}")
            return text
    
    def create_parent_child_relationships(self, chunks: List[Dict], file_path: Path) -> List[Dict]:
        """Create hierarchical relationships between chunks"""
        chunk_logger.debug(f"[{file_path.name}] Creating parent-child relationships...")
        
        for i, chunk in enumerate(chunks):
            # Parent relationship (previous chunk)
            if i > 0:
                chunk["parent_id"] = f"{file_path.name}_{chunk['section_idx']}_{i-1}"
            
            # Child relationship (next chunk)
            if i < len(chunks) - 1:
                chunk["child_id"] = f"{file_path.name}_{chunk['section_idx']}_{i+1}"
            
            # Sibling relationships (same section)
            siblings = [j for j, c in enumerate(chunks) if c["section_idx"] == chunk["section_idx"] and j != i]
            chunk["sibling_ids"] = [f"{file_path.name}_{chunk['section_idx']}_{j}" for j in siblings]
        
        return chunks
    
    def process_files(self, source_folders: List[str] = None):
        """Process files from specified folders"""
        if source_folders is None:
            folders_env = os.getenv('SOURCE_FOLDERS', 'downloaded_content_text,extracted_content/other_pages,extracted_content/warning_letters,processed_pdfs')
            source_folders = [f.strip() for f in folders_env.split(',')]
        
        logger.info("="*80)
        logger.info("STARTING FILE PROCESSING")
        logger.info(f"Source folders: {source_folders}")
        logger.info("="*80)
        
        total_files = 0
        total_chunks = 0
        
        for folder in source_folders:
            folder_path = self.workspace_path / folder
            if folder_path.exists():
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing folder: {folder}")
                logger.info(f"{'='*60}")
                
                folder_files = 0
                txt_files = list(folder_path.rglob("*.txt"))
                logger.info(f"Found {len(txt_files)} text files in {folder}")
                
                for idx, file_path in enumerate(txt_files, 1):
                    logger.info(f"\n[{idx}/{len(txt_files)}] Processing: {file_path.name}")
                    chunks_added = self.process_file(file_path, folder)
                    if chunks_added > 0:
                        folder_files += 1
                        total_chunks += chunks_added
                
                logger.info(f"\n✓ Completed folder: {folder}")
                logger.info(f"  Files processed: {folder_files}")
                logger.info(f"  Total chunks: {total_chunks}")
                total_files += folder_files
            else:
                logger.warning(f"✗ Folder not found: {folder}")
        
        logger.info("\n" + "="*80)
        logger.info("FILE PROCESSING COMPLETE")
        logger.info(f"Total files processed: {total_files}")
        logger.info(f"Total chunks added: {total_chunks}")
        logger.info("="*80)
        
        # Log final statistics
        self.stats.log_summary()
    
    def process_file(self, file_path: Path, source_folder: str) -> int:
        """Process individual file with advanced chunking and metadata"""
        file_start_time = time.time()
        logger.info(f"[{file_path.name}] Starting file processing...")
        logger.info(f"[{file_path.name}] File size: {file_path.stat().st_size:,} bytes")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"[{file_path.name}] ✗ Error reading file: {e}")
            self.stats.errors += 1
            return 0
        
        # Clean and validate text
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < self.min_text_length:
            logger.info(f"[{file_path.name}] ✗ Skipping - too short ({len(text)} chars)")
            return 0
        
        logger.info(f"[{file_path.name}] Text length: {len(text):,} chars")
        self.stats.files_processed += 1
        
        # Advanced semantic chunking
        chunk_data_list = self.semantic_chunk_text(text, file_path.name)
        if not chunk_data_list:
            logger.warning(f"[{file_path.name}] ✗ No chunks generated")
            return 0
            
        logger.info(f"[{file_path.name}] Generated {len(chunk_data_list)} chunks")
        
        # Create parent-child relationships
        chunk_data_list = self.create_parent_child_relationships(chunk_data_list, file_path)
        
        chunks_added = 0
        
        # Process each chunk
        for chunk_idx, chunk_data in enumerate(chunk_data_list):
            chunk_start_time = time.time()
            try:
                logger.info(f"[{file_path.name}] Processing chunk {chunk_idx + 1}/{len(chunk_data_list)}")
                logger.info(f"[{file_path.name}] Chunk size: {chunk_data['token_count']} tokens, {chunk_data['char_count']} chars")
                
                # Improve text readability
                improved_text = self.improve_text(chunk_data["text"], file_path.name)
                
                # Generate advanced metadata
                metadata = self.generate_advanced_metadata(improved_text, file_path, chunk_data)
                
                # Generate embedding
                logger.info(f"[{file_path.name}] Generating embedding for chunk {chunk_idx + 1}...")
                embedding = self.embedding_model.encode(improved_text).tolist()
                
                # Create unique document ID
                doc_id = hashlib.md5(
                    f"{file_path.name}_{chunk_data['section_idx']}_{chunk_data['chunk_idx']}".encode()
                ).hexdigest()
                
                # Add to ChromaDB
                db_logger.info(f"[{file_path.name}] Storing chunk {chunk_idx + 1} in ChromaDB...")
                self.collection.add(
                    documents=[improved_text],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                chunks_added += 1
                self.stats.chunks_stored += 1
                
                chunk_time = time.time() - chunk_start_time
                logger.info(f"[{file_path.name}] ✓ Chunk {chunk_idx + 1} processed successfully in {chunk_time:.2f}s")
                
                # Progress update
                if (chunk_idx + 1) % 5 == 0 or chunk_idx == len(chunk_data_list) - 1:
                    progress = (chunk_idx + 1) / len(chunk_data_list) * 100
                    logger.info(f"[{file_path.name}] Progress: {progress:.1f}% ({chunk_idx + 1}/{len(chunk_data_list)} chunks)")
                
            except Exception as e:
                logger.error(f"[{file_path.name}] ✗ Error processing chunk {chunk_idx + 1}: {e}")
                self.stats.errors += 1
                continue
        
        file_time = time.time() - file_start_time
        logger.info(f"[{file_path.name}] ✓ File processing complete")
        logger.info(f"[{file_path.name}] Added {chunks_added} chunks in {file_time:.2f}s")
        logger.info(f"[{file_path.name}] Average time per chunk: {file_time/len(chunk_data_list):.2f}s")
        
        return chunks_added

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("RAG BUILDER STARTING")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info("="*80)
    
    try:
        builder = RAGBuilder()
        builder.process_files()
        
        logger.info("="*80)
        logger.info("RAG BUILDER COMPLETED SUCCESSFULLY!")
        logger.info(f"End time: {datetime.now().isoformat()}")
        logger.info("="*80)
    except Exception as e:
        logger.error("="*80)
        logger.error(f"RAG BUILDER FAILED: {e}")
        logger.error("="*80)
        raise
