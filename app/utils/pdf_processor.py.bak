"""
PDF processor for text extraction, chunking, and embedding.
"""

import os
import tempfile
import time
import uuid
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict
import asyncio
from datetime import datetime
import json

import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app import app_config as config
from app.storage import json_storage
from app.utils.logging_utils import get_logger, log_function_call
from app.prompts.document_prompts import is_summary_query, COMMON_SECTION_PATTERNS, SUMMARY_PROMPT, STANDARD_DOCUMENT_QUERY_PROMPT, SELF_VERIFICATION_PROMPT

# Define Document type
class Document(TypedDict):
    id: str
    filename: str
    file_path: str
    upload_time: str
    processed: bool
    processing_status: str
    full_text: Optional[str]
    text_length: Optional[int]
    page_count: Optional[int]
    chunks: Optional[List[Dict[str, Any]]]
    chunk_count: Optional[int]
    enhanced_metadata: Optional[Dict[str, Any]]
    vector_data_path: Optional[str]

# Import document structure extractor
from app.utils.document_structure import document_structure_extractor

# Get logger for this module
logger = get_logger(__name__)

# Initialize OCR if available
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract OCR initialized successfully")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning(
        "Tesseract OCR not available. Install with: pip install pytesseract pdf2image"
    )
except Exception as e:
    TESSERACT_AVAILABLE = False
    logger.warning(f"Tesseract OCR not available: {str(e)}")

class PDFProcessor:
    """PDF processor for text extraction, chunking, and embedding."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        # Load embedding model
        self.embedding_model_path = config.MODELS_DIR / "sentence-transformers"
        self.embedding_model = None
        self.load_embedding_model()
        
        # Set OCR availability
        self.ocr_available = TESSERACT_AVAILABLE and config.ENABLE_OCR
    
    @log_function_call
    def load_embedding_model(self) -> None:
        """
        Load the embedding model.
        """
        try:
            model_path = Path(config.MODELS_DIR) / "sentence-transformers"
            if not model_path.exists():
                logger.warning(f"Model directory {model_path} does not exist")
                return
            
            # Check if we need to use the compatibility approach
            config_file = model_path / "1_Pooling/config.json"
            if config_file.exists():
                try:
                    # Try loading with the standard approach first
                    self.embedding_model = SentenceTransformer(str(model_path))
                    logger.info("Embedding model loaded successfully")
                    return
                except TypeError as e:
                    error_msg = str(e)
                    if any(param in error_msg for param in ["pooling_mode_weightedmean_tokens", "pooling_mode_lasttoken", "include_prompt"]):
                        logger.warning("Detected incompatible model format, trying compatibility mode")
                        # Initialize with a compatible model and copy over the files
                        import json
                        import shutil
                        
                        # Load a base model temporarily
                        temp_model = SentenceTransformer("all-MiniLM-L6-v2")
                        
                        # Replace the config.json file in the model path to be compatible
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                        
                        # Remove all problematic parameters if they exist
                        problematic_keys = [
                            'pooling_mode_weightedmean_tokens', 
                            'pooling_mode_lasttoken',
                            'include_prompt'
                        ]
                        
                        for key in problematic_keys:
                            if key in config_data:
                                del config_data[key]
                        
                        # Create a backup of the original file
                        shutil.copy(config_file, str(config_file) + '.backup')
                        
                        # Write the modified config
                        with open(config_file, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        
                        # Try loading again
                        self.embedding_model = SentenceTransformer(str(model_path))
                        logger.info("Embedding model loaded successfully with compatibility mode")
                    else:
                        raise
            else:
                # If config file doesn't exist, load a default model
                logger.warning(f"Config file not found at {config_file}, loading default model")
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Default embedding model loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            self.embedding_model = None
    
    @log_function_call
    async def process_uploaded_file(
        self,
        file_path: Union[str, Path],
        file_name: str,
        process_async: bool = False,
    ) -> str:
        """
        Process an uploaded PDF file.
        
        Args:
            file_path: Path to the PDF file
            file_name: Original file name
            process_async: Whether to process in background
            
        Returns:
            Document ID
        """
        file_path = Path(file_path)
        
        try:
            # Create document record
            doc_obj = {
                "filename": file_name,
                "file_path": str(file_path),
                "upload_time": datetime.now().isoformat(),
                "processed": False,
                "processing_status": "pending",
            }
            
            # Properly await the document creation
            doc_id = await json_storage.create_document(doc_obj)
            
            if process_async:
                # Create a background task that is fully awaitable
                # This prevents the "coroutine was never awaited" error
                loop = asyncio.get_event_loop()
                loop.create_task(self.process_document(doc_id))
            else:
                # Process synchronously
                await self.process_document(doc_id)
            
            # Return the document ID as string
            return str(doc_id)
        
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            raise
    
    @log_function_call
    async def process_document(self, doc_id: str) -> None:
        """
        Process a document.
        
        Args:
            doc_id: Document ID
        """
        try:
            # Get document
            doc = await json_storage.get_document(doc_id)
            if not doc:
                logger.error(f"Document not found: {doc_id}")
                return
            
            # Update status
            await json_storage.update_document(
                doc_id,
                {
                    "processing_status": "processing",
                    "processed": False,
                }
            )
            
            # Extract text
            file_path = doc["file_path"]
            text, page_count = await self.extract_text(file_path)
            
            # Save full text and page count to document
            await json_storage.update_document(
                doc_id,
                {
                    "full_text": text,
                    "text_length": len(text),
                    "page_count": page_count
                }
            )
            
            # Enhance document with metadata and structure extraction
            await self.improve_document_metadata(doc_id, text)
            
            # Chunk text
            chunks = await self.chunk_text(text)
            
            # Create embeddings for chunks
            if self.embedding_model is not None:
                embeddings = await self.embed_text(chunks)
                
                # Save chunks and embeddings
                chunk_objects = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_objects.append({
                        "id": i,
                        "text": chunk,
                        "embedding": embedding.tolist(),
                    })
                
                await json_storage.update_document(
                    doc_id,
                    {
                        "chunks": chunk_objects,
                        "chunk_count": len(chunks),
                    }
                )
                
                # Create the embeddings directory if it doesn't exist
                embeddings_dir = Path(config.DATA_DIR) / "embeddings"
                os.makedirs(embeddings_dir, exist_ok=True)
                
                # Save embeddings to a separate file for easier retrieval
                embeddings_file = embeddings_dir / f"{doc_id}.json"
                embeddings_data = {
                    "chunks": [chunk_obj["text"] for chunk_obj in chunk_objects],
                    "embeddings": [chunk_obj["embedding"] for chunk_obj in chunk_objects]
                }
                
                with open(embeddings_file, 'w') as f:
                    json.dump(embeddings_data, f)
                
                logger.info(f"Embeddings saved to file: {embeddings_file}")
            
            # Update status
            await json_storage.update_document(
                doc_id,
                {
                    "processing_status": "completed",
                    "processed": True,
                }
            )
            
            logger.info(f"Document processed: {doc_id}")
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            
            # Update status
            await json_storage.update_document(
                doc_id,
                {
                    "processing_status": "failed",
                    "processed": False,
                    "error": str(e),
                }
            )
        
        finally:
            # Clean up temporary file after processing (whether success or failure)
            try:
                doc = await json_storage.get_document(doc_id)
                if doc and "file_path" in doc:
                    file_path = Path(doc["file_path"])
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"Temporary file deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")
    
    @log_function_call
    async def improve_document_metadata(self, doc_id: str, full_text: str) -> None:
        """
        Enhance document with additional metadata and structure information.
        
        Args:
            doc_id: Document ID
            full_text: Document full text
        """
        try:
            # Extract document structure
            structure = await document_structure_extractor.extract_document_structure(doc_id, full_text)
            
            # Save metadata to a separate vector database/cache file
            vector_path = Path(config.DATA_DIR) / "processed" / f"{doc_id}_vectors.json"
            with open(vector_path, "w") as f:
                import json
                json.dump({"structure": structure}, f, indent=2)
            
            # Create a summarized document metadata for quick reference
            metadata = {
                "has_structure_data": True,
                "structure_info": {
                    "document_type": structure.get("ai_analysis", {}).get("document_type", "Unknown"),
                    "main_sections": [h["text"] for h in structure.get("headings", [])[:10]],
                    "section_count": len(structure.get("sections", [])),
                    "key_phrases": structure.get("key_phrases", [])[:10],
                    "has_toc": len(structure.get("toc", [])) > 0,
                }
            }
            
            # Update document with enhanced metadata
            await json_storage.update_document(
                doc_id,
                {
                    "enhanced_metadata": metadata,
                    "vector_data_path": str(vector_path),
                }
            )
            
            logger.info(f"Document metadata enhanced: {doc_id}")
        except Exception as e:
            logger.error(f"Error enhancing document metadata: {str(e)}")
            # Continue processing even if metadata enhancement fails
    
    @log_function_call
    async def get_document_structure(self, doc_id: str) -> Dict[str, Any]:
        """
        Get document structure for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document structure data
        """
        try:
            # Try to get from cache first
            vector_path = Path(config.DATA_DIR) / "processed" / f"{doc_id}_vectors.json"
            if vector_path.exists():
                with open(vector_path, "r") as f:
                    import json
                    data = json.load(f)
                    return data.get("structure", {})
            
            # If not in cache, get document and extract structure
            doc = await json_storage.get_document(doc_id)
            if not doc or "full_text" not in doc:
                logger.error(f"Document {doc_id} not found or has no text")
                return {}
            
            # Extract structure
            structure = await document_structure_extractor.extract_document_structure(
                doc_id, doc["full_text"]
            )
            return structure
        
        except Exception as e:
            logger.error(f"Error getting document structure: {str(e)}")
            return {}
    
    @log_function_call
    async def extract_text(self, file_path: Union[str, Path]) -> Tuple[str, int]:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of extracted text and page count
        """
        file_path = Path(file_path)
        
        try:
            # First try normal PDF text extraction
            with open(file_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                page_count = len(pdf.pages)
                
                text_parts = []
                for page_num in range(page_count):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    
                    if text:
                        text_parts.append(text)
                    elif self.ocr_available:
                        # If page has no text, try OCR
                        logger.info(f"Page {page_num + 1} has no text, trying OCR")
                        ocr_text = await self._extract_text_with_ocr(file_path, page_num)
                        text_parts.append(ocr_text)
                
                return "\n\n".join(text_parts), page_count
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            
            # If normal extraction fails and OCR is available, try OCR on the whole document
            if self.ocr_available:
                logger.info("Falling back to OCR for entire document")
                
                text_parts = []
                with open(file_path, "rb") as f:
                    pdf = PyPDF2.PdfReader(f)
                    page_count = len(pdf.pages)
                    
                    for page_num in range(page_count):
                        ocr_text = await self._extract_text_with_ocr(file_path, page_num)
                        text_parts.append(ocr_text)
                
                return "\n\n".join(text_parts), page_count
            
            raise
    
    @log_function_call
    async def _extract_text_with_ocr(self, file_path: Union[str, Path], page_num: int) -> str:
        """
        Extract text from a PDF page using OCR.
        
        Args:
            file_path: Path to the PDF file
            page_num: Page number to extract text from
            
        Returns:
            Extracted text
        """
        if not self.ocr_available:
            logger.warning("OCR not available")
            return ""
        
        try:
            # Convert PDF page to image
            images = convert_from_path(
                str(file_path),
                first_page=page_num + 1,
                last_page=page_num + 1
            )
            
            if not images:
                logger.warning(f"No images extracted from page {page_num + 1}")
                return ""
            
            # Extract text from image
            text = pytesseract.image_to_string(images[0])
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text with OCR: {str(e)}")
            return ""
    
    @log_function_call
    async def chunk_text(self, text, chunk_size=None, chunk_overlap=None):
        """
        Split text into meaningful chunks for vector storage, with improved
        section boundary awareness and overlap for better context.
        
        Args:
            text: Text to chunk
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Use defaults if not specified
        if not chunk_size:
            chunk_size = 500  # Increased from previous value for better context
        
        if not chunk_overlap:
            chunk_overlap = 100  # Increased overlap for better context continuity
            
        # First attempt to detect and respect section boundaries in the text
        # Look for section headers and other structural elements
        section_boundaries = []
        
        # Find all potential section boundaries
        lines = text.split('\n')
        current_position = 0
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                current_position += len(line) + 1  # +1 for newline
                continue
            
            # Check if this line matches any section pattern
            is_section_header = False
            for pattern in COMMON_SECTION_PATTERNS:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_section_header = True
                    break
            
            # If it's a section header and not at the start, add the previous position as a boundary
            if is_section_header and current_position > 0:
                section_boundaries.append(current_position)
            
            current_position += len(line) + 1  # +1 for newline
        
        # Add additional boundaries for logical paragraphs (double newlines)
        paragraph_boundaries = [m.start() for m in re.finditer(r'\n\s*\n', text)]
        
        # Combine all boundaries
        all_boundaries = sorted(set(section_boundaries + paragraph_boundaries))
        
        # If no clear sections found or they're too far apart, fall back to simple chunking
        if not all_boundaries or (len(all_boundaries) >= 2 and 
                                 max([all_boundaries[i+1] - all_boundaries[i] 
                                     for i in range(len(all_boundaries) - 1)]) > chunk_size * 2):
            # Fall back to simple chunking with overlap
            chunks = []
            start = 0
            
            while start < len(text):
                # Get chunk of appropriate size
                end = start + chunk_size
                if end >= len(text):
                    chunks.append(text[start:])
                    break
                
                # Try to find a good break point (end of sentence)
                # Look for period, question mark, or exclamation followed by space or newline
                last_period = text.rfind('. ', start, end)
                last_question = text.rfind('? ', start, end)
                last_exclamation = text.rfind('! ', start, end)
                
                # Also check for paragraph breaks
                last_newline = text.rfind('\n\n', start, end)
                
                # Find the latest good break point
                break_point = max(last_period, last_question, last_exclamation, last_newline)
                
                if break_point > start:
                    # Found a good break point
                    if break_point in [last_period, last_question, last_exclamation]:
                        # Include the punctuation and space
                        chunks.append(text[start:break_point+2])
                    else:
                        # It's a newline
                        chunks.append(text[start:break_point+2])
                    
                    # Next chunk starts with overlap
                    start = break_point + 2 - chunk_overlap
                else:
                    # No good break point, just cut at the chunk size
                    chunks.append(text[start:end])
                    start = end - chunk_overlap
        else:
            # Use detected section boundaries for smarter chunking
            chunks = []
            boundaries = [0] + all_boundaries + [len(text)]
            
            for i in range(len(boundaries) - 1):
                section_start = boundaries[i]
                section_end = boundaries[i + 1]
                section_text = text[section_start:section_end]
                
                # If section is short enough, keep it as one chunk
                if len(section_text) <= chunk_size:
                    chunks.append(section_text)
                else:
                    # Split the section into chunks with overlap
                    start = 0
                    while start < len(section_text):
                        end = start + chunk_size
                        if end >= len(section_text):
                            chunks.append(section_text[start:])
                        break
                
                        # Try to find a good break point
                        last_period = section_text.rfind('. ', start, end)
                        last_question = section_text.rfind('? ', start, end)
                        last_exclamation = section_text.rfind('! ', start, end)
                        last_newline = section_text.rfind('\n', start, end)
                        
                        break_point = max(last_period, last_question, last_exclamation, last_newline)
                        
                        if break_point > start:
                            # Found a good break point
                            chunks.append(section_text[start:break_point+2])
                            start = break_point + 2 - chunk_overlap
                        else:
                            # No good break point
                            chunks.append(section_text[start:end])
                            start = end - chunk_overlap
        
        # Add metadata to chunks for better context
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Try to identify a title or section name for this chunk
            chunk_title = "Unknown section"
            chunk_lines = chunk.split('\n')
        
            # Look for potential section title in the first few lines
            for line in chunk_lines[:3]:
                if line.strip() and any(re.match(pattern, line.strip(), re.IGNORECASE) for pattern in COMMON_SECTION_PATTERNS):
                    chunk_title = line.strip()
                    break
            
            # Add metadata as a header
            metadata = f"[Document section: {chunk_title} | Chunk {i+1} of {len(chunks)}]\n\n"
            enhanced_chunks.append(metadata + chunk)
        
        return enhanced_chunks
    
    @log_function_call
    async def embed_text(self, chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Array of embeddings
        """
        if not chunks:
            return np.array([])
        
        if self.embedding_model is None:
            logger.error("Embedding model not loaded")
            raise ValueError("Embedding model not loaded")
        
        # Create embeddings - this is a blocking operation, run it in a thread pool
        embeddings = await asyncio.to_thread(self.embedding_model.encode, chunks)
        
        return embeddings
    
    @log_function_call
    async def get_relevant_chunks(
        self, doc: Document, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get the most relevant chunks from a document based on a query.
        
        Args:
            doc: Document object
            query: User query
            top_k: Number of top chunks to return
            
        Returns:
            List of relevant chunks with text and similarity scores
        """
        try:
            # Check if document has chunks
            chunks = doc.get("chunks", [])
            if not chunks:
                logger.warning(f"Document {doc['id']} has no chunks")
                return []
            
            # Create embedding for query
            if self.embedding_model is None:
                logger.error("Embedding model not loaded")
                return []
            
            # Create query embedding - this is a blocking operation, run it in a thread pool
            query_embedding = await asyncio.to_thread(self.embedding_model.encode, [query])
            query_embedding = query_embedding[0]  # Get the first (and only) embedding
            
            # Calculate similarity scores with all chunks
            chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
            
            # Calculate cosine similarity - this is a blocking operation, run it in a thread pool
            similarity_scores = await asyncio.to_thread(
                lambda: cosine_similarity([query_embedding], chunk_embeddings)[0]
            )
            
            # Create chunks with similarity scores
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                scored_chunks.append({
                    "text": chunk["text"],
                    "score": float(similarity_scores[i]),
                    "id": chunk.get("id", i)
                })
                
            # Sort by similarity score
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            
            # Get top k chunks
            top_chunks = scored_chunks[:top_k]
                
            # If best chunk has very low similarity, return an empty list
            if top_chunks and top_chunks[0]["score"] < 0.2:
                logger.warning(f"Best chunk has low similarity score: {top_chunks[0]['score']}")
            return []
    
            return top_chunks
        
        except Exception as e:
            logger.error(f"Error getting relevant chunks: {str(e)}")
            return []
    
    @log_function_call
    async def query_document(
        self, doc_id: str, query: str, doc: Optional[Document] = None
    ) -> Dict[str, Any]:
        """
        Query a document using RAG with improved response quality.
        
        Args:
            doc_id: The document ID
            query: The user's query
            doc: Optional Document object if already loaded
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Load document if not provided
            if not doc:
                doc = await json_storage.get_document(doc_id)
            if not doc:
                return {
                    "error": "Document not found",
                    "content": "I couldn't find the document you're looking for. Please try again.",
                }

            # Get relevant chunks
            chunks = await self.get_relevant_chunks(doc, query)
            if not chunks:
                return {
                    "error": "No relevant content found",
                    "content": "I couldn't find any relevant information in the document to answer your question. Please try asking something else.",
                }

            # Clean and combine chunks into a single context
            context = self._clean_and_combine_chunks(chunks)
            
            # Determine if this is a summary query
            from app.prompts.document_prompts import is_summary_query
            if is_summary_query(query):
                # Use the summary prompt for document overview queries
                prompt = SUMMARY_PROMPT.format(context=context, query=query)
            else:
                # Use the standard prompt for specific questions
                prompt = STANDARD_DOCUMENT_QUERY_PROMPT.format(context=context, query=query)

            # Generate response using AI utils
            from app.utils.ai_utils import ai_response_generator
            response = await ai_response_generator.generate_response(query, context)
            
            # Verify response if needed
            from app.prompts.document_prompts import should_verify_response
            if should_verify_response(query, response):
                verification_prompt = SELF_VERIFICATION_PROMPT.format(
                    query=query,
                    context=context,
                    response=response
                )
                verification = await ai_response_generator.generate_response(verification_prompt, context)
                
                # If verification suggests improvements, use them
                if "improved response" in verification.lower():
                    # Extract the improved response from the verification
                    improved_response = verification.split("improved response:", 1)[-1].strip()
                    if improved_response:
                        response = improved_response

            return {
                "content": response,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_count": len(chunks),
                    "total_chunks": len(doc.get("chunks", [])),
                }
            }
        
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            return {
                "error": "Error processing query",
                "content": "I encountered an error while processing your question. Please try again.",
            }

    def _clean_and_combine_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Clean and combine chunks into a single coherent context.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Cleaned and combined context string
        """
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Clean each chunk
        cleaned_texts = []
        for text in texts:
            # Remove chunk identifiers
            text = re.sub(r'\[Document section:.*?\]', '', text)
            text = re.sub(r'Chunk \d+ of \d+', '', text)
            
            # Remove any remaining metadata markers
            text = re.sub(r'\[.*?\]', '', text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            cleaned_texts.append(text)
        
        # Combine chunks with proper spacing
        combined_text = ' '.join(cleaned_texts)
        
        # Final cleanup
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        return combined_text


# Initialize PDF processor as a singleton
pdf_processor = PDFProcessor()
