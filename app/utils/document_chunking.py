"""
Enhanced document chunking utilities for T5 model.
Provides improved chunking strategies to make documents more manageable for T5 while maintaining context.
"""

import re
from typing import List, Dict, Any, Optional
import logging
import nltk
from app.config.model_config import PIPELINE_CONFIG

# Try to download nltk resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class EnhancedDocumentChunker:
    """Enhanced document chunking strategy for better T5 comprehension."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Amount of overlap between chunks in characters
        """
        self.chunk_size = chunk_size or PIPELINE_CONFIG.get("chunk_size", 1024)
        self.chunk_overlap = chunk_overlap or PIPELINE_CONFIG.get("chunk_overlap", 256)
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Split document into semantically coherent chunks.
        
        Args:
            text: Document text to chunk
        
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        # First try to split by sections
        sections = self._split_by_sections(text)
        
        chunks = []
        for section in sections:
            # If section is small enough, add it as is
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue
            
            # Otherwise, chunk the section with attention to paragraph and sentence boundaries
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)
        
        # Combine very small chunks
        chunks = self._combine_small_chunks(chunks)
        
        # Ensure no chunk is too big
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # If still too big, split by size with overlap
                final_chunks.extend(self._split_by_size(chunk))
        
        logger.info(f"Split document into {len(final_chunks)} chunks")
        return final_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace and normalizing."""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split document by section headings."""
        # This regex looks for common section headings (Chapter X, Section X, X. Title, etc.)
        section_pattern = r'(?:\n\s*|\A)((?:Chapter|Section|CHAPTER|SECTION)\s+\d+[.:)]?|(?:\d+[.:)])\s+\w+\s*?(?:\n|$))'
        
        sections = []
        last_match_end = 0
        
        for match in re.finditer(section_pattern, text):
            # Get the text from the last section end to this section start
            if last_match_end > 0:  # Skip the first one - it's just the beginning of the document
                section_text = text[last_match_end:match.start()]
                sections.append(section_text.strip())
            
            last_match_end = match.start()
        
        # Add the final section
        if last_match_end < len(text):
            sections.append(text[last_match_end:].strip())
        
        # If no sections found, return the whole text as one section
        if not sections:
            return [text]
        
        return sections
    
    def _chunk_section(self, section: str) -> List[str]:
        """Split a section into chunks respecting paragraph boundaries."""
        # Split by paragraphs
        paragraphs = [p for p in section.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds the chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If the paragraph itself is too big
                if len(paragraph) > self.chunk_size:
                    sentences = nltk.sent_tokenize(paragraph)
                    sentence_chunks = self._chunk_sentences(sentences)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Split a list of sentences into chunks."""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds the chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If the sentence itself is too big, split it by size
                if len(sentence) > self.chunk_size:
                    chunks.extend(self._split_by_size(sentence))
                    current_chunk = ""
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _combine_small_chunks(self, chunks: List[str]) -> List[str]:
        """Combine small chunks to reduce the number of chunks."""
        if not chunks:
            return []
        
        # If there's only one chunk, return it
        if len(chunks) == 1:
            return chunks
        
        combined_chunks = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            # If combining would be smaller than chunk_size
            if len(current_chunk) + len(chunks[i]) + 2 <= self.chunk_size:  # +2 for newline chars
                current_chunk += "\n\n" + chunks[i]
            else:
                combined_chunks.append(current_chunk)
                current_chunk = chunks[i]
        
        # Add the last chunk
        if current_chunk:
            combined_chunks.append(current_chunk)
        
        return combined_chunks
    
    def _split_by_size(self, text: str) -> List[str]:
        """Split text by size with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            if end > text_length:
                end = text_length
            
            # Get chunk
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move to next position with overlap
            start = end - self.chunk_overlap
            if start < 0 or start >= text_length:
                break
        
        return chunks
    
    def get_most_relevant_chunks(self, chunks: List[str], query: str, n: int = None) -> List[str]:
        """
        Get the most relevant chunks for a query using a simple keyword matching strategy.
        
        Args:
            chunks: List of document chunks
            query: User query
            n: Maximum number of chunks to return
        
        Returns:
            List of most relevant chunks
        """
        if not chunks:
            return []
        
        max_chunks = n or PIPELINE_CONFIG.get("max_context_chunks", 4)
        
        # Prepare query words (remove stopwords and punctuation)
        query = re.sub(r'[^\w\s]', '', query.lower())
        query_words = set(query.split())
        
        # Remove common words
        stopwords = {"what", "is", "the", "a", "an", "and", "or", "of", "in", "to", "how", "why", "when", "where"}
        query_words = query_words - stopwords
        
        # If no meaningful query words left, return the first chunks
        if not query_words:
            return chunks[:max_chunks]
        
        # Score chunks based on word overlap
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_text = re.sub(r'[^\w\s]', '', chunk.lower())
            chunk_words = set(chunk_text.split())
            
            # Score based on how many query words appear in the chunk
            score = sum(1 for word in query_words if word in chunk_words)
            chunk_scores.append((i, score))
        
        # Sort by score (descending)
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top chunks
        top_chunk_indices = [idx for idx, _ in chunk_scores[:max_chunks]]
        top_chunk_indices.sort()  # Keep original order
        
        return [chunks[idx] for idx in top_chunk_indices]

# Create a singleton instance
document_chunker = EnhancedDocumentChunker() 