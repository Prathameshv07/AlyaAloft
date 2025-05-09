"""
Document structure extraction and enhanced metadata generation.
"""

import re
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio

from app import app_config as config
from app.utils.logging_utils import get_logger, log_function_call
from app.prompts.document_prompts import COMMON_SECTION_PATTERNS, DOCUMENT_STRUCTURE_EXTRACTION_PROMPT
from app.utils.ai_utils import ai_response_generator
from app.utils.document_chunking import document_chunker

# Get logger for this module
logger = get_logger(__name__)

class DocumentStructureExtractor:
    """Extract document structure from text like headers, sections, and hierarchy."""
    
    def __init__(self):
        """Initialize the document structure extractor."""
        self.processed_dir = Path(config.DATA_DIR) / "processed"
        self.processed_dir.mkdir(exist_ok=True, parents=True)
    
    @log_function_call
    async def extract_document_structure(self, doc_id: str, full_text: str) -> Dict[str, Any]:
        """
        Extract and analyze document structure from full text.
        
        Args:
            doc_id: Document ID
            full_text: Full document text
            
        Returns:
            Dictionary with structure information
        """
        # Check for cached structure
        cached_structure = self._get_cached_structure(doc_id)
        if cached_structure:
            logger.info(f"Using cached document structure for {doc_id}")
            return cached_structure
        
        # Extract basic structure using regex
        structure = {
            "doc_id": doc_id,
            "sections": await self._extract_sections(full_text),
            "headings": self._extract_headings(full_text),
            "toc": [],  # Will be populated if a table of contents is found
            "ai_analysis": await self._analyze_structure_with_ai(full_text),
        }
        
        # Try to extract table of contents
        toc = self._extract_toc(full_text)
        if toc:
            structure["toc"] = toc
        
        # Extract key phrases
        structure["key_phrases"] = self._extract_key_phrases(full_text)
        
        # Create enhanced chunks using document chunker
        chunks = document_chunker.chunk_document(full_text)
        structure["chunks"] = {
            "count": len(chunks),
            "avg_size": sum(len(chunk) for chunk in chunks) // max(1, len(chunks)),
            "sample": chunks[0] if chunks else ""
        }
        
        # Save to cache
        self._cache_structure(doc_id, structure)
        
        return structure
    
    def _get_cached_structure(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached document structure if available.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Cached structure or None if not found
        """
        cache_file = self.processed_dir / f"{doc_id}_structure.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cached structure: {str(e)}")
        
        return None
    
    def _cache_structure(self, doc_id: str, structure: Dict[str, Any]) -> None:
        """
        Cache document structure for future use.
        
        Args:
            doc_id: Document ID
            structure: Document structure data
        """
        try:
            cache_file = self.processed_dir / f"{doc_id}_structure.json"
            with open(cache_file, "w") as f:
                json.dump(structure, f, indent=2)
        except Exception as e:
            logger.error(f"Error caching structure: {str(e)}")
    
    @log_function_call
    async def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract sections from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of section dictionaries
        """
        sections = []
        lines = text.split("\n")
        
        current_section = {"title": "Introduction", "content": [], "level": 1}
        
        for line in lines:
            line_text = line.strip()
            if not line_text:
                continue
            
            # Check if this line matches a section header pattern
            is_header = False
            header_level = 1
            
            for pattern in COMMON_SECTION_PATTERNS:
                if re.match(pattern, line_text, re.IGNORECASE):
                    is_header = True
                    # Try to determine the header level
                    if re.match(r'^[\d\.]+\s', line_text):
                        # Count the dots to determine level
                        parts = line_text.split(" ")[0].strip()
                        header_level = parts.count(".") + 1
                    
                    # If it's a chapter, it's top level
                    if "chapter" in line_text.lower():
                        header_level = 1
                    
                    break
            
            if is_header:
                # Save current section if it has content
                if current_section["content"]:
                    sections.append({
                        "title": current_section["title"],
                        "content": "\n".join(current_section["content"]),
                        "level": current_section["level"]
                    })
                
                # Start new section
                current_section = {
                    "title": line_text,
                    "content": [],
                    "level": header_level
                }
            else:
                # Add to current section
                current_section["content"].append(line_text)
        
        # Add the last section
        if current_section["content"]:
            sections.append({
                "title": current_section["title"],
                "content": "\n".join(current_section["content"]),
                "level": current_section["level"]
            })
        
        return sections
    
    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract headings from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of heading dictionaries
        """
        headings = []
        lines = text.split("\n")
        
        for i, line in enumerate(lines):
            line_text = line.strip()
            if not line_text:
                continue
            
            # Check if this line matches a header pattern
            for pattern in COMMON_SECTION_PATTERNS:
                if re.match(pattern, line_text, re.IGNORECASE):
                    # Determine level
                    level = 1
                    if re.match(r'^[\d\.]+\s', line_text):
                        parts = line_text.split(" ")[0].strip()
                        level = parts.count(".") + 1
                    
                    # If it's a chapter, it's top level
                    if "chapter" in line_text.lower():
                        level = 1
                    
                    headings.append({
                        "text": line_text,
                        "line_number": i,
                        "level": level
                    })
                    break
        
        return headings
    
    def _extract_toc(self, text: str) -> List[Dict[str, Any]]:
        """
        Try to extract table of contents if it exists.
        
        Args:
            text: Document text
            
        Returns:
            List of TOC entries or empty list if none found
        """
        toc = []
        lines = text.split("\n")
        
        # Look for typical TOC markers
        toc_start = -1
        toc_end = -1
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check for TOC header
            if (line_lower == "table of contents" or 
                line_lower == "contents" or 
                line_lower == "toc") and toc_start == -1:
                toc_start = i
                continue
            
            # Check for end of TOC (usually when we hit Chapter 1 or Introduction)
            if toc_start != -1 and (
                "chapter 1" in line_lower or 
                "introduction" in line_lower or
                "1. introduction" in line_lower or
                i - toc_start > 100  # Limit TOC to reasonable size
            ):
                toc_end = i
                break
        
        # If we found a TOC section
        if toc_start != -1 and toc_end != -1:
            # Process TOC lines
            for i in range(toc_start + 1, toc_end):
                line = lines[i].strip()
                if not line:
                    continue
                
                # Look for patterns like "1. Introduction..........5"
                # or "Chapter 1: Getting Started....10"
                toc_match = re.match(r'([\d\.]+\s+.*?)\.{2,}(\d+)', line)
                if toc_match:
                    title = toc_match.group(1).strip()
                    page = toc_match.group(2).strip()
                    
                    # Determine level from indentation or numbering
                    level = 1
                    if re.match(r'^[\d\.]+\s', title):
                        parts = title.split(" ")[0].strip()
                        level = parts.count(".") + 1
                    
                    toc.append({
                        "title": title,
                        "page": page,
                        "level": level
                    })
                else:
                    # Try to match entries without page numbers
                    toc_simple = re.match(r'([\d\.]+\s+.*?)$', line)
                    if toc_simple:
                        title = toc_simple.group(1).strip()
                        
                        # Determine level from indentation or numbering
                        level = 1
                        if re.match(r'^[\d\.]+\s', title):
                            parts = title.split(" ")[0].strip()
                            level = parts.count(".") + 1
                        
                        toc.append({
                            "title": title,
                            "page": "unknown",
                            "level": level
                        })
        
        return toc
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from text.
        
        Args:
            text: Document text
            
        Returns:
            List of key phrases
        """
        phrases = []
        
        # Look for terms in quotes
        quotes = re.findall(r'"([^"]+)"', text)
        
        # Add phrases that look like terms (not too long, not too short)
        for phrase in quotes:
            if 2 <= len(phrase.split()) <= 5 and len(phrase) <= 50:
                phrases.append(phrase)
        
        # Look for capitalized phrases (potential terms)
        cap_phrases = re.findall(r'(?<!\.\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})', text)
        for phrase in cap_phrases:
            if phrase not in phrases and len(phrase) <= 50:
                phrases.append(phrase)
        
        # Deduplicate and limit
        unique_phrases = list(set(phrases))
        
        # Limit to a reasonable number
        return unique_phrases[:50]
    
    @log_function_call
    async def _analyze_structure_with_ai(self, text: str) -> Dict[str, Any]:
        """
        Use AI to analyze document structure more intelligently.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with AI analysis results
        """
        try:
            # Use enhanced document chunking
            chunks = document_chunker.chunk_document(text[:10000])  # First 10k chars for analysis
            
            # Combine chunks with highest relevance for structure analysis
            structure_text = "\n\n".join(chunks[:3])
            
            # Use our T5 model to analyze document structure
            prompt = DOCUMENT_STRUCTURE_EXTRACTION_PROMPT.format(document_sample=structure_text)
            
            # Generate AI response for structure analysis
            response = await ai_response_generator.generate_response(
                query="Analyze the document structure and provide metadata",
                context=structure_text
            )
            
            # Parse the response
            return self._parse_ai_structure_response(response)
        except Exception as e:
            logger.error(f"Error in AI structure analysis: {str(e)}")
            return {
                "error": str(e),
                "document_type": "unknown",
                "subject_areas": [],
                "estimated_sections": 0
            }
    
    def _parse_ai_structure_response(self, response: str) -> Dict[str, Any]:
        """
        Parse AI response for structure analysis.
        
        Args:
            response: AI response
            
        Returns:
            Dictionary with parsed structure data
        """
        result = {
            "document_type": "unknown",
            "subject_areas": [],
            "estimated_sections": 0,
            "key_topics": [],
            "authors": [],
            "estimated_pages": 0
        }
        
        try:
            # Try to extract key-value pairs
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for key-value patterns
                match = re.match(r'(\w+(?:\s+\w+)*):\s*(.*)', line)
                if match:
                    key = match.group(1).lower().strip()
                    value = match.group(2).strip()
                    
                    if key in ["document type", "type", "document"]:
                        result["document_type"] = value
                    elif key in ["subject", "subject area", "subject areas", "subjects"]:
                        result["subject_areas"] = [s.strip() for s in value.split(",")]
                    elif key in ["sections", "estimated sections", "section count"]:
                        try:
                            result["estimated_sections"] = int(re.search(r'\d+', value).group())
                        except (AttributeError, ValueError):
                            pass
                    elif key in ["topics", "key topics", "main topics"]:
                        result["key_topics"] = [s.strip() for s in value.split(",")]
                    elif key in ["author", "authors"]:
                        result["authors"] = [s.strip() for s in value.split(",")]
                    elif key in ["pages", "estimated pages", "page count"]:
                        try:
                            result["estimated_pages"] = int(re.search(r'\d+', value).group())
                        except (AttributeError, ValueError):
                            pass
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
        
        return result

# Create singleton instance
document_structure_extractor = DocumentStructureExtractor() 