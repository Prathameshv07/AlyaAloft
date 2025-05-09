"""
Advanced prompt engineering for T5 model with comprehensive examples.
This module dramatically improves response quality through structured prompting.
"""

import re
import json
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple

from app.config.model_config import PIPELINE_CONFIG
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Import existing examples from the current file
try:
    from app.utils.prompt_manager import (
        NLP_EXAMPLES, LINGUISTICS_EXAMPLES, COMPUTER_SCIENCE_EXAMPLES,
        GENERAL_KNOWLEDGE_EXAMPLES, AMBIGUITY_EXAMPLES, DOMAIN_EXAMPLES,
        DETAILED_QUESTION_TEMPLATE, STEP_BY_STEP_REASONING_TEMPLATE
    )
except ImportError:
    # Define example sets if not available
    # Define domain-specific examples
    NLP_EXAMPLES = [
        {
            "question": "What is NLP?",
            "answer": "NLP (Natural Language Processing) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves developing algorithms and models that enable computers to understand, interpret, and generate human language in a valuable way. NLP combines computational linguistics, machine learning, and deep learning to process and analyze large amounts of natural language data."
        },
        {
            "question": "Explain the concept of tokenization in NLP.",
            "answer": "Tokenization in NLP is the process of breaking down text into smaller units called tokens. These tokens can be words, characters, or subwords. Tokenization is a fundamental preprocessing step in natural language processing that converts raw text into a format that can be processed by algorithms. For example, the sentence 'NLP is fascinating' would be tokenized into ['NLP', 'is', 'fascinating']."
        },
        {
            "question": "What are the key components of a transformer architecture in NLP and how do they function together to process language?",
            "answer": "A transformer architecture in NLP consists of several key components working together: 1) Input Embeddings convert tokens to vector representations, 2) Positional Encodings add position information since transformers lack inherent sequence understanding, 3) Multi-Head Self-Attention mechanisms allow the model to focus on different parts of the input simultaneously, calculating relationships between all tokens, 4) Feed-Forward Networks process each position independently through fully connected layers, 5) Layer Normalization stabilizes the learning process, 6) Residual Connections help combat vanishing gradients by adding the input of a sub-layer to its output. These components combine to create powerful language models capable of understanding context across long sequences, where the self-attention mechanism is particularly important for capturing dependencies regardless of distance between words."
        }
    ]

    LINGUISTICS_EXAMPLES = [
        {
            "question": "What is ambiguity in linguistics?",
            "answer": "Ambiguity in linguistics refers to the potential for multiple interpretations or meanings in a word, phrase, or sentence. It occurs when language can be understood in more than one way. Linguistic ambiguity is a fundamental feature of human languages and is studied extensively in semantics and pragmatics."
        },
        {
            "question": "Explain lexical ambiguity with examples.",
            "answer": "Lexical ambiguity is a type of ambiguity that occurs when a word has multiple meanings. For example, the word 'bank' can refer to a financial institution, the side of a river, or to tilt or incline. Another example is 'bat', which can be a flying mammal or a piece of sports equipment. Lexical ambiguity is resolved through context, which helps listeners/readers determine the intended meaning."
        },
        {
            "question": "What is structural ambiguity?",
            "answer": "Structural ambiguity (also called syntactic ambiguity) occurs when a phrase or sentence can be interpreted in different ways due to its grammatical structure. For example, the sentence 'I saw the man with the telescope' could mean either that I used a telescope to see the man, or that I saw a man who had a telescope. Unlike lexical ambiguity, structural ambiguity arises from the relationship between words rather than from individual word meanings."
        }
    ]

    COMPUTER_SCIENCE_EXAMPLES = [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a branch of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data. Instead of being explicitly programmed to perform a task, these systems learn from experience and improve their performance over time. Machine learning algorithms build mathematical models based on sample data to make predictions or decisions without being explicitly programmed to do so."
        },
        {
            "question": "Explain deep learning.",
            "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to analyze various factors of data. It's particularly powerful for processing unstructured data like images, text, and audio. Deep learning models have revolutionized many fields, enabling breakthroughs in speech recognition, computer vision, natural language processing, and many other domains. The 'deep' in deep learning refers to the use of multiple layers in the neural network."
        }
    ]

    GENERAL_KNOWLEDGE_EXAMPLES = [
        {
            "question": "What is climate change?",
            "answer": "Climate change refers to significant, long-term changes in the global climate. The term can refer to natural or human-induced changes. However, in current usage, it's primarily used to describe changes driven by human activities, especially the burning of fossil fuels, which increases heat-trapping greenhouse gas levels in Earth's atmosphere. These human-produced greenhouse gases are causing Earth's average temperature to rise, resulting in climate disruptions including more frequent and severe storms, droughts, and heat waves."
        }
    ]

    AMBIGUITY_EXAMPLES = [
        {
            "question": "Explain one type of ambiguity and provide an example of that type.",
            "answer": "Lexical ambiguity is one of the main types of ambiguity in linguistics. It occurs when a single word has multiple meanings, leading to different interpretations of a sentence. For example, in the sentence 'I went to the bank,' the word 'bank' is lexically ambiguous because it could refer to a financial institution or the side of a river. Another example is 'The coach gave the team a bat.' Here, 'bat' could mean a baseball bat or a flying mammal, creating ambiguity in the meaning of the sentence."
        }
    ]

    DOMAIN_EXAMPLES = {
        "nlp": NLP_EXAMPLES,
        "linguistics": LINGUISTICS_EXAMPLES,
        "computer_science": COMPUTER_SCIENCE_EXAMPLES,
        "general": GENERAL_KNOWLEDGE_EXAMPLES,
        "ambiguity": AMBIGUITY_EXAMPLES
    }

    # New templates for handling complex, detailed questions
    DETAILED_QUESTION_TEMPLATE = """You are a highly knowledgeable expert. The following question requires a detailed, comprehensive answer with technical precision:

Context: {context}

Question: {query}

Please provide an in-depth answer that:
1. Addresses all aspects of the question
2. Includes technical details and precise terminology
3. Organizes information in a logical structure
4. Prioritizes accuracy and completeness
5. Provides examples where appropriate

Detailed answer:"""

    STEP_BY_STEP_REASONING_TEMPLATE = """You are an expert analytical thinker. The following complex question requires systematic reasoning:

Context: {context}

Complex question: {query}

Please analyze this step-by-step:
1. First, identify the key components of the question
2. For each component, examine the relevant information from the context
3. Build logical connections between these components
4. Consider implications and edge cases
5. Arrive at a comprehensive conclusion that fully addresses the question

Detailed reasoning:"""

# New advanced templates for extremely complex questions
EXPERT_ANALYSIS_TEMPLATE = """You are a leading expert in this domain with decades of specialized knowledge. This extremely complex question requires the highest level of expertise:

Context: {context}

Expert-level question: {query}

Frame your response as a scholarly analysis that:
1. Identifies the fundamental theoretical principles involved
2. Examines methodological considerations and their implications
3. Analyzes competing frameworks and their strengths/weaknesses
4. Integrates cross-disciplinary perspectives where relevant
5. Addresses limitations in current understanding
6. Suggests directions for further investigation

Expert analysis:"""

ADVANCED_SYNTHESIS_TEMPLATE = """You are tasked with synthesizing complex information across multiple domains and theoretical frameworks. This challenging question requires integrative thinking:

Context: {context}

Complex synthesis question: {query}

Your synthesis should:
1. Identify relevant theoretical frameworks from multiple domains
2. Extract core principles and methodologies from each framework
3. Create conceptual bridges between seemingly disparate perspectives
4. Develop an integrated framework that resolves apparent contradictions
5. Apply this synthesized framework to address the specific question
6. Acknowledge limitations and boundary conditions of your synthesis

Integrated synthesis:"""

NLP_TECHNICAL_TEMPLATE = """As a computational linguist with expertise in NLP, provide a technically precise analysis of this specialized question:

Context: {context}

NLP technical question: {query}

Your analysis should include:
1. Relevant algorithms, models, and mathematical formulations
2. Data structures and computational complexity considerations
3. Implementation considerations and potential optimizations
4. Evaluation methodologies and metrics
5. Current state-of-the-art approaches and their limitations
6. Future research directions

Technical NLP analysis:"""

class PromptManager:
    """
    Advanced prompt engineering manager for generating high-quality responses.
    """
    
    def __init__(self):
        """Initialize the prompt manager."""
        # Domain-specific keywords for detection
        self.domain_keywords = {
            "nlp": ["nlp", "natural language processing", "language model", "tokenization", "stemming", "lemmatization", 
                    "named entity", "sentiment analysis", "speech recognition", "language understanding", "token", 
                    "embedding", "transformer", "bert", "gpt", "word2vec", "language understanding", "text classification",
                    "sequence tagging", "named entity recognition", "ner", "pos tagging", "part of speech", "dependency parsing",
                    "semantic role labeling", "coreference resolution", "discourse analysis", "language generation",
                    "neural language model", "word embeddings", "contextual embeddings", "attention mechanism",
                    "transfer learning", "fine-tuning", "zero-shot learning", "few-shot learning", "prompt engineering"],
            "linguistics": ["linguistics", "syntax", "semantics", "phonetics", "morphology", "grammar", "language", 
                           "pragmatics", "discourse", "phonology", "morpheme", "phoneme", "lexicon", "inflection",
                           "derivation", "affix", "prefix", "suffix", "infix", "allomorph", "language family", 
                           "syntax tree", "constituent", "phrase structure", "case marking", "agreement", "tense", "aspect",
                           "mood", "voice", "diachronic", "synchronic", "descriptive linguistics", "prescriptive",
                           "computational linguistics", "psycholinguistics", "sociolinguistics", "historical linguistics",
                           "comparative linguistics", "cognitive linguistics", "generative grammar", "universal grammar",
                           "minimalist program", "construction grammar", "lexical-functional grammar", "head-driven phrase structure"],
            "ambiguity": ["ambiguity", "ambiguous", "multiple meaning", "unclear meaning", "double meaning",
                         "lexical ambiguity", "structural ambiguity", "syntactic ambiguity", "semantic ambiguity",
                         "homonym", "polysemy", "vague", "vagueness", "interpretation", "garden path sentence",
                         "attachment ambiguity", "scope ambiguity", "quantifier scope", "disambiguation",
                         "pragmatic ambiguity", "contextual disambiguation"],
            "computer_science": ["algorithm", "data structure", "programming", "software", "hardware", "database", 
                                "machine learning", "ai", "artificial intelligence", "neural network", "deep learning",
                                "cloud computing", "computer", "code", "compiler", "operating system", "complexity",
                                "big o notation", "recursion", "iteration", "object-oriented", "functional programming",
                                "distributed systems", "parallel computing", "quantum computing", "blockchain",
                                "computational complexity", "np-complete", "optimization", "heuristic", "computational theory",
                                "automata theory", "formal language", "graph theory", "cryptography", "information theory",
                                "computer architecture", "concurrency", "memory management", "garbage collection",
                                "virtualization", "containerization", "microservices", "serverless", "web development"]
        }
        
        # Add new advanced domains
        self.domain_keywords.update({
            "computational_linguistics": ["computational linguistics", "natural language understanding", "formal grammar",
                                         "statistical language modeling", "dependency parsing", "constituency parsing",
                                         "grammar induction", "weighted finite-state transducer", "probabilistic context-free grammar",
                                         "lexicalized grammar", "tree-adjoining grammar", "combinatory categorial grammar",
                                         "head-driven phrase structure grammar", "lexical functional grammar", "unification grammar",
                                         "semantic parsing", "distributional semantics", "frame semantics", "lexical semantics"],
            "multimodal_processing": ["multimodal", "cross-modal", "vision-language", "text-to-image", "image-to-text",
                                     "speech recognition", "text-to-speech", "speech-to-text", "audio-visual", "vision-language",
                                     "multi-modal integration", "cross-modal transfer", "multimodal fusion", "multimodal alignment",
                                     "multimodal embedding", "cross-modal retrieval", "audio-visual synchronization"]
        })
        
        # Load examples (could be from file in a more complex implementation)
        self.examples = DOMAIN_EXAMPLES
        
        # Keywords for query type detection
        self.query_types = {
            "definition": ["what is", "define", "meaning of", "definition of", "can you tell me what", "explain what"],
            "process_explanation": ["how does", "how do", "how is", "how are", "process of", "steps in", "stages of", "mechanism of"],
            "comparison": ["difference between", "compare", "contrast", "versus", "vs", "similarities between", "differences between", "distinguish between"],
            "example_request": ["example of", "examples of", "instance of", "give me an example", "provide an example", "show me an example", "illustrate with"],
            "cause_effect": ["why does", "why do", "cause of", "effect of", "reason for", "impact of", "consequence of", "result in", "leads to"],
            "classification": ["types of", "categories of", "classes of", "kinds of", "forms of", "classifications of", "varieties of"],
            "analysis": ["analyze", "examine", "investigate", "explore", "evaluate", "assess", "critique", "review", "discuss"],
            "synthesis": ["synthesize", "integrate", "combine", "merge", "unify", "bring together", "connect", "relate"],
            "historical": ["history of", "evolution of", "development of", "origins of", "background of", "timeline of"],
            "application": ["application of", "use of", "usage of", "implement", "apply", "practical use", "in practice", "real-world"]
        }
        
        # Advanced query types for detecting highly complex questions
        self.advanced_query_types = {
            "theoretical_analysis": ["theoretical analysis", "theoretical framework", "theoretical perspective", "theoretical approach", 
                                    "theoretical foundation", "theoretical basis", "conceptual framework", "conceptual analysis",
                                    "philosophical foundation", "philosophical perspective", "epistemological", "ontological"],
            "methodological_critique": ["methodological critique", "critique methodology", "evaluate methodology", "methodological approach",
                                       "methodological framework", "research design", "experimental design", "study design",
                                       "methodological limitations", "methodological challenges", "validity", "reliability"],
            "interdisciplinary_synthesis": ["interdisciplinary", "cross-disciplinary", "multidisciplinary", "transdisciplinary",
                                           "across disciplines", "across fields", "spanning multiple domains", "integrating perspectives"],
            "meta_analysis": ["meta-analysis", "systematic review", "literature review", "state of the art", "current research landscape",
                             "critical review", "evaluative synthesis", "research synthesis", "evidence synthesis"]
        }
        
        # Add these advanced query types to the main query types
        self.query_types.update(self.advanced_query_types)
        
        # New detection for complexity level
        self.complexity_indicators = {
            "high": ["comprehensive", "detailed", "thorough", "in-depth", "extensively", "analyze in detail", 
                    "complex interplay", "intricate", "nuanced", "sophisticated", "underlying mechanisms", 
                    "theoretical implications", "comprehensive analysis", "elaborate on", "synthesize",
                    "advanced concept", "critically evaluate", "multifaceted", "interdisciplinary"],
            "technical": ["technical", "technically", "specifications", "mechanism", "architecture", "framework",
                         "implementation details", "algorithm", "protocol", "methodology", "procedure",
                         "at the bit level", "low-level", "mathematical model", "statistical", "theoretical foundation",
                         "computation", "mathematically", "formal definition"]
        }
        
        # Add indicators for extremely complex questions
        self.complexity_indicators["extreme"] = [
            "paradigm shift", "theoretical revolution", "epistemological foundations", "ontological commitment",
            "methodological triangulation", "hermeneutic analysis", "dialectical approach", "phenomenological reduction",
            "poststructuralist critique", "metanarrative", "deconstructionist", "interdisciplinary synthesis",
            "cross-paradigm integration", "meta-theoretical", "theoretical reconceptualization", "explanatory framework",
            "grand unified theory", "theoretical reconciliation", "competing paradigms", "conceptual abstraction",
            "theoretical unification", "emergent properties", "systemic perspective", "holistic integration"
        ]
    
    def detect_domain(self, query: str) -> str:
        """
        Detect the domain of a query based on keywords.
        
        Args:
            query: The user query
            
        Returns:
            Detected domain or "general" if no specific domain is detected
        """
        query_lower = query.lower()
        
        # Special case processing for common queries
        if "nlp" in query_lower and len(query_lower) < 20:
            logger.info("Detected domain: nlp (high confidence)")
            return "nlp"
            
        if "ambiguity" in query_lower:
            logger.info("Detected domain: ambiguity (high confidence)")
            return "ambiguity"
        
        # Check each domain's keywords
        domain_scores = {"general": 1}  # Start with a baseline score for general
        
        for domain, keywords in self.domain_keywords.items():
            # Initialize domain score
            domain_scores[domain] = 0
            
            # Check each keyword
            for keyword in keywords:
                if keyword in query_lower:
                    # Add to score based on specificity (longer keywords are more specific)
                    domain_scores[domain] += len(keyword) / 5
                    
                    # Multiple occurrences increase score
                    occurrences = query_lower.count(keyword)
                    if occurrences > 1:
                        domain_scores[domain] += occurrences - 1
                        
                    logger.debug(f"Keyword match: {keyword} for domain {domain}, score: {domain_scores[domain]}")
        
        # Get the domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        
        # Only return a specific domain if its score is significantly higher than "general"
        if domain_scores[best_domain] > domain_scores.get("general", 0) + 0.5:
            logger.info(f"Detected domain: {best_domain} with score {domain_scores[best_domain]}")
            return best_domain
        
        logger.info("No specific domain detected with high confidence, using general")
        return "general"
    
    def detect_query_type(self, query: str) -> str:
        """
        Detect the type of query based on keywords.
        
        Args:
            query: The user query
            
        Returns:
            Query type or "general" if no specific type is detected
        """
        query_lower = query.lower()
        
        # Check each query type's keywords
        type_scores = {"general": 1}  # Start with a baseline score for general
        
        for q_type, patterns in self.query_types.items():
            # Initialize type score
            type_scores[q_type] = 0
            
            # Check each pattern
            for pattern in patterns:
                if pattern in query_lower:
                    # Position affects relevance (patterns at beginning are stronger indicators)
                    position = query_lower.find(pattern)
                    position_weight = 1.0 if position < 10 else 0.8
                    
                    # Add to score based on specificity and position
                    type_scores[q_type] += (len(pattern) / 5) * position_weight
                    
                    logger.debug(f"Pattern match: {pattern} for type {q_type}, score: {type_scores[q_type]}")
        
        # Get the type with highest score
        best_type = max(type_scores, key=type_scores.get)
        
        # Only return a specific type if its score is meaningfully higher than "general"
        if type_scores[best_type] > type_scores.get("general", 0) + 0.3:
            logger.info(f"Detected query type: {best_type} with score {type_scores[best_type]}")
            return best_type
        
        logger.info("No specific query type detected with high confidence, using general")
        return "general"
    
    def detect_complexity(self, query: str) -> Dict[str, float]:
        """
        Detect the complexity level of a query.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with complexity scores
        """
        query_lower = query.lower()
        complexity = {
            "length": min(1.0, len(query.split()) / 20),  # Normalize by max expected length
            "high_complexity": 0.0,
            "technical": 0.0,
            "extreme": 0.0
        }
        
        # Check for high complexity indicators
        for indicator in self.complexity_indicators["high"]:
            if indicator in query_lower:
                complexity["high_complexity"] += 0.2
                logger.debug(f"High complexity indicator found: {indicator}")
        
        # Check for technical indicators
        for indicator in self.complexity_indicators["technical"]:
            if indicator in query_lower:
                complexity["technical"] += 0.2
                logger.debug(f"Technical indicator found: {indicator}")
        
        # Check for extreme complexity indicators
        for indicator in self.complexity_indicators["extreme"]:
            if indicator in query_lower:
                complexity["extreme"] += 0.25
                logger.debug(f"Extreme complexity indicator found: {indicator}")
                
        # Check for sentence structure complexity
        # Long sentences with multiple clauses indicate complexity
        sentences = re.split(r'[.!?]', query)
        for sentence in sentences:
            if len(sentence.split()) > 20:
                complexity["high_complexity"] += 0.15
                logger.debug(f"Complex sentence structure detected: {len(sentence.split())} words")
            
            # Check for conjunctions and subordinating clauses
            clause_indicators = ["and", "or", "but", "however", "although", "because", "since", "while", "whereas"]
            clause_count = sum(1 for word in sentence.lower().split() if word in clause_indicators)
            if clause_count >= 2:
                complexity["high_complexity"] += 0.1 * min(clause_count, 3)
                logger.debug(f"Multiple clauses detected: {clause_count} conjunctions/subordinators")
        
        # Advanced linguistic feature detection
        academic_terms = ["methodology", "theoretical", "empirical", "paradigm", "framework", "epistemological", 
                         "ontological", "conceptual", "analytical", "synthesis", "critique", "discourse", 
                         "dialectical", "hermeneutic", "phenomenological", "teleological", "axiological"]
        
        academic_term_count = sum(1 for term in academic_terms if term in query_lower)
        if academic_term_count >= 1:
            complexity["high_complexity"] += 0.1 * min(academic_term_count, 5)
            complexity["extreme"] += 0.1 * min(academic_term_count, 5)
            logger.debug(f"Academic terminology detected: {academic_term_count} terms")
        
        # Cap scores at 1.0
        complexity["high_complexity"] = min(1.0, complexity["high_complexity"])
        complexity["technical"] = min(1.0, complexity["technical"])
        complexity["extreme"] = min(1.0, complexity["extreme"])
        
        # Overall complexity score with weighted extreme factor
        complexity["overall"] = (complexity["length"] + complexity["high_complexity"] + 
                                complexity["technical"] + (complexity["extreme"] * 1.5)) / 4
        
        # Cap overall score
        complexity["overall"] = min(1.0, complexity["overall"])
        
        logger.info(f"Query complexity: {complexity['overall']:.2f} (length: {complexity['length']:.2f}, high: {complexity['high_complexity']:.2f}, technical: {complexity['technical']:.2f}, extreme: {complexity['extreme']:.2f})")
        
        return complexity
    
    def get_examples_for_domain(self, domain: str, max_examples: int = 2) -> List[Dict[str, str]]:
        """
        Get examples for a specific domain.
        
        Args:
            domain: The domain to get examples for
            max_examples: Maximum number of examples to include
            
        Returns:
            List of examples
        """
        if domain in self.examples:
            examples = self.examples[domain]
            
            # For complex queries, prefer the more detailed examples
            if len(examples) > max_examples:
                # Get one basic example and the most detailed ones
                examples = [examples[0]] + examples[-(max_examples-1):]
            
            return examples[:max_examples]
        
        # Fall back to general examples if domain-specific ones aren't available
        return self.examples["general"][:max_examples]
    
    def format_examples(self, examples: List[Dict[str, str]]) -> str:
        """
        Format examples for inclusion in a prompt.
        
        Args:
            examples: List of example dictionaries with 'question' and 'answer' keys
            
        Returns:
            Formatted examples string
        """
        formatted = "Here are some examples to guide your response:\n\n"
        
        for i, example in enumerate(examples, 1):
            formatted += f"Example {i}:\nQuestion: {example['question']}\nAnswer: {example['answer']}\n\n"
        
        return formatted
    
    def generate_structured_prompt(self, query: str, context: str) -> str:
        """
        Generate a structured prompt with few-shot examples.
        
        Args:
            query: The user query
            context: The document context
            
        Returns:
            Structured prompt
        """
        # Detect domain, query type, and complexity
        domain = self.detect_domain(query)
        query_type = self.detect_query_type(query)
        complexity = self.detect_complexity(query)
        
        # Select appropriate template based on complexity
        template = None
        
        # Handle extremely complex questions with specialized templates
        if complexity["extreme"] > 0.6 or complexity["overall"] > 0.85:
            if "theoretical" in query_type or complexity["extreme"] > 0.7:
                template = EXPERT_ANALYSIS_TEMPLATE
                logger.info("Using expert analysis template for extremely complex question")
            else:
                template = ADVANCED_SYNTHESIS_TEMPLATE
                logger.info("Using advanced synthesis template for extremely complex question")
        elif complexity["overall"] > 0.6:
            template = DETAILED_QUESTION_TEMPLATE
            logger.info("Using detailed question template for complex question")
        
        # For complex technical questions that need step-by-step reasoning
        if complexity["technical"] > 0.5 and complexity["overall"] > 0.5:
            if domain == "nlp" or domain == "computational_linguistics":
                template = NLP_TECHNICAL_TEMPLATE
                logger.info("Using NLP technical template for complex NLP question")
            else:
                template = STEP_BY_STEP_REASONING_TEMPLATE
                logger.info("Using step-by-step reasoning template for complex technical question")
        
        # Get relevant examples - more examples for more complex questions
        max_examples = 2
        if complexity["overall"] > 0.7:
            max_examples = 3
        
        examples = self.get_examples_for_domain(domain, max_examples=max_examples)
        
        # Start building the structured prompt
        prompt = "You are a precise, accurate, and helpful AI assistant with deep expertise across multiple domains. "
        
        # Add specific instructions based on query type
        if query_type == "definition":
            prompt += "Provide a clear and comprehensive definition. Be precise and include key aspects of the concept. "
        elif query_type == "process_explanation":
            prompt += "Explain the process step by step. Be clear about the sequence and why each step matters. "
        elif query_type == "comparison":
            prompt += "Compare the items by highlighting key similarities and differences. Structure your answer with clear points. "
        elif query_type == "example_request":
            prompt += "Provide specific, concrete examples that clearly illustrate the concept. Include varied examples if appropriate. "
        elif query_type == "cause_effect":
            prompt += "Clearly explain the causes and their effects. Be specific about how the causes lead to the effects. "
        elif query_type == "classification":
            prompt += "Organize the types or categories clearly. For each type, provide a brief description and how it differs from others. "
        elif query_type == "analysis":
            prompt += "Analyze the subject thoroughly, examining its components, relationships, and implications. "
        elif query_type == "synthesis":
            prompt += "Synthesize information from multiple perspectives, showing how different aspects connect and interact. "
        elif query_type == "historical":
            prompt += "Present historical developments chronologically, highlighting key events, figures, and turning points. "
        elif query_type == "application":
            prompt += "Explain practical applications clearly, describing how concepts are implemented in real-world situations. "
        elif query_type == "theoretical_analysis":
            prompt += "Provide a rigorous theoretical analysis that examines foundational principles, assumptions, implications, and limitations. "
        elif query_type == "methodological_critique":
            prompt += "Critically examine methodological approaches, addressing design choices, validity, reliability, and potential biases. "
        elif query_type == "interdisciplinary_synthesis":
            prompt += "Integrate perspectives across multiple disciplines, identifying conceptual bridges, methodological differences, and synthesized insights. "
        elif query_type == "meta_analysis":
            prompt += "Systematically analyze existing research, identifying patterns, contradictions, strengths, limitations, and research gaps. "
        
        # Add domain-specific instructions
        if domain == "nlp":
            prompt += "Focus on accurate technical details about natural language processing. Use established NLP terminology and concepts. "
        elif domain == "linguistics":
            prompt += "Use precise linguistic terminology and include relevant linguistic concepts. Distinguish between different schools of thought when applicable. "
        elif domain == "ambiguity":
            prompt += "Explain ambiguity concepts clearly, distinguishing between different types of ambiguity (lexical, structural, semantic, pragmatic). Provide clear examples to illustrate each type discussed. "
        elif domain == "computer_science":
            prompt += "Provide technically accurate information about computer science concepts. Include algorithmic complexity, architectural details, or implementation considerations when relevant. "
        elif domain == "computational_linguistics":
            prompt += "Address both computational and linguistic aspects, discussing formal representations, algorithms, and their connection to linguistic theory. "
        elif domain == "multimodal_processing":
            prompt += "Explain how multiple modalities (text, speech, vision) integrate and the specific challenges of cross-modal processing. "
        
        # Add complexity-based instructions
        if complexity["overall"] > 0.7:
            prompt += "This is a complex question requiring a detailed, comprehensive answer. Structure your response logically, addressing all aspects of the question. "
        
        if complexity["technical"] > 0.5:
            prompt += "Include technical details, precise terminology, and theoretical foundations in your answer. "
        
        # For extremely complex questions
        if complexity["extreme"] > 0.4:
            prompt += "This question requires expert-level analysis. Incorporate advanced theoretical frameworks, address methodological nuances, consider multiple perspectives, and acknowledge limitations. "
        
        # Add examples
        prompt += self.format_examples(examples)
        
        # Apply the selected template or default format
        if template:
            final_prompt = template.format(context=context, query=query)
            logger.info(f"Using specialized template for complex question")
        else:
            # Add the user query and context
            final_prompt = f"{prompt}\n\nContext: {context}\n\nQuestion: {query}\n\nProvide a detailed and accurate answer:"
        
        return final_prompt
    
    def apply_chain_of_thought(self, prompt: str) -> str:
        """
        Apply chain-of-thought prompting technique.
        
        Args:
            prompt: The base prompt
            
        Returns:
            Enhanced prompt with chain-of-thought instructions
        """
        cot_instruction = """

To ensure accuracy, break down your thinking process:

1. First, identify the key aspects of the question and what it's really asking for
2. Extract the most relevant information from the context
3. Consider multiple perspectives or approaches if applicable
4. Reason step-by-step through the implications and connections
5. Check if your preliminary conclusions are fully supported by the context
6. Formulate a comprehensive answer that addresses all parts of the question
7. Review your answer for completeness and accuracy

Develop your answer systematically through these steps:"""
        
        return prompt + cot_instruction
    
    def apply_advanced_reasoning(self, prompt: str) -> str:
        """
        Apply advanced reasoning techniques for extremely complex questions.
        
        Args:
            prompt: The base prompt
            
        Returns:
            Enhanced prompt with advanced reasoning instructions
        """
        advanced_instruction = """

Apply advanced reasoning methodologies to this complex question:

1. Epistemic Analysis: Identify what is known, unknown, and knowable based on the available context
2. Conceptual Decomposition: Break down complex concepts into their constituent components
3. First Principles Thinking: Derive insights from foundational truths rather than analogies
4. Systems Perspective: Consider how components interact within broader systems
5. Counterfactual Reasoning: Explore hypothetical alternatives to understand causal relationships
6. Meta-level Analysis: Reflect on the assumptions and frameworks being applied
7. Synthesis: Integrate multiple theoretical perspectives into a coherent framework
8. Boundary Conditions: Identify where and when your analysis applies and where it doesn't

Organize your response to demonstrate this advanced reasoning process:"""
        
        return prompt + advanced_instruction
    
    def post_process_response(self, response: str, query: str) -> str:
        """
        Clean up and enhance the model's response.
        
        Args:
            response: The raw model response
            query: The original query
            
        Returns:
            Enhanced response
        """
        # Remove any "Question:" prefixes that might have been generated
        response = re.sub(r'^Question:.*?\n', '', response, flags=re.DOTALL)
        
        # Remove any "Context:" prefixes that might have been generated
        response = re.sub(r'^Context:.*?\n', '', response, flags=re.DOTALL)
        
        # Remove any instances of "Answer:" at the beginning
        response = re.sub(r'^Answer:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^Detailed answer:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^Detailed reasoning:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^Expert analysis:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^Integrated synthesis:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^Technical NLP analysis:\s*', '', response, flags=re.IGNORECASE)
        
        # Remove numbered steps from chain-of-thought reasoning
        response = re.sub(r'^\d+\.\s*', '', response, flags=re.MULTILINE)
        
        # Check if the response is incomplete (ends without punctuation)
        if response and response[-1] not in ('.', '?', '!', ':', ';'):
            # Try to find the last complete sentence
            last_period = max(response.rfind('.'), response.rfind('?'), response.rfind('!'))
            if last_period > len(response) * 0.7:  # If we have at least 70% of the response
                response = response[:last_period+1]
        
        # Ensure the first letter is capitalized
        if response and len(response) > 0:
            response = response[0].upper() + response[1:]
        
        # Ensure we're answering the question specifically
        query_type = self.detect_query_type(query)
        if query_type == "comparison" and "versus" not in response.lower() and "compared to" not in response.lower():
            # Add a comparison framing if it's missing
            parts = re.split(r'vs\.?|versus', query, flags=re.IGNORECASE)
            if len(parts) == 2:
                item1 = parts[0].strip()
                item2 = parts[1].strip()
                response = f"When comparing {item1} and {item2}: {response}"
        
        return response.strip()

# Create singleton instance
prompt_manager = PromptManager() 