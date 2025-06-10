import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Dict, Optional
import argparse
import sys
import json
import os
from difflib import SequenceMatcher

class KhmerNameMatcher:
    def __init__(self, names_file: str, model_name: str = "distiluse-base-multilingual-cased-v1"):
        """
        Initialize the KhmerNameMatcher with a file containing Khmer names and an embedding model.
        
        Args:
            names_file: Path to file containing Khmer names (one per line)
            model_name: SentenceTransformer model to use for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.names = self._load_names(names_file)
        self.name_embeddings = None
        self.index = None
        self._build_index()
        
    def _load_names(self, filename: str) -> List[str]:
        """Load names from a file, filtering out empty lines."""
        with open(filename, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f.readlines()]
        return [name for name in names if name]  # Filter out empty lines
    
    def _build_index(self):
        """Build FAISS index for fast similarity search."""
        # Generate embeddings for all names
        self.name_embeddings = self.model.encode(self.names, convert_to_tensor=True)
        
        # Convert to numpy for FAISS
        embeddings_np = self.name_embeddings.cpu().numpy()
        
        # Build FAISS index
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        faiss.normalize_L2(embeddings_np)  # Normalize vectors for cosine similarity
        self.index.add(embeddings_np)
    
    def semantic_match(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Match a potentially noisy OCR output to the closest Khmer names using semantic similarity.
        
        Args:
            query: OCR output of a Khmer name
            top_k: Number of top matches to return
            
        Returns:
            List of (name, similarity_score) tuples
        """
        # Clean query
        query = query.strip()
        if not query:
            return []
            
        # Get embedding for query
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)
        
        # Search the index
        scores, indices = self.index.search(query_embedding_np, min(top_k, len(self.names)))
        
        # Return results
        results = [(self.names[idx], float(score)) for score, idx in zip(scores[0], indices[0])]
        return results

    def character_match(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform character-based matching for names.
        
        Args:
            query: OCR output of a Khmer name
            top_k: Number of top matches to return
            
        Returns:
            List of (name, match_score) tuples
        """
        if not query:
            return []
            
        # Character overlap scoring
        scores = []
        for name in self.names:
            # Calculate character overlap
            query_chars = set(query)
            name_chars = set(name)
            overlap = len(query_chars.intersection(name_chars))
            total = len(query_chars.union(name_chars))
            score = overlap / total if total > 0 else 0
            scores.append((name, score))
            
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def sequence_match(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Use sequence matching for edit distance based matching.
        
        Args:
            query: OCR output of a Khmer name
            top_k: Number of top matches to return
            
        Returns:
            List of (name, match_score) tuples
        """
        if not query:
            return []
            
        # Use sequence matcher for edit distance
        scores = []
        for name in self.names:
            ratio = SequenceMatcher(None, query, name).ratio()
            scores.append((name, ratio))
            
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def hybrid_match(self, query: str, top_k: int = 5, 
                    semantic_weight: float = 0.5,
                    char_weight: float = 0.3,
                    seq_weight: float = 0.2) -> List[Tuple[str, float]]:
        """
        Combine multiple matching strategies for better accuracy.
        
        Args:
            query: OCR output of a Khmer name
            top_k: Number of top matches to return
            semantic_weight: Weight given to semantic search (0-1)
            char_weight: Weight given to character matching (0-1)
            seq_weight: Weight given to sequence matching (0-1)
            
        Returns:
            List of (name, combined_score) tuples
        """
        # If semantic embedding fails, fall back to simpler methods
        try:
            # Ensure weights sum to 1
            total_weight = semantic_weight + char_weight + seq_weight
            semantic_weight = semantic_weight / total_weight
            char_weight = char_weight / total_weight
            seq_weight = seq_weight / total_weight
            
            # Get results from each method
            semantic_results = self.semantic_match(query, top_k=top_k*2)
            char_results = self.character_match(query, top_k=top_k*2)
            seq_results = self.sequence_match(query, top_k=top_k*2)
            
            # Combine results
            result_dict = {}
            
            for name, score in semantic_results:
                result_dict[name] = score * semantic_weight
                
            for name, score in char_results:
                if name in result_dict:
                    result_dict[name] += score * char_weight
                else:
                    result_dict[name] = score * char_weight
                    
            for name, score in seq_results:
                if name in result_dict:
                    result_dict[name] += score * seq_weight
                else:
                    result_dict[name] = score * seq_weight
            
            # Sort by combined score
            combined_results = [(name, score) for name, score in result_dict.items()]
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            return combined_results[:top_k]
        except Exception as e:
            print(f"Warning: Semantic matching failed, falling back to simpler methods: {e}")
            # Fall back to character + sequence matching
            adjusted_char_weight = char_weight / (char_weight + seq_weight)
            adjusted_seq_weight = seq_weight / (char_weight + seq_weight)
            
            char_results = self.character_match(query, top_k=top_k*2)
            seq_results = self.sequence_match(query, top_k=top_k*2)
            
            result_dict = {}
            
            for name, score in char_results:
                result_dict[name] = score * adjusted_char_weight
                
            for name, score in seq_results:
                if name in result_dict:
                    result_dict[name] += score * adjusted_seq_weight
                else:
                    result_dict[name] = score * adjusted_seq_weight
            
            combined_results = [(name, score) for name, score in result_dict.items()]
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            return combined_results[:top_k]
        
    def process_ocr_output(self, ocr_text: str, top_k: int = 3) -> List[Dict]:
        """
        Process OCR output text containing potentially multiple Khmer names.
        
        Args:
            ocr_text: Text from OCR containing Khmer names
            top_k: Number of top matches to return per name
            
        Returns:
            List of dicts with OCR text and matches
        """
        # Split the OCR text into potential name segments
        # This is a simple split by whitespace - you might need a more
        # sophisticated method depending on your OCR output format
        segments = [s.strip() for s in ocr_text.split() if s.strip()]
        
        results = []
        for segment in segments:
            # Skip segments that are too short
            if len(segment) < 2:
                continue
                
            matches = self.hybrid_match(segment, top_k=top_k)
            
            result = {
                "ocr_text": segment,
                "matches": [{"name": name, "score": float(score)} for name, score in matches]
            }
            
            results.append(result)
            
        return results


def main():
    parser = argparse.ArgumentParser(description='Khmer Name OCR Matcher')
    parser.add_argument('--names_file', type=str, default='khmer_names.txt', 
                        help='Path to file containing Khmer names')
    parser.add_argument('--query', type=str, help='Direct OCR text query')
    parser.add_argument('--query_file', type=str, help='File containing OCR text')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top matches to return')
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--model', type=str, default='distiluse-base-multilingual-cased-v1',
                        help='SentenceTransformer model to use')
    parser.add_argument('--weights', type=str, default='0.5,0.3,0.2', 
                        help='Weights for semantic,character,sequence matching (comma separated)')
    parser.add_argument('--no_semantic', action='store_true',
                        help='Skip semantic matching (for low-memory systems)')
    
    args = parser.parse_args()
    
    # Check if we have either query or query file
    if not args.query and not args.query_file:
        print("Error: Either --query or --query_file must be provided")
        sys.exit(1)
    
    # Parse weights
    try:
        semantic_w, char_w, seq_w = map(float, args.weights.split(','))
        
        # If no_semantic flag is set, redistribute weights
        if args.no_semantic:
            total = char_w + seq_w
            char_w = char_w / total * 1.0
            seq_w = seq_w / total * 1.0
            semantic_w = 0.0
            
    except:
        print("Error: Weights must be three comma-separated numbers")
        sys.exit(1)
    
    # Initialize the matcher
    print(f"Initializing Khmer Name Matcher with {args.names_file}...")
    
    try:
        matcher = KhmerNameMatcher(args.names_file, model_name=args.model)
        print(f"Loaded {len(matcher.names)} Khmer names")
    except Exception as e:
        print(f"Error initializing matcher: {e}")
        print("Trying fallback method without semantic matching...")
        
        # Simple fallback implementation using just character and sequence matching
        class SimpleMatcher:
            def __init__(self, names_file):
                with open(names_file, 'r', encoding='utf-8') as f:
                    self.names = [line.strip() for line in f.readlines()]
                self.names = [name for name in names if name]
                
            def hybrid_match(self, query, top_k=5, **kwargs):
                # Character overlap scoring
                char_scores = []
                for name in self.names:
                    query_chars = set(query)
                    name_chars = set(name)
                    overlap = len(query_chars.intersection(name_chars))
                    total = len(query_chars.union(name_chars))
                    score = overlap / total if total > 0 else 0
                    char_scores.append((name, score))
                
                # Sequence matching
                seq_scores = []
                for name in self.names:
                    ratio = SequenceMatcher(None, query, name).ratio()
                    seq_scores.append((name, ratio))
                
                # Combine scores
                result_dict = {}
                for name, score in char_scores:
                    result_dict[name] = score * 0.6
                
                for name, score in seq_scores:
                    if name in result_dict:
                        result_dict[name] += score * 0.4
                    else:
                        result_dict[name] = score * 0.4
                
                # Sort and return top_k
                combined_results = [(name, score) for name, score in result_dict.items()]
                combined_results.sort(key=lambda x: x[1], reverse=True)
                return combined_results[:top_k]
                
            def process_ocr_output(self, ocr_text, top_k=3):
                segments = [s.strip() for s in ocr_text.split() if s.strip()]
                results = []
                for segment in segments:
                    if len(segment) < 2:
                        continue
                    matches = self.hybrid_match(segment, top_k=top_k)
                    result = {
                        "ocr_text": segment,
                        "matches": [{"name": name, "score": float(score)} for name, score in matches]
                    }
                    results.append(result)
                return results
        
        matcher = SimpleMatcher(args.names_file)
        print(f"Loaded {len(matcher.names)} Khmer names (using simple matching only)")
    
    # Process query
    if args.query:
        ocr_text = args.query
    else:
        try:
            with open(args.query_file, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
        except Exception as e:
            print(f"Error reading query file: {e}")
            sys.exit(1)
    
    # Process the OCR output
    results = matcher.process_ocr_output(ocr_text, top_k=args.top_k)
    
    # Output results
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}")
            print("Displaying results to console instead:")
    
    # Display results to console
    for result in results:
        print(f"\nOCR Text: '{result['ocr_text']}'")
        print("Top matches:")
        for i, match in enumerate(result['matches'], 1):
            print(f"  {i}. {match['name']} (score: {match['score']:.4f})")

if __name__ == "__main__":
    main() 