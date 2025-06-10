import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Dict, Optional
import re

class KhmerNameMatcher:
    def __init__(self, names_file: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
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
    
    def match_name(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Match a potentially noisy OCR output to the closest Khmer names.
        
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

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform keyword-based search for names containing parts of the query.
        
        Args:
            query: OCR output of a Khmer name
            top_k: Number of top matches to return
            
        Returns:
            List of (name, match_score) tuples
        """
        if not query:
            return []
            
        # Simple character overlap scoring
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
    
    def hybrid_match(self, query: str, top_k: int = 5, 
                    semantic_weight: float = 0.7) -> List[Tuple[str, float]]:
        """
        Combine semantic search and keyword search for better matching.
        
        Args:
            query: OCR output of a Khmer name
            top_k: Number of top matches to return
            semantic_weight: Weight given to semantic search (0-1)
            
        Returns:
            List of (name, combined_score) tuples
        """
        semantic_results = self.match_name(query, top_k=top_k*2)
        keyword_results = self.keyword_search(query, top_k=top_k*2)
        
        # Combine results
        result_dict = {}
        for name, score in semantic_results:
            result_dict[name] = score * semantic_weight
            
        for name, score in keyword_results:
            if name in result_dict:
                result_dict[name] += score * (1 - semantic_weight)
            else:
                result_dict[name] = score * (1 - semantic_weight)
        
        # Sort by combined score
        combined_results = [(name, score) for name, score in result_dict.items()]
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:top_k]

# Demo function to test the system
def demo_khmer_name_matcher(names_file: str, test_queries: List[str]):
    """Run a demonstration of the Khmer name matcher."""
    print("Initializing Khmer Name Matcher...")
    matcher = KhmerNameMatcher(names_file)
    print(f"Loaded {len(matcher.names)} Khmer names")
    
    print("\n===== DEMONSTRATION =====")
    for query in test_queries:
        print(f"\nInput (simulated OCR output): '{query}'")
        
        # Get hybrid matches
        matches = matcher.hybrid_match(query, top_k=3)
        
        print("Top matches:")
        for i, (name, score) in enumerate(matches, 1):
            print(f"  {i}. {name} (score: {score:.4f})")
    
    print("\nSystem ready for use with real OCR output.")

if __name__ == "__main__":
    # Example test queries - simulating noisy OCR outputs
    test_queries = [
        "លឹមហ៊ន",  # Slightly misspelled "លឹមហ៊ុន"
        "សុខ",     # Exact match
        "វុឌ្ឍាស", # Partial name "វុឌ្ឍាសុភា"
        "ឡេងហង",  # Similar to "ឡេងហេង"
    ]
    
    demo_khmer_name_matcher("khmer_names.txt", test_queries) 