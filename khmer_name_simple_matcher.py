import argparse
import json
import sys
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

class SimpleKhmerNameMatcher:
    def __init__(self, names_file: str):
        """
        Initialize a simple Khmer name matcher without neural models.
        
        Args:
            names_file: Path to file containing Khmer names (one per line)
        """
        self.names = self._load_names(names_file)
        
    def _load_names(self, filename: str) -> List[str]:
        """Load names from a file, filtering out empty lines."""
        with open(filename, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f.readlines()]
        return [name for name in names if name]  # Filter out empty lines
    
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
    
    def subseq_match(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Look for names where query is a subsequence.
        
        Args:
            query: OCR output of a Khmer name
            top_k: Number of top matches to return
            
        Returns:
            List of (name, match_score) tuples
        """
        if not query:
            return []
            
        scores = []
        for name in self.names:
            # If query is a subsequence of name
            # Convert both to lists of characters
            q_chars = list(query)
            n_chars = list(name)
            
            # Try to find subsequence
            i, j = 0, 0
            while i < len(q_chars) and j < len(n_chars):
                if q_chars[i] == n_chars[j]:
                    i += 1
                j += 1
            
            # If we consumed all of query, it's a subsequence
            is_subseq = (i == len(q_chars))
            
            # Score based on ratio of query length to name length
            # Higher score if query is closer to full name length
            if is_subseq:
                score = len(query) / len(name)
            else:
                score = 0
                
            scores.append((name, score))
            
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def hybrid_match(self, query: str, top_k: int = 5, 
                    char_weight: float = 0.4,
                    seq_weight: float = 0.4,
                    subseq_weight: float = 0.2) -> List[Tuple[str, float]]:
        """
        Combine multiple matching strategies for better accuracy.
        
        Args:
            query: OCR output of a Khmer name
            top_k: Number of top matches to return
            char_weight: Weight given to character matching (0-1)
            seq_weight: Weight given to sequence matching (0-1)
            subseq_weight: Weight given to subsequence matching (0-1)
            
        Returns:
            List of (name, combined_score) tuples
        """
        # Ensure weights sum to 1
        total_weight = char_weight + seq_weight + subseq_weight
        char_weight = char_weight / total_weight
        seq_weight = seq_weight / total_weight
        subseq_weight = subseq_weight / total_weight
        
        # Get results from each method
        char_results = self.character_match(query, top_k=top_k*2)
        seq_results = self.sequence_match(query, top_k=top_k*2)
        subseq_results = self.subseq_match(query, top_k=top_k*2)
        
        # Combine results
        result_dict = {}
        
        for name, score in char_results:
            result_dict[name] = score * char_weight
            
        for name, score in seq_results:
            if name in result_dict:
                result_dict[name] += score * seq_weight
            else:
                result_dict[name] = score * seq_weight
                
        for name, score in subseq_results:
            if name in result_dict:
                result_dict[name] += score * subseq_weight
            else:
                result_dict[name] = score * subseq_weight
        
        # Sort by combined score
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
    parser = argparse.ArgumentParser(description='Simple Khmer Name OCR Matcher')
    parser.add_argument('--names_file', type=str, default='khmer_names.txt', 
                        help='Path to file containing Khmer names')
    parser.add_argument('--query', type=str, help='Direct OCR text query')
    parser.add_argument('--query_file', type=str, help='File containing OCR text')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top matches to return')
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--weights', type=str, default='0.4,0.4,0.2', 
                        help='Weights for character,sequence,subsequence matching (comma separated)')
    
    args = parser.parse_args()
    
    # Check if we have either query or query file
    if not args.query and not args.query_file:
        print("Error: Either --query or --query_file must be provided")
        sys.exit(1)
    
    # Parse weights
    try:
        char_w, seq_w, subseq_w = map(float, args.weights.split(','))
    except:
        print("Error: Weights must be three comma-separated numbers")
        sys.exit(1)
    
    # Initialize the matcher
    print(f"Initializing Simple Khmer Name Matcher with {args.names_file}...")
    matcher = SimpleKhmerNameMatcher(args.names_file)
    print(f"Loaded {len(matcher.names)} Khmer names")
    
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