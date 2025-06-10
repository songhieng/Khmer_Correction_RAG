# Khmer Name OCR Matcher

A retrieval-augmented generation (RAG) system for matching noisy OCR outputs of Khmer names to a database of known Khmer names.

## Features

- Uses semantic embeddings (via SentenceTransformer) to capture meaning and context
- Employs FAISS for fast vector similarity search
- Combines multiple matching strategies:
  - Semantic matching (vector similarity)
  - Character-based matching (character overlap)
  - Sequence matching (edit distance)
  - Subsequence matching
- Command-line interface for easy integration
- User-friendly web interface with Streamlit
- Supports batch processing of OCR outputs
- Handles noisy or partial OCR results
- Fallback to simpler methods for low-memory environments

## Installation

For the full version with neural network models:

```bash
pip install sentence-transformers faiss-cpu torch numpy streamlit pandas
```

For the lightweight version that doesn't require neural models:

```bash
pip install streamlit pandas
```

## Usage

### Web Interface (Recommended)

Run the Streamlit web interface for the most user-friendly experience:

```bash
streamlit run khmer_name_app.py
```

This will open a web browser with an interactive interface where you can:
- Choose between simple and advanced matching
- Upload OCR text files or enter text directly
- Adjust matching parameters
- View color-coded results based on confidence
- Download results in JSON or CSV format

### Command Line Interface

#### Advanced Version (with neural models)

```bash
python khmer_ocr_matcher.py --query "វុឌ្ឍាស"
```

#### Simple Version (no neural models)

```bash
python khmer_name_simple_matcher.py --query "វុឌ្ឍាស"
```

### Process a file containing OCR output

```bash
python khmer_ocr_matcher.py --query_file ocr_output.txt
# or
python khmer_name_simple_matcher.py --query_file ocr_output.txt
```

### Save results to a JSON file

```bash
python khmer_ocr_matcher.py --query "វុឌ្ឍាស" --output results.json
# or
python khmer_name_simple_matcher.py --query "វុឌ្ឍាស" --output results.json
```

### Adjust the number of top matches

```bash
python khmer_ocr_matcher.py --query "វុឌ្ឍាស" --top_k 5
```

### Change the matching weights

For advanced version (semantic, character, sequence):

```bash
python khmer_ocr_matcher.py --query "វុឌ្ឍាស" --weights "0.6,0.3,0.1"
```

For simple version (character, sequence, subsequence):

```bash
python khmer_name_simple_matcher.py --query "វុឌ្ឍាស" --weights "0.4,0.4,0.2"
```

### Disable semantic matching (for low memory)

```bash
python khmer_ocr_matcher.py --query "វុឌ្ឍាស" --no_semantic
```

## How It Works

The system combines retrieval-based methods with generative capabilities:

1. **Retrieval**: The OCR output is used as a query to retrieve the most similar Khmer names from the database using multiple matching strategies.
2. **Fusion**: The results from different matching strategies are combined with weighted scores.
3. **Ranking**: The combined results are ranked to provide the most likely matches.

This approach is effective for handling OCR errors and distortions in Khmer text, substantially improving hit rates over standalone fuzzy matching or pure LLM completion.

### Advanced vs. Simple Versions

- **Advanced Version**: Uses neural network embeddings for semantic matching, which can better understand the context and meaning of names. Requires more computational resources.
- **Simple Version**: Uses character overlap, sequence matching, and subsequence matching without neural networks. More lightweight but may be less accurate for complex cases.

## Sample Results

Input (OCR text): `លឹមហ៊ន`  
Top matches:
1. លឹមហ៊ុន (score: 0.8835)
2. លឹម (score: 0.4667)
3. ហ៊ុន (score: 0.4114)

Input (OCR text): `ឡេងហង`  
Top matches:
1. ឡេងហេង (score: 1.4606)
2. ឡេង (score: 0.6000)
3. ឡេងហួន (score: 0.5576)

## Screenshots

The Streamlit interface provides an intuitive way to interact with the matcher:

- Configure matching parameters
- Enter OCR text directly or upload files
- View results with confidence indicators
- Download results in JSON or CSV format

## License

MIT 