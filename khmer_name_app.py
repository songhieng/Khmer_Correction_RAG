import streamlit as st
import json
import tempfile
import os
import sys
import pandas as pd
from typing import List, Dict, Union
import traceback

# Import our matchers - try both versions and use what's available
try:
    from khmer_name_simple_matcher import SimpleKhmerNameMatcher
    HAS_SIMPLE_MATCHER = True
except ImportError:
    HAS_SIMPLE_MATCHER = False
    st.error("Simple matcher not available. Please check your installation.")

# Try to import the advanced matcher
try:
    # Set environment variable to use pure Python implementation for protobuf
    # This helps avoid issues with protobuf versions
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    from khmer_ocr_matcher import KhmerNameMatcher
    HAS_ADVANCED_MATCHER = True
except Exception as e:
    HAS_ADVANCED_MATCHER = False
    error_message = str(e)
    error_traceback = traceback.format_exc()
    
    # This will be displayed later in the UI

# Set page title and layout
st.set_page_config(
    page_title="Khmer Name OCR Matcher",
    page_icon="🇰🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
.big-font {
    font-size:24px !important;
    font-weight: bold;
}
.khmer-text {
    font-family: 'Khmer OS', 'Hanuman', 'Moul', sans-serif;
    font-size: 18px;
}
.match-box {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin: 5px 0;
    background-color: #f8f9fa;
}
.confidence-high {
    color: #0f5132;
    background-color: #d1e7dd;
}
.confidence-medium {
    color: #664d03;
    background-color: #fff3cd;
}
.confidence-low {
    color: #842029;
    background-color: #f8d7da;
}
.error-box {
    border: 1px solid #dc3545;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f8d7da;
}
</style>
""", unsafe_allow_html=True)

def load_names(file_path: str) -> List[str]:
    """Load Khmer names from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f.readlines()]
        return [name for name in names if name]
    except Exception as e:
        st.error(f"Error loading names file: {e}")
        return []

def format_match_result(result: Dict, index: int) -> str:
    """Format a match result with color coding based on confidence."""
    score = result["score"]
    name = result["name"]
    
    # Determine confidence level color
    if score > 0.8:
        confidence_class = "confidence-high"
    elif score > 0.5:
        confidence_class = "confidence-medium"
    else:
        confidence_class = "confidence-low"
    
    # Return formatted HTML
    return f"""
    <div class="match-box {confidence_class}">
        <span class="khmer-text">{index}. {name}</span>
        <span style="float:right;">Score: {score:.4f}</span>
    </div>
    """

def process_matches(matcher, query_text: str, top_k: int, weights: List[float]) -> List[Dict]:
    """Process OCR text and get matches."""
    if HAS_ADVANCED_MATCHER and isinstance(matcher, KhmerNameMatcher):
        semantic_w, char_w, seq_w = weights
        results = matcher.process_ocr_output(query_text, top_k=top_k)
    elif HAS_SIMPLE_MATCHER and isinstance(matcher, SimpleKhmerNameMatcher):
        char_w, seq_w, subseq_w = weights
        results = matcher.process_ocr_output(query_text, top_k=top_k)
    else:
        results = []
    
    return results

def main():
    # Title
    st.markdown('<p class="big-font">Khmer Name OCR Matcher</p>', unsafe_allow_html=True)
    st.markdown("Match noisy OCR outputs of Khmer names to a database of known names.")
    
    # Display errors about advanced matcher if applicable
    if not HAS_ADVANCED_MATCHER and 'error_message' in locals():
        with st.expander("Advanced matcher not available due to errors"):
            st.markdown(f"<div class='error-box'>{error_message}</div>", unsafe_allow_html=True)
            st.code(error_traceback, language="python")
            st.info("Using simple matcher instead. To use the advanced matcher, try installing the required dependencies:\n```\npip install protobuf==3.20.3 sentence-transformers faiss-cpu torch numpy\n```")
    
    # Sidebar - Configuration
    st.sidebar.header("Configuration")
    
    # Names file selection
    names_file = st.sidebar.text_input("Names File Path", "khmer_names.txt")
    
    # Check if names file exists
    if not os.path.exists(names_file):
        st.sidebar.error(f"File not found: {names_file}")
        # Provide option to use included file
        if st.sidebar.button("Use Included Sample File"):
            names_file = "khmer_names.txt"
            if os.path.exists(names_file):
                st.sidebar.success(f"Using included file: {names_file}")
            else:
                st.sidebar.error("Sample file not found either!")
                return
    else:
        st.sidebar.success(f"Found names file: {names_file}")
        num_names = len(load_names(names_file))
        st.sidebar.info(f"Contains {num_names} names")
    
    # Matcher selection - only show options that are available
    matcher_options = []
    if HAS_SIMPLE_MATCHER:
        matcher_options.append("Simple (No Neural Network)")
    if HAS_ADVANCED_MATCHER:
        matcher_options.append("Advanced (Neural Network)")
    
    # Default to simple if advanced is not available
    default_index = 0 if "Simple (No Neural Network)" in matcher_options else 0
    
    if matcher_options:
        matcher_type = st.sidebar.radio("Matcher Type", matcher_options, index=default_index)
    else:
        st.error("No matcher implementations available. Please check your installation.")
        return
    
    # Top K selection
    top_k = st.sidebar.slider("Number of Top Matches", min_value=1, max_value=10, value=3)
    
    # Weights configuration based on matcher type
    if matcher_type == "Advanced (Neural Network)":
        if not HAS_ADVANCED_MATCHER:
            st.sidebar.error("Advanced matcher not available. Missing dependencies.")
            st.sidebar.info("Try: pip install protobuf==3.20.3 sentence-transformers faiss-cpu torch numpy")
            return
        
        st.sidebar.subheader("Matching Weights")
        semantic_weight = st.sidebar.slider("Semantic Weight", 0.0, 1.0, 0.5, 0.1)
        char_weight = st.sidebar.slider("Character Weight", 0.0, 1.0, 0.3, 0.1)
        seq_weight = st.sidebar.slider("Sequence Weight", 0.0, 1.0, 0.2, 0.1)
        weights = [semantic_weight, char_weight, seq_weight]
        
        # Create matcher instance
        try:
            with st.spinner("Loading neural model..."):
                matcher = KhmerNameMatcher(names_file)
            st.sidebar.success("Neural model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading neural model: {e}")
            st.sidebar.info("Falling back to simple matcher...")
            matcher_type = "Simple (No Neural Network)"
    
    if matcher_type == "Simple (No Neural Network)":
        if not HAS_SIMPLE_MATCHER:
            st.sidebar.error("Simple matcher not available.")
            return
        
        st.sidebar.subheader("Matching Weights")
        char_weight = st.sidebar.slider("Character Weight", 0.0, 1.0, 0.4, 0.1)
        seq_weight = st.sidebar.slider("Sequence Weight", 0.0, 1.0, 0.4, 0.1)
        subseq_weight = st.sidebar.slider("Subsequence Weight", 0.0, 1.0, 0.2, 0.1)
        weights = [char_weight, seq_weight, subseq_weight]
        
        # Create matcher instance
        try:
            matcher = SimpleKhmerNameMatcher(names_file)
            st.sidebar.success("Simple matcher initialized!")
        except Exception as e:
            st.sidebar.error(f"Error initializing matcher: {e}")
            return
    
    # Main area - Input and Results
    st.subheader("Input")
    
    # Tabs for different input methods
    input_tab, file_tab, sample_tab = st.tabs(["Direct Input", "File Upload", "Sample Input"])
    
    with input_tab:
        query_text = st.text_area("Enter Khmer OCR Text", height=150, 
                               placeholder="Enter Khmer text (each name on a new line)")
        
        if st.button("Match Names", key="match_direct", type="primary"):
            if not query_text:
                st.warning("Please enter some text to match.")
            else:
                with st.spinner("Processing..."):
                    results = process_matches(matcher, query_text, top_k, weights)
                display_results(results, top_k)
    
    with file_tab:
        uploaded_file = st.file_uploader("Upload OCR output file", type=["txt"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                # Read the file content
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Show file content
                st.text_area("File Content", file_content, height=150, disabled=True)
                
                if st.button("Match Names", key="match_file", type="primary"):
                    with st.spinner("Processing..."):
                        results = process_matches(matcher, file_content, top_k, weights)
                    display_results(results, top_k)
            except Exception as e:
                st.error(f"Error reading file: {e}")
            finally:
                # Clean up the temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    with sample_tab:
        # Sample inputs for testing
        sample_inputs = [
            "លឹមហ៊ន",
            "ឡេងសុផល",
            "សុធ",
            "វុឌ្ឍាស",
            "ចន្ទដា",
            "ឡេងហង"
        ]
        
        # Display sample inputs
        st.markdown('<div class="khmer-text">' + '<br>'.join(sample_inputs) + '</div>', unsafe_allow_html=True)
        
        if st.button("Match Sample Names", key="match_sample", type="primary"):
            sample_text = '\n'.join(sample_inputs)
            with st.spinner("Processing..."):
                results = process_matches(matcher, sample_text, top_k, weights)
            display_results(results, top_k)

def display_results(results: List[Dict], top_k: int):
    """Display matching results in a nice format."""
    if not results:
        st.warning("No matches found.")
        return
    
    st.subheader("Results")
    
    # Create columns for results
    cols = st.columns(min(3, len(results)))
    
    # Display each result in a column
    for i, result in enumerate(results):
        col_idx = i % len(cols)
        with cols[col_idx]:
            ocr_text = result["ocr_text"]
            matches = result["matches"]
            
            # Display OCR text
            st.markdown(f'<p class="khmer-text">OCR Text: <b>{ocr_text}</b></p>', unsafe_allow_html=True)
            
            # Display matches
            for j, match in enumerate(matches, 1):
                st.markdown(format_match_result(match, j), unsafe_allow_html=True)
    
    # Option to download results as JSON
    if results:
        json_results = json.dumps(results, ensure_ascii=False, indent=2)
        st.download_button(
            label="Download Results (JSON)",
            data=json_results,
            file_name="khmer_name_matches.json",
            mime="application/json"
        )
        
        # Also create a CSV for easier viewing
        rows = []
        for result in results:
            ocr_text = result["ocr_text"]
            for match in result["matches"]:
                rows.append({
                    "OCR Text": ocr_text,
                    "Matched Name": match["name"],
                    "Score": match["score"]
                })
        
        if rows:
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name="khmer_name_matches.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 