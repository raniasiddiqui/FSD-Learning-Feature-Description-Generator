from xml.parsers.expat import model
import streamlit as st
import json
import fitz  # PyMuPDF
from pathlib import Path
import tempfile
import os
from groq import Groq


# Page configuration
st.set_page_config(
    page_title="Feature Description Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# def load_config(config_path: str = "config.json") -> dict:
#     """Loads configuration values from the config.json file."""
#     try:
#         with open(config_path, "r", encoding="utf-8") as f:
#             config = json.load(f)
#         return config
#     except FileNotFoundError:
#         st.error(f"❌ Config file not found at {config_path}. Please create it first.")
#         return {}
#     except json.JSONDecodeError as e:
#         st.error(f"❌ Error parsing config file: {e}")
#         return {}

api_key = st.secrets.get("groq_api_key")
model_name = st.secrets.get("groq_default_model", "llama-3.3-70b-versatile")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def extract_text_from_pdf(pdf_file) -> str:
    """Extracts all text from an uploaded PDF file."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text
        doc = fitz.open(tmp_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return full_text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""


def generate_feature_descriptions(document_text: str, feature_list: list, api_key: str, model_name: str) -> dict:
    """Uses Google's Generative AI to create feature descriptions."""
    
    prompt = f"""
    You are a professional QA analyst. Your task is to create detailed descriptions of software features based on a Functional Specification Document (FSD). These descriptions will be used to write comprehensive test cases.

    **Instructions:**
    1. Read the entire `DOCUMENT_TEXT` provided below.
    2. For each feature name in the `FEATURE_LIST`, locate the corresponding section in the `DOCUMENT_TEXT`.
    3. Synthesize a detailed, paragraph-style description for each feature. This description must include:
        - The primary goal or purpose of the feature.
        - The event/trigger for each feature. 
        - The main user/actor who interacts with it.
        - The basic workflow or sequence of actions the user performs.
        - The alternate flow or sequence of actions the user performs.
        - The pre-condition 
        - The post condition
        - Validations/rules mentioned.
        - The description must be highly verbose and exahustive. Include every single bullet point, sub-point, field list and rule from the relevant section.
        - Do not summarize or condense any part of the document. Capture 100% of the details without omission.
        - All sections, elements and fields should be included from the feature.
    4. IMPORTANT: Use ONLY the information provided in the `DOCUMENT_TEXT`. Do not add any information or make assumptions.
    5. Format your final output as a single JSON object, where the keys are the exact feature names from the `FEATURE_LIST` and the values are the detailed string descriptions.
    6. Before generating the feature description, please add the following paragraph at the start of each description:
        Generate the most comprehensive and detailed set of test cases for the following feature description. Each line in each workflow must be tested through multiple functional, alternate, negative, boundary, UI, API, and access-control scenarios to ensure complete coverage.
        Below is the feature description:

    **FEATURE_LIST:**
    {json.dumps(feature_list, indent=2)}

    **DOCUMENT_TEXT:**
    ---
    {document_text}
    ---

    **OUTPUT (JSON Object Only):**
    Respond with ONLY a valid JSON object. No explanations, no markdown, no extra text.
    Example format:
    {{
      "Login Functionality": "Generate the most comprehensive... Below is the feature description: The user can log in using username and password...",
      "Dashboard View": "..."
    }}
   
    """
    
    try:
        client = Groq(api_key=api_key)

        with st.spinner("🤖 AI is analyzing your document and generating descriptions..."):
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert QA analyst. Return ONLY valid JSON. No markdown, no extra text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096
            )

        content = response.choices[0].message.content.strip()

        # Method 1: Try to extract JSON block if wrapped in ```
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = content

        try:
            feature_descriptions = json.loads(json_str)
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing failed: {e}")
            st.code(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)
            return {}

        return feature_descriptions

    except Exception as e:
        st.error(f"Error communicating with Groq model: {e}")
        return {}


# Load configuration at startup
# Initialize session state
if 'descriptions' not in st.session_state:
    st.session_state.descriptions = None
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None

# Load secrets from Streamlit Cloud
api_key = st.secrets.get("groq_api_key", None)
model_name = st.secrets.get("groq_default_model", "llama-3.3-70b-versatile")

# Main UI
st.markdown('<p class="main-header">📄 Feature Description Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automatically generate detailed feature descriptions from your FSD documents using AI</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Show API key status
    st.subheader("🔑 API Settings")
    if api_key:
        st.success("✅ Groq API Key loaded from Streamlit Secrets")
    else:
        st.error("❌ Groq API Key not found in secrets.toml")
        st.info("Add it in Streamlit → Settings → Secrets")

    # Model name from secrets
    if "model_name" in st.secrets:
        st.info(f"📦 Model Loaded: {model_name}")
    else:
        st.warning("⚠️ Model not found in secrets. Using default: llama-3.3-70b-versatile")

    st.divider()
    
    st.subheader("📋 Features to Analyze")
    feature_input_method = st.radio(
        "Input Method",
        ["Manual Entry", "Upload JSON"]
    )
    
    features_list = []
    
    if feature_input_method == "Manual Entry":
        feature_text = st.text_area(
            "Enter features (one per line)",
            height=150,
            placeholder="Feature 1\nFeature 2\nFeature 3"
        )
        if feature_text:
            features_list = [f.strip() for f in feature_text.split('\n') if f.strip()]
    else:
        uploaded_json = st.file_uploader("Upload features JSON", type=['json'])
        if uploaded_json:
            try:
                features_data = json.load(uploaded_json)
                if isinstance(features_data, list):
                    features_list = features_data
                elif isinstance(features_data, dict) and 'features_to_describe' in features_data:
                    features_list = features_data['features_to_describe']
                else:
                    st.error("JSON should be a list or contain 'features_to_describe' key")
            except Exception as e:
                st.error(f"Error reading JSON: {e}")
    
    if features_list:
        st.success(f"✅ {len(features_list)} features loaded")

# config = load_config()

# # Initialize session state
# if 'descriptions' not in st.session_state:
#     st.session_state.descriptions = None
# if 'pdf_text' not in st.session_state:
#     st.session_state.pdf_text = None
# if 'config' not in st.session_state:
#     st.session_state.config = config


# # Main UI
# st.markdown('<p class="main-header">📄 Feature Description Generator</p>', unsafe_allow_html=True)
# st.markdown('<p class="sub-header">Automatically generate detailed feature descriptions from your FSD documents using AI</p>', unsafe_allow_html=True)

# # Sidebar for configuration
# with st.sidebar:
#     st.header("⚙️ Configuration")
    
#     # Show API key status
#     st.subheader("🔑 API Settings")
#     if st.session_state.config and 'google_api_key' in st.session_state.config:
#         st.success("✅ API Key loaded from config.json")
#         api_key = st.session_state.config['google_api_key']
#     else:
#         st.error("❌ API Key not found in config.json")
#         st.info("Please add 'google_api_key' to your config.json file")
#         api_key = None
    
#     # Model selection from config
#     if st.session_state.config and 'model_name' in st.session_state.config:
#         model_name = st.session_state.config['model_name']
#         st.info(f"📦 Model: {model_name} (from config.json)")
#     else:
#         st.warning("⚠️ Model not found in config.json, using default")
#         model_name = "gemini-1.5-flash"
    
#     st.divider()
    
#     st.subheader("📋 Features to Analyze")
#     feature_input_method = st.radio(
#         "Input Method",
#         ["Manual Entry", "Upload JSON"]
#     )
    
#     features_list = []
    
#     if feature_input_method == "Manual Entry":
#         feature_text = st.text_area(
#             "Enter features (one per line)",
#             height=150,
#             placeholder="Feature 1\nFeature 2\nFeature 3"
#         )
#         if feature_text:
#             features_list = [f.strip() for f in feature_text.split('\n') if f.strip()]
#     else:
#         uploaded_json = st.file_uploader("Upload features JSON", type=['json'])
#         if uploaded_json:
#             try:
#                 features_data = json.load(uploaded_json)
#                 if isinstance(features_data, list):
#                     features_list = features_data
#                 elif isinstance(features_data, dict) and 'features_to_describe' in features_data:
#                     features_list = features_data['features_to_describe']
#                 else:
#                     st.error("JSON should be a list or contain 'features_to_describe' key")
#             except Exception as e:
#                 st.error(f"Error reading JSON: {e}")
    
#     if features_list:
#         st.success(f"✅ {len(features_list)} features loaded")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Document")
    uploaded_pdf = st.file_uploader(
        "Upload your FSD PDF",
        type=['pdf'],
        help="Upload the Functional Specification Document"
    )
    
    if uploaded_pdf:
        st.success(f"✅ File uploaded: {uploaded_pdf.name}")
        
        if st.button("🔍 Extract Text from PDF"):
            with st.spinner("Extracting text..."):
                pdf_text = extract_text_from_pdf(uploaded_pdf)
                if pdf_text:
                    st.session_state.pdf_text = pdf_text
                    st.success(f"✅ Extracted {len(pdf_text)} characters")
                    
                    with st.expander("📖 Preview Extracted Text"):
                        st.text_area("Document Text", pdf_text[:2000] + "..." if len(pdf_text) > 2000 else pdf_text, height=300)

with col2:
    st.header("🚀 Generate Descriptions")
    
    can_generate = (
        api_key and 
        features_list and 
        st.session_state.pdf_text is not None
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar")
    if not features_list:
        st.warning("⚠️ Please add features to analyze in the sidebar")
    if st.session_state.pdf_text is None:
        st.warning("⚠️ Please upload and extract text from a PDF")
    
    if can_generate:
        st.info(f"Ready to analyze {len(features_list)} features")
        
        if st.button("✨ Generate Feature Descriptions", disabled=not can_generate):
            descriptions = generate_feature_descriptions(
                st.session_state.pdf_text,
                features_list,
                api_key,
                model_name
            )
            
            if descriptions:
                st.session_state.descriptions = descriptions
                st.success("✅ Descriptions generated successfully!")

# Display results
if st.session_state.descriptions:
    st.header("📊 Generated Feature Descriptions")
    
    # Download button
    json_output = json.dumps(st.session_state.descriptions, indent=2)
    st.download_button(
        label="💾 Download as JSON",
        data=json_output,
        file_name="feature_descriptions.json",
        mime="application/json"
    )
    
    st.divider()
    
    # Display each feature
    for idx, (feature, description) in enumerate(st.session_state.descriptions.items(), 1):
        with st.container():
            st.markdown(f"""
                <div class="feature-box">
                    <h3>🎯 Feature {idx}: {feature}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(description)
            
            # Individual download button for each feature
            col1, col2, col3 = st.columns([2, 1, 1])
            with col3:
                feature_json = json.dumps({feature: description}, indent=2)
                st.download_button(
                    label="Download",
                    data=feature_json,
                    file_name=f"{feature.replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"download_{idx}"
                )
            
            st.divider()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit and Google Generative AI</p>",
    unsafe_allow_html=True

)
