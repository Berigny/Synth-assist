import streamlit as st
from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)
from knowledge_gpt.core.caching import bootstrap_caching
from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm

# Initialize session state if it doesn't exist
if 'processed' not in st.session_state:
    st.session_state['processed'] = False

if 'queried' not in st.session_state:
    st.session_state['queried'] = False

# Initialize an empty string to accumulate document text
all_documents_text = ""

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

# Page setup
st.set_page_config(page_title="HCD-Helper", layout="wide")
st.header("HCD-Helper")

# Enable caching for expensive functions
bootstrap_caching()

openai_api_key = st.text_input(
    "Enter your OpenAI API key. You can get a key at "
    "[https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)"
)

uploaded_files = st.file_uploader(
    "Upload pdf, docx, or txt files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Scanned documents are not supported yet!",
)

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

if not uploaded_files:
    st.stop()

folder_indices = []
processed_files = []  # List to store processed files

# Process uploaded files
for uploaded_file in uploaded_files:
    try:
        file = read_file(uploaded_file)
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)
        continue  # Skip to the next file on error

    if not is_file_valid(file):
        continue  # Skip to the next file if it's not valid

    chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)
    processed_files.append(chunked_file)  # Store processed files for later access

    # Accumulate document text
    all_documents_text += '\n'.join([doc.text for doc in chunked_file.docs]) + '\n'

    with st.spinner("Indexing document... This may take a while⏳"):
        folder_index = embed_files(
            files=[chunked_file],
            embedding=EMBEDDING if model != "debug" else "debug",
            vector_store=VECTOR_STORE if model != "debug" else "debug",
            openai_api_key=openai_api_key,
        )
        folder_indices.append(folder_index)  # Store folder indices for later querying

st.session_state['processed'] = True  # Set processed to True once documents are processed

