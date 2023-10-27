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
    "[https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)",
    type='password'  # this line masks the API key input
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

if uploaded_files:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key to proceed.")
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

    # with st.progress(0):
        folder_index = embed_files(
            files=[chunked_file],
            embedding=EMBEDDING if model != "debug" else "debug",
            vector_store=VECTOR_STORE if model != "debug" else "debug",
            openai_api_key=openai_api_key,
        )
        folder_indices.append(folder_index)  # Store folder indices for later querying

st.session_state['processed'] = True  # Set processed to True once documents are processed

if show_full_doc:
    with st.expander("Document"):
        # For simplicity, this code assumes you want to display the last processed file.
        # You might want to adjust this to show all/selected documents.
        last_processed_file = processed_files[-1]  # Get the last processed file
        st.markdown(f"<p>{wrap_doc_in_html(last_processed_file.docs)}</p>", unsafe_allow_html=True)

with st.form(key="qa_form1"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

# Create a list of document options, adding an "All documents" option at the start
document_options = ["All documents"] + [f"Document {i}" for i, _ in enumerate(uploaded_files, start=1)]
selected_document = st.selectbox("Select document", options=document_options)

if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Column (single column for both answers and sources)
    output_col = st.empty()

    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)

    if selected_document == "All documents":
        # Query all documents
        for uploaded_file, folder_index in zip(uploaded_files, folder_indices):
            result = query_folder(
                folder_index=folder_index,
                query=query,
                return_all=return_all_chunks,
                llm=llm,
            )
            with output_col:
                st.markdown(f"#### Answer for {uploaded_file.name}")
                st.markdown(result.answer)
                
                with st.expander(f"Sources for {uploaded_file.name}"):
                    for source in result.sources:
                        st.markdown(source.page_content)
                        st.markdown(source.metadata["source"])
                        st.markdown("---")
    else:
        # Query the selected document
        doc_index = document_options.index(selected_document) - 1
        folder_index = folder_indices[doc_index]
        uploaded_file = uploaded_files[doc_index]
        result = query_folder(
            folder_index=folder_index,
            query=query,
            return_all=return_all_chunks,
            llm=llm,
        )
        with output_col:
            st.markdown(f"#### Answer for {uploaded_file.name}")
            st.markdown(result.answer)

            with st.expander(f"Sources for {uploaded_file.name}"):
                for source in result.sources:
                    st.markdown(source.page_content)
                    st.markdown(source.metadata["source"])
                    st.markdown("---")

    # Set queried to True after processing a query
    st.session_state['queried'] = True

