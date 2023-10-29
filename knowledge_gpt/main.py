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

import tiktoken

def count_tokens(text):
    return len(list(tiktoken.tokenize(text)))

# Initialize session state if it doesn't exist
if 'processed' not in st.session_state:
    st.session_state['processed'] = False

if 'queried' not in st.session_state:
    st.session_state['queried'] = False

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

# Mapping between the models in MODEL_LIST and OpenAI model names
OPENAI_MODEL_MAPPING = {
    "gpt-4": "gpt-4",  # Adjust as needed
    "gpt-3.5-turbo": "gpt-3.5-turbo",  # Adjust as needed
}

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

if 'uploaded_document_count' not in st.session_state:
    st.session_state['uploaded_document_count'] = 0

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

if not uploaded_files:
    st.stop()

if uploaded_files:
    if len(uploaded_files) != st.session_state['uploaded_document_count']:
        # Clear responses and sources if new documents are uploaded
        st.session_state['responses_and_sources'] = []
    st.session_state['uploaded_document_count'] = len(uploaded_files)
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
        if processed_files:
            # Create a list of document options for the user to choose from
            document_options = [f"Document {i + 1}: {file.name}" for i, file in enumerate(uploaded_files)]
            selected_document_to_view = st.selectbox("Select a document to view", options=document_options, index=len(document_options) - 1)

            # Find the index of the selected document
            selected_index = document_options.index(selected_document_to_view)
            
            # Get the processed content of the selected document
            selected_processed_file = processed_files[selected_index]
            
            # Display the content of the selected document
            st.markdown(f"<p>{wrap_doc_in_html(selected_processed_file.docs)}</p>", unsafe_allow_html=True)
        else:
            st.warning("No processed documents are available to display.")


with st.form(key="qa_form1"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

# Create a list of document options, adding an "All documents" option at the start
document_options = ["All documents"] + [f"Document {i}" for i, _ in enumerate(uploaded_files, start=1)]
selected_document = st.selectbox("Select document", options=document_options)

# ...

if submit:
    if not is_query_valid(query):
        st.stop()

    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)

    if 'responses_and_sources' not in st.session_state:
        st.session_state['responses_and_sources'] = []

    if selected_document == "All documents":
        # Query all documents
        for uploaded_file, folder_index in zip(uploaded_files, folder_indices):
            result = query_folder(
                folder_index=folder_index,
                query=query,
                return_all=return_all_chunks,
                llm=llm,
            )
            response_and_sources = {
                'answer': result.answer,
                'sources': [{'content': source.page_content, 'metadata': source.metadata["source"]} for source in result.sources]
            }
            st.session_state['responses_and_sources'].append(response_and_sources)

            # Display responses and sources
            st.markdown(f"#### Answer for {uploaded_file.name}")
            st.markdown(result.answer)
            st.markdown("#### Sources:")
            for source in response_and_sources['sources']:
                st.markdown(source['content'])
                st.markdown(source['metadata'])
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
        response_and_sources = {
            'answer': result.answer,
            'sources': [{'content': source.page_content, 'metadata': source.metadata["source"]} for source in result.sources]
        }
        st.session_state['responses_and_sources'].append(response_and_sources)

        # Display responses and sources
        st.markdown(f"#### Answer for {uploaded_file.name}")
        st.markdown(result.answer)
        st.markdown("#### Sources:")
        for source in response_and_sources['sources']:
            st.markdown(source['content'])
            st.markdown(source['metadata'])
            st.markdown("---")

    # Set queried to True after processing a query
    st.session_state['queried'] = True

def synthesize_insights(text, api_key, openai_model):
    openai.api_key = api_key
    
    prompt = f"Provide a summary of the key themes and insights from this:\n{text}"
    print("Model:", openai_model)
    print("Prompt:", prompt)
    print("Token count:", count_tokens(prompt))
    
    try:
        response = openai.Completion.create(
            model=openai_model,
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except openai.error.InvalidRequestError as e:
        st.error("Invalid request. Please check your input and try again.")
        print("OpenAI API Error:", str(e))
        return ""
    except openai.error.OpenAIError as e:
        st.error("An error occurred while communicating with OpenAI's servers. Please try again later.")
        print("OpenAI API Error:", str(e))
        return ""


if st.session_state.get('responses_and_sources'):
    if st.button("Synthesize All Documents"):
        all_responses_and_sources = "\n".join(
            f"{item['answer']}\n{' '.join(source['content'] for source in item['sources'])}"
            for item in st.session_state['responses_and_sources']
        )
        
        # Get the OpenAI model name based on the user's selection
        openai_model = OPENAI_MODEL_MAPPING.get(model)
        if openai_model is None:
            st.error(f"Model {model} is not supported.")
        else:
            summary = synthesize_insights(all_responses_and_sources, openai_api_key, openai_model)
            st.markdown("### Synthesized Insights")
            st.markdown(summary)


