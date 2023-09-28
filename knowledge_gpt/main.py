import streamlit as st
import openai

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


def synthesize_answer(text, api_key):
    try:
        # Making an API call to OpenAI's GPT-3 with a prompt to summarize the text
        response = openai.Completion.create(
            engine="text-davinci-003",  # or "text-davinci-003" for GPT-3.5
            prompt=f"Summarize the following document responses:\n\n{text}",
            max_tokens=150,  # You might want to adjust this value
            api_key=api_key
        )
        
        # Assuming the response contains the answer in 'choices' field
        answer = response['choices'][0]['text'].strip()
        return answer
    except Exception as e:
        return str(e)  # Return the error message in case of an exception

def process_document(document):
    # Extract text content from the document
    text_content = document.docs[0].page_content  # Adjusted to access the text content of the document

    # Process the text content to generate a synthesized answer
    synthesized_answer = synthesize_answer(text_content, openai_api_key)  # Adjusted to use the synthesize_answer function

    # Extract source information from the document
    source_info = document.docs[0].metadata["source"]  # Adjusted to access the source information of the document

    # Return the synthesized answer and source information
    return synthesized_answer, source_info

def process_all_documents(documents):
    all_sources = []
    all_synthesized_answers = []
    for document in documents:
        synthesized_answer, source_info = process_document(document)
        all_sources.append(source_info)
        all_synthesized_answers.append(synthesized_answer)
    
    # Create a layout with two columns
    col1, col2 = st.columns(2)  # Changed from st.beta_columns to st.columns
    
    # Display all the sources in column 2
    with col2:
        st.write("Sources:")
        for source_info in all_sources:
            st.write(source_info)
    
    # Display all the synthesized answers in column 1
    with col1:
        st.write("Synthesized Answers:")
        for synthesized_answer in all_synthesized_answers:
            st.write(synthesized_answer)


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

if uploaded_files:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key to proceed.")
        st.stop()

    folder_indices = []

    processed_files = []  # List to store processed files

all_documents_text = []  # List to store text of all documents

def query_all_documents(concatenated_documents, query, llm):
    # This is a simplified example. In practice, you might need a more
    # sophisticated method to handle various document structures and formats.
    
    # Concatenated document text can be very long, and it may be beneficial
    # to divide it into smaller chunks, perform the query on each chunk,
    # and then aggregate the results.
    
    # Here we just assume that the concatenated text can be processed in one go.
    result = llm.query(concatenated_documents, query)
    
    # Extract relevant information from the result
    # This is a simplification and your actual extraction process may be more complex
    answer = result.get('answer', 'No answer found')
    
    # Assume each document is separated by a special separator in the concatenated text
    # and extract the source document(s) for the answer
    document_separator = '--- DOCUMENT SEPARATOR ---'
    documents = concatenated_documents.split(document_separator)
    sources = []  # List to store source documents
    
    for document in documents:
        if answer in document:
            sources.append(document.docs[0].page_content)
    
    return {
        'answer': answer,
        'sources': sources
    }

# Process uploaded files
for uploaded_file in uploaded_files:
    try:
        file = read_file(uploaded_file)
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)
        continue  # Skip to the next file on error

    if not is_file_valid(file):
        continue  # Skip to the next file if it's not valid
    all_documents_text.append(file.docs[0].page_content)  # Accessing the text content of the document
    chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)
    processed_files.append(chunked_file)  # Store processed files for later access

    with st.spinner("Indexing document... This may take a while⏳"):
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

# Join all document texts into a single string
all_documents_concatenated = ' '.join(all_documents_text)

if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0.7)

    if selected_document == "All documents":
        process_all_documents(processed_files)  # Call process_all_documents when "All documents" is selected
    else:
        answers = {}  # Dictionary to store answers per document

        # Adjusted index due to "All documents" option
        folder_index = folder_indices[document_options.index(selected_document) - 1]

        # Query the selected document
        result = query_folder(
            folder_index=folder_index,
            query=query,
            return_all=return_all_chunks,
            llm=llm,
        )

        with answer_col:
            st.markdown("#### Answer")
            st.markdown(result.answer)  # assuming result.answer is a string containing the answer

        with sources_col:
            st.markdown("#### Sources")
            for source in result.sources:  # assuming result.sources is a list of source documents
                st.markdown(source.page_content)  # assuming source.page_content is a string representing the document content
                st.markdown(source.metadata["source"])  # assuming source.metadata["source"] provides source document information
                st.markdown("---")  # Separate sources with a line

    # Set queried to True after processing a query
    st.session_state['queried'] = True
