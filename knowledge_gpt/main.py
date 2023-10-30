import openai
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

# Mapping between the models in MODEL_LIST and OpenAI model names
OPENAI_MODEL_MAPPING = {
    "gpt-4": "gpt-4",  # Adjust as needed
    "gpt-3.5-turbo": "gpt-3.5-turbo",  # Adjust as needed
}

# Page setup
st.set_page_config(page_title="Synth-Assist", layout="wide")
st.header("Synth-Assist")

# Enable caching for expensive functions
bootstrap_caching()

openai_api_key = st.text_input(
    "Enter your OpenAI API key. You can get a key at "
    "[https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)",
    type='password'  # this line masks the API key input
)

# Place the query type selector outside the form
query_type = st.selectbox(
    "What synthesis do you need?",
    options=["Find main themes and insights", "Find key opportunities and recommendations", "Ask another question"],
    key='selected_query_type'
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

# with st.expander("Advanced Options"):
#     return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
#     show_full_doc = st.checkbox("Show parsed contents of the document")

return_all_chunks = False  # Set default value
show_full_doc = False  # Set default value

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
            selected_document = "All documents"  # Default to "All documents"

            # Find the index of the selected document
            selected_index = document_options.index(selected_document) - 1
            
            # Get the processed content of the selected document
            selected_processed_file = processed_files[selected_index]
            
            # Display the content of the selected document
            st.markdown(f"<p>{wrap_doc_in_html(selected_processed_file.docs)}</p>", unsafe_allow_html=True)
        else:
            st.warning("No processed documents are available to display.")


def handle_form_submission():
    st.session_state.query_type = st.session_state.selected_query_type

with st.form(key="qa_form1"):
    query = ""
    if query_type == "Find main themes and insights":
        query = "provide a detailed analysis of the key insights, patterns, and themes present in the transcript. Identify the associated pain points or unmet needs. Also, identify the associated gain points or met needs. Include specific examples or quotes to support your analysis, and highlight any supporting facts, evidence, or statistics if available. Please ensure the response is in a paragraph, is clear, direct, concise, and well-structured for easy readability, maintaining a formal and analytical tone."
    elif query_type == "Find key opportunities and recommendations":
        query = "list potential opportunities or recommendations that could address issues present in the transcript. Provide a rationale for each opportunity or recommendation, explaining why it is valuable and how it addresses the specific issue. Ensure that your suggestions are practical, feasible, and well-suited to the context of the interview. Please ensure the response is in a paragraph, is clear, direct, concise, and well-structured for easy readability, maintaining a formal, solution-oriented, and persuasive tone throughout your analysis."
    elif query_type == "Ask another question":
        query = st.text_area("Ask a question about the transcript/s")

    submit = st.form_submit_button("Start Synthesis", on_click=handle_form_submission)

# Create a list of document options, adding an "All documents" option at the start
document_options = ["All documents"] + [f"Document {i}" for i, _ in enumerate(uploaded_files, start=1)]
# selected_document = st.selectbox("Select document", options=document_options)
selected_document = "All documents"

# ...

if submit:
    # st.session_state.query_type = query_type  # This line should be removed
    if query_type == "Ask another question" and not is_query_valid(query):
        st.error("Please enter a valid question.")
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

# Initialize 'previous_responses' in session state if it doesn't exist
if 'previous_responses' not in st.session_state:
    st.session_state['previous_responses'] = []

def synthesize_insights(text, api_key, openai_model):
    if not api_key or not isinstance(api_key, str):
        st.error("Invalid API key. Please check your input and try again.")
        return ""

    prompt = f"Provide a summary of the key themes and insights from this:\n{text}"
    
    # Print the prompt to check its content
    print("Prompt:", prompt)

    try:
        if "turbo" in openai_model:
            # If using a chat model, use the chat completions endpoint
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ]
            print("Sending to OpenAI:", messages)  # Add this line to print the request
            response = openai.ChatCompletion.create(
                model=openai_model,
                messages=messages,
                api_key=api_key
            )
            return response['choices'][0]['message']['content'].strip()
        else:
            # If using a non-chat model, use the completions endpoint
            print("Sending to OpenAI:", prompt)  # Add this line to print the request
            response = openai.Completion.create(
                model=openai_model,
                prompt=prompt,
                max_tokens=150,
                api_key=api_key
            )
            return response['choices'][0]['text'].strip()
    except openai.error.InvalidRequestError as e:
        print("OpenAI Error:", str(e))  # Print the error message
        st.error("An error occurred while processing your request. Please check the console for more details.")
        return ""

if st.session_state.get('responses_and_sources'):
    if st.button("Synthesize All Transcripts"):
        all_responses = "\n".join(
            item['answer'] for item in st.session_state['responses_and_sources']
        )
        
        # Constructing the prompt
        prompt = (
            "I have gathered information from various transcripts. Below is a summary of the key points from each document:\n"
            f"{all_responses}\n"
            "Based on the information above, please provide a comprehensive summary highlighting the main themes, insights, and important points. Please ensure the response is in a paragraph, is clear, concise, and well-structured for easy readability, maintaining a formal and analytical tone."
        )
        
        openai_model = OPENAI_MODEL_MAPPING.get(model)
        if openai_model is None:
            st.error(f"Model {model} is not supported.")
        else:
            summary = synthesize_insights(prompt, openai_api_key, openai_model)
            st.markdown("### Synthesized Insights")
            st.markdown(summary)
            st.session_state['previous_responses'].append({'answer': summary, 'sources': []})


