import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import os

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

# Function to ensure the OpenAI client is initialized
def ensure_openai_client():
    if 'openai_client' not in st.session_state:
        api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to create the ChromaDB collection
def create_lab4_collection():
    if 'Lab4_vectorDB' not in st.session_state:
        client = chromadb.Client()
        collection = client.create_collection("Lab4Collection")
        
        ensure_openai_client()
        
        # Updated PDF directory path
        pdf_dir = os.path.join(os.getcwd(), "Lab-04-DataFiles")
        st.write(f"Searching for PDFs in: {pdf_dir}")
        
        if not os.path.exists(pdf_dir):
            st.error(f"Directory not found: {pdf_dir}")
            return None

        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_dir, filename)
                try:
                    with open(filepath, "rb") as file:
                        pdf_reader = PdfReader(file)
                        text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])
                    
                    # Generate the embeddings using OpenAI
                    response = st.session_state.openai_client.embeddings.create(
                        input=text, model="text-embedding-3-small"
                    )
                    embedding = response.data[0].embedding

                    # Add the document to ChromaDB
                    collection.add(
                        documents=[text], 
                        metadatas=[{"filename": filename}], 
                        ids=[filename], 
                        embeddings=[embedding]
                    )
                    st.write(f"Processed: {filename}")
                except Exception as e:
                    st.error(f"Error processing {filename}: {str(e)}")
        
        # Store the collection in session state
        st.session_state.Lab4_vectorDB = collection

    return st.session_state.Lab4_vectorDB

# Function to test querying the vector database
def test_vector_db(collection, query):
    ensure_openai_client()
    try:
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        return [result['filename'] for result in results['metadatas'][0]]
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return []

# Page content for Lab 4
st.title("Lab 4 - ChromaDB Document Search")

# Create the ChromaDB collection
collection = create_lab4_collection()

if collection:
    # Topic selection in the sidebar
    topic = st.sidebar.selectbox("Choose a topic to search", ["Generative AI", "Text Mining", "Data Science Overview"])

    # Button to trigger the search
    if st.sidebar.button('Search'):
        results = test_vector_db(collection, topic)
        st.subheader("Top 3 relevant documents:")
        
        # Display the results
        for i, doc in enumerate(results, 1):
            st.write(f"{i}. {doc}")
        
        # Simple validation check
        st.write("---")
        st.write("Validation:")
        if results:
            st.write("Results seem to be returned successfully. Please manually verify their relevance to the chosen topic.")
        else:
            st.write("No results returned. There might be an issue with the search or the database.")

    # Add some instructions for the user
    st.sidebar.markdown("""
    ## Instructions
    1. Select a topic from the dropdown menu.
    2. Click the 'Search' button to find relevant documents.
    3. The top 3 most relevant document names will be displayed.
    4. Verify that the returned documents seem relevant to the chosen topic.
    """)
else:
    st.error("Failed to create or load the document collection. Please check the file path and try again.")