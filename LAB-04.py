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
        pdf_dir = "/workspaces/document-qa/IST-688-Docs"  # Update to your PDF directory
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_dir, filename)
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
        
        # Store the collection in session state
        st.session_state.Lab4_vectorDB = collection

    return st.session_state.Lab4_vectorDB

# Function to test querying the vector database
def test_vector_db(collection, query):
    ensure_openai_client()
    response = st.session_state.openai_client.embeddings.create(
        input=query, model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return [result['filename'] for result in results['metadatas'][0]]

# Page content for Lab 4
st.title("Lab 4 - ChromaDB Document Search")

# Create the ChromaDB collection
collection = create_lab4_collection()

# Topic selection in the sidebar
topic = st.sidebar.selectbox("Choose a topic to search", ["Generative AI", "Text Mining", "Data Science Overview"])

# Button to trigger the search
if st.sidebar.button('Search'):
    results = test_vector_db(collection, topic)
    st.subheader("Top 3 relevant documents:")
    
    # Display the results
    for i, doc in enumerate(results, 1):
        st.write(f"{i}. {doc}")
    st.write("---")