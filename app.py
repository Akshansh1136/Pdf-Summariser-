import os, tempfile
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# Streamlit app
st.title('PDF Summarizer')

st.image('image.png', width=500)

st.subheader('Streamlit web application for summarizing pdf documents using LangChain and Chroma.')

st.write("""
Welcome to the LangChain Document Summarizer! This web application is a powerful tool designed to leverage OpenAI models, along with LangChain and Chroma, to provide you with concise and informative summaries of your PDF documents.

Here's what happens behind the scenes:
1. **Upload Your Document**: Select a PDF file that you'd like to have summarized.
2. **Processing and Extraction**: The document is split into pages and the text content is extracted.
3. **Creating Embeddings**: We create embeddings for the text using OpenAI models, a cutting-edge language processing technology.
4. **Storing and Searching**: The embeddings are stored in the Chroma database, an open-source, lightweight vector database that allows for efficient similarity search.
5. **Summarization**: LangChain, a framework designed specifically to work with large language models like those from OpenAI, is used to run a summarization chain on the text.
6. **Receive Your Summary**: The final summary is then presented to you right on the web page, all within seconds.

By combining the advanced capabilities of OpenAI's language models with the innovative frameworks of LangChain and Chroma, this application offers a seamless experience for summarizing pdf documents. Whether for professional use, academic research, or personal curiosity, the PDF Summarizer is here to simplify and expedite your text processing needs.

Feel free to upload your document and explore the power of AI-driven summarization. Your concise summary is just a click away!
""")

# Get OpenAI API key and source document input
openai_api_key = st.text_input("OpenAI API Key", type="password")
source_doc = st.file_uploader("Upload Source Document", type="pdf")

# Check if the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not source_doc:
        st.write(f"Please provide the missing fields.")
    else:
        try:
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            os.remove(tmp_file.name)
            
            # Create embeddings for the pages and insert into Chroma database
            embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = Chroma.from_documents(pages, embeddings)

            # Initialize the OpenAI module, load and run the summarize chain
            llm=OpenAI(temperature=0, openai_api_key=openai_api_key)
            chain = load_summarize_chain(llm, chain_type="stuff")
            search = vectordb.similarity_search(" ")
            summary = chain.run(input_documents=search, question="Write a summary and explain about each part in breif.")
            
            st.write(summary)
        except Exception as e:
            st.write(f"An error occurred: {e}")