import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sentence-transformers and TF-IDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# -----------------------------
# Load API keys
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Simple RAG App with Dense + TF-IDF Hybrid")

pdf_path = st.text_input("Enter PDF path", value="./attention.pdf")
question = st.text_input("Ask a question from the document")

# -----------------------------
# Build Hybrid Retriever (local)
# -----------------------------
def build_retriever(pdf_path):
    # Step 1: Load and split PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    texts = [d.page_content for d in chunks]

    # Step 2: Dense embeddings (sentence-transformers)
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    dense_vectors = dense_model.encode(texts, convert_to_numpy=True)

    # Step 3: Sparse vectors (TF-IDF)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)

    return chunks, dense_model, dense_vectors, tfidf, tfidf_matrix

# -----------------------------
# Button to prepare retriever
# -----------------------------
if st.button("Prepare Index"):
    st.session_state["chunks"], st.session_state["dense_model"], \
    st.session_state["dense_vectors"], st.session_state["tfidf"], \
    st.session_state["tfidf_matrix"] = build_retriever(pdf_path)
    st.success("Hybrid retriever is ready!")

# -----------------------------
# Ask question
# -----------------------------
if question and "chunks" in st.session_state:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

    # Encode query
    query_dense = st.session_state["dense_model"].encode([question], convert_to_numpy=True)
    query_sparse = st.session_state["tfidf"].transform([question])

    # Dense similarity
    from sklearn.metrics.pairwise import cosine_similarity
    dense_scores = cosine_similarity(query_dense, st.session_state["dense_vectors"])[0]

    # Sparse similarity
    sparse_scores = (query_sparse @ st.session_state["tfidf_matrix"].T).toarray()[0]

    # Combine scores (simple sum)
    hybrid_scores = dense_scores + sparse_scores

    # Get top chunks
    top_indices = np.argsort(hybrid_scores)[::-1][:3]
    context = "\n\n".join([st.session_state["chunks"][i].page_content for i in top_indices])

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer:\n<context>\n{context}\n</context>\nQuestion: {input}"
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain( retriver, document_chain)

    response = document_chain.invoke({"input": question, "context": context})
    st.write("Answer:", response)

    with st.expander("Relevant document chunks"):
        for i in top_indices:
            st.write(st.session_state["chunks"][i].page_content)
            st.write("---")
