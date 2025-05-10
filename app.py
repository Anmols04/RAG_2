# app.py

import os
import glob
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")

MODEL = "gpt-4o-mini"
db_name = "vector_db"

st.title("🧠 RAG-based Chat Assistant")

# Load and split documents
@st.cache_resource
def load_data():
    folders = glob.glob("knowledge-base/**/*", recursive=True)
    loaders = [TextLoader(path) for path in folders if path.endswith(".txt")]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return split_docs

# Initialize the vector store
@st.cache_resource
def create_vector_store(docs):
    embeddings = OpenAIEmbeddings(model=MODEL)
    db = Chroma.from_documents(docs, embeddings, persist_directory=db_name)
    return db

docs = load_data()
db = create_vector_store(docs)

# Set up the conversational chain
llm = ChatOpenAI(model=MODEL)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=db.as_retriever(), memory=memory
)

# Chat Interface
user_input = st.chat_input("Ask me anything about your documents:")
if user_input:
    result = qa_chain.invoke({"question": user_input})
    st.write(result['answer'])
