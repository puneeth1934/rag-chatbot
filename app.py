import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Add it to a .env file or the environment.")
    st.stop()

# Setup
st.title("🧠 RAG Chatbot with Memory (Gemini + FAISS)")

# Upload text file
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
if uploaded_file:
    with open("uploaded_data.txt", "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded!")

    # Load and split data
    loader = TextLoader("uploaded_data.txt")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Chat model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Retrieval QA chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # User input
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask me anything about your document...")
    if user_input:
        response = qa.run(user_input)
        st.session_state.history.append((user_input, response))

    # Display chat history
    for user_q, bot_a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            st.markdown(bot_a)
else:
    st.info("Please upload a .txt file to start the RAG chatbot.")
