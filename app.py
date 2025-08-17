import os
from fastapi import FastAPI
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="IPC PDF QA with Groq LLM")

# ===============================
# Groq API setup
# ===============================
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",  # 🔑 Replace with your key
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

SAVE_PATH = "pdf_faiss_index"
PDF_PATH = "IPC.pdf"

# ===============================
# Helper Functions
# ===============================
def get_pdf_text(pdf_path: str):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks, save_path=SAVE_PATH):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(save_path)
    return vector_store

def load_vector_store(save_path=SAVE_PATH):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# ===============================
# Pydantic Models
# ===============================
class QuestionRequest(BaseModel):
    question: str

# ===============================
# Load or create FAISS index at startup
# ===============================
if os.path.exists(SAVE_PATH):
    vector_store = load_vector_store(SAVE_PATH)
    print("📂 Loaded existing FAISS index")
else:
    raw_text = get_pdf_text(PDF_PATH)
    chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(chunks)
    print("✅ FAISS index created from IPC.pdf")

conversation = get_conversational_chain(vector_store)

# ===============================
# FastAPI Route
# ===============================
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the IPC.pdf.
    """
    response = conversation({"question": request.question})
    chat_history = response['chat_history']
    last_reply = chat_history[-1].content if chat_history else "No response"
    return {"answer": last_reply}
