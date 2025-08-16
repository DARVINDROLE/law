import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# ===============================
# FastAPI app
# ===============================
app = FastAPI(title="PDF QA Bot", description="Ask questions about PDF content", version="1.0.0")

# ===============================
# Groq API setup
# ===============================
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",   # <-- replace with your key
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# ===============================
# Helper Functions
# ===============================
def get_pdf_text(pdf_path: str):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embedding=embeddings)


def get_conversational_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

# ===============================
# Load PDF and prepare model
# ===============================
PDF_PATH = "IPC.pdf"   # change path if needed

raw_text = get_pdf_text(PDF_PATH)
text_chunks = get_text_chunks(raw_text)
vector_store = get_vector_store(text_chunks)
conversation = get_conversational_chain(vector_store)

# ===============================
# Request/Response Models
# ===============================
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# ===============================
# Endpoints
# ===============================
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    response = conversation({"question": req.question})
    return {"answer": response["answer"]}


@app.get("/")
def root():
    return {"message": "PDF QA Bot is running. Use POST /ask with {'question': '...'}"}
