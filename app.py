from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

app = FastAPI()

# Allow CORS (frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq setup
llm = ChatGroq(
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),   # use Render environment variable
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# ---- Utility functions ----
def get_pdf_text(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_chunks(text, chunk_size=800, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def generate_flashcards_from_chunk(chunk, num_cards=2):
    prompt = f"""
    Create {num_cards} flashcards (question-answer pairs) from the following text.
    Format strictly as JSON list like:
    [
      {{"question": "Q1?", "answer": "A1"}},
      {{"question": "Q2?", "answer": "A2"}}
    ]
    Text:
    {chunk}
    """
    try:
        response = llm.invoke(prompt)
        return eval(response.content)
    except Exception as e:
        print("⚠️ Error:", e)
        return []

# ---- API endpoint ----
@app.get("/flashcards")
def get_flashcards():
    pdf_path = "NOTES UNIT-4 ANN.pdf"  # must exist in your repo or Render disk
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    text = get_pdf_text(pdf_path)
    chunks = get_chunks(text)

    flashcards = []
    for chunk in chunks[:5]:
        flashcards.extend(generate_flashcards_from_chunk(chunk, num_cards=2))

    # pick only 5 flashcards
    flashcards = random.sample(flashcards, min(5, len(flashcards)))

    return {"flashcards": flashcards}
