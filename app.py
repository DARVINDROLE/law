import os
import json
import random
import tempfile
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import uvicorn

# ===============================
# FastAPI App Setup
# ===============================
app = FastAPI(
    title="PDF Flashcard Generator API",
    description="Generate flashcards from PDF documents using AI",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Pydantic Models
# ===============================
class Flashcard(BaseModel):
    question: str
    answer: str

class FlashcardResponse(BaseModel):
    flashcards: List[Flashcard]
    total_count: int
    message: str

class ErrorResponse(BaseModel):
    error: str
    message: str

# ===============================
# Groq API setup
# ===============================
try:
    llm = ChatGroq(
        temperature=0.3,
        groq_api_key="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",
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
    )
except Exception as e:
    print(f"Warning: Could not initialize Groq API: {e}")
    llm = None

# ===============================
# Helper Functions
# ===============================
def get_pdf_text(pdf_files: List[UploadFile]) -> str:
    """Extract text from uploaded PDF files"""
    text = ""
    
    for pdf_file in pdf_files:
        try:
            # Read the uploaded file content
            pdf_content = pdf_file.file.read()
            
            # Create a temporary file to work with PyPDF2
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name
            
            # Extract text using PyPDF2
            pdf_reader = PdfReader(temp_file_path)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Error processing PDF {pdf_file.filename}: {e}")
            continue
        finally:
            # Reset file pointer for potential reuse
            pdf_file.file.seek(0)
    
    return text

def get_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

def generate_flashcards_from_chunk(chunk: str, num_cards: int = 2) -> List[dict]:
    """Generate flashcards from a single text chunk"""
    if not llm:
        raise HTTPException(status_code=500, detail="Groq API not initialized")
    
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
        # Try to parse the JSON response
        flashcards_data = json.loads(response.content)
        return flashcards_data
    except json.JSONDecodeError:
        # Fallback: try to use eval (less safe but might work)
        try:
            return eval(response.content)
        except:
            print(f"⚠️ Error parsing flashcards from chunk")
            return []
    except Exception as e:
        print(f"⚠️ Error generating flashcards from chunk: {e}")
        return []

def generate_flashcards(text: str, final_count: int = 5) -> List[dict]:
    """Generate flashcards from full text"""
    if not text.strip():
        return []
    
    chunks = get_text_chunks(text)
    flashcards = []

    # Collect small batches from each chunk (limit to avoid overload)
    max_chunks = min(len(chunks), 5)
    for chunk in chunks[:max_chunks]:
        if len(chunk.strip()) < 50:  # Skip very short chunks
            continue
        chunk_cards = generate_flashcards_from_chunk(chunk, num_cards=2)
        flashcards.extend(chunk_cards)

    # Randomly pick exactly `final_count` cards if we have more than needed
    if len(flashcards) > final_count:
        flashcards = random.sample(flashcards, final_count)

    return flashcards

# ===============================
# API Endpoints
# ===============================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PDF Flashcard Generator API is running!",
        "version": "1.0.0",
        "endpoints": {
            "generate_flashcards": "/generate-flashcards",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check with API status"""
    return {
        "status": "healthy",
        "groq_api_available": llm is not None,
        "message": "API is running properly"
    }

@app.post("/generate-flashcards", response_model=FlashcardResponse)
async def generate_flashcards_endpoint(
    files: List[UploadFile] = File(..., description="PDF files to process"),
    count: int = Form(5, description="Number of flashcards to generate")
):
    """Generate flashcards from uploaded PDF files"""
    
    # Validate input
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if count < 1 or count > 20:
        raise HTTPException(status_code=400, detail="Count must be between 1 and 20")
    
    # Check file types
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is not a PDF"
            )
    
    try:
        # Extract text from PDFs
        print("📄 Extracting text from PDFs...")
        raw_text = get_pdf_text(files)
        
        if not raw_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the provided PDFs"
            )
        
        # Generate flashcards
        print(f"⏳ Generating {count} flashcards...")
        flashcards_data = generate_flashcards(raw_text, final_count=count)
        
        if not flashcards_data:
            raise HTTPException(
                status_code=500, 
                detail="Could not generate flashcards from the provided content"
            )
        
        # Convert to Pydantic models for validation
        flashcards = [Flashcard(**card) for card in flashcards_data]
        
        return FlashcardResponse(
            flashcards=flashcards,
            total_count=len(flashcards),
            message=f"Successfully generated {len(flashcards)} flashcards"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/generate-flashcards-simple")
async def generate_flashcards_simple(
    files: List[UploadFile] = File(...),
    count: Optional[int] = 5
):
    """Simplified endpoint that returns raw JSON"""
    
    try:
        # Extract text from PDFs
        raw_text = get_pdf_text(files)
        
        if not raw_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No text extracted", "flashcards": []}
            )
        
        # Generate flashcards
        flashcards = generate_flashcards(raw_text, final_count=count)
        
        return {
            "success": True,
            "flashcards": flashcards,
            "count": len(flashcards)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "flashcards": []}
        )

# ===============================
# Run the application
# ===============================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # assuming this file is named main.py
        host="0.0.0.0",
        port=8000,
        reload=True
    )
