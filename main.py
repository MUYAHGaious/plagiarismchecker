# Final cleaned-up and complete version of your backend code
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import fitz  # PyMuPDF
from docx import Document
import io
import logging

app = FastAPI()
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# In-memory index and mapping
index = None
doc_store = {}  # id: (sentence, doc_name)

# File paths
INDEX_FILE = "index.faiss"
STORE_FILE = "doc_store.pkl"

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Load existing index and doc_store if available
def load_state():
    global index, doc_store
    if os.path.exists(INDEX_FILE) and os.path.exists(STORE_FILE):
        with open(STORE_FILE, "rb") as f:
            doc_store = pickle.load(f)
        index = faiss.read_index(INDEX_FILE)
        logger.info("Loaded existing index and doc_store")
    else:
        index = faiss.IndexFlatL2(384)
        logger.info("Created new index")

# Save current state
def save_state():
    faiss.write_index(index, INDEX_FILE)
    with open(STORE_FILE, "wb") as f:
        pickle.dump(doc_store, f)
    logger.info("Saved index and doc_store")

# Load initial state
load_state()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Text extraction methods
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = Document(io.BytesIO(file.read()))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text(file: UploadFile):
    if file.filename.endswith(".pdf"):
        return extract_text_from_pdf(file.file)
    elif file.filename.endswith(".docx"):
        return extract_text_from_docx(file.file)
    elif file.filename.endswith(".txt"):
        return extract_text_from_txt(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

# Index a document
@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    global index, doc_store
    try:
        text = extract_text(file)
        sentences = [s for s in text.split(".") if len(s.strip()) > 20]
        embeddings = model.encode(sentences)

        start_id = len(doc_store)
        for i, sentence in enumerate(sentences):
            doc_store[start_id + i] = (sentence, file.filename)

        index.add(embeddings)
        save_state()

        logger.info(f"Indexed {len(sentences)} sentences from {file.filename}")
        return {"message": f"Indexed {len(sentences)} sentences from {file.filename}"}
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error indexing document")

# Check document for plagiarism
@app.post("/check")
async def check_document(file: UploadFile = File(...)):
    try:
        if index.ntotal == 0:
            logger.info("No documents indexed yet")
            raise HTTPException(status_code=400, detail="No documents indexed yet.")

        text = extract_text(file)
        sentences = [s for s in text.split(".") if len(s.strip()) > 20]
        embeddings = model.encode(sentences)

        k = 1
        D, I = index.search(embeddings, k)

        matches = []
        matched = 0
        for i, (distances, indices) in enumerate(zip(D, I)):
            score = 1 - distances[0] / 4  # pseudo-similarity
            if score > 0.75:
                source_text, source_file = doc_store[indices[0]]
                matches.append({
                    "input": sentences[i],
                    "matched_with": source_text,
                    "source": source_file,
                    "score": round(score, 2)
                })
                matched += 1

        match_percentage = round((matched / len(sentences)) * 100, 2) if sentences else 0
        logger.info(f"Checked document with {len(sentences)} sentences and found {matched} matches")
        return {"match_percentage": match_percentage, "matches": matches}
    except Exception as e:
        logger.error(f"Error checking document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking document")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
