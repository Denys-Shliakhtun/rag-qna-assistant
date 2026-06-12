from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
from sentence_transformers import SentenceTransformer
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

conn = psycopg2.connect("postgresql://user:password@localhost:5432/rag_db")
conn.autocommit = True
cursor = conn.cursor()

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding vector(384)
    );
""")

class Document(BaseModel):
    text: str

class Query(BaseModel):
    question: str
    k: int = 3

@app.post("/add_document")
def add_document(doc: Document):
    clean_text = re.sub(r'\s+', ' ', doc.text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = text_splitter.split_text(clean_text)
    vectors = embedder.encode(chunks).tolist()
    for chunk, vector in zip(chunks, vectors):
        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (chunk, vector)
        )
        
    return {"status": "success", "chunks_added": len(chunks)}

@app.post("/query")
def query_documents(q: Query):
    query_vector = embedder.encode(q.question).tolist()
    cursor.execute("""
        SELECT content FROM documents 
        ORDER BY embedding <=> %s::vector 
        LIMIT %s
    """, (query_vector, q.k))
    return {"top_chunks": [row[0] for row in cursor.fetchall()]}