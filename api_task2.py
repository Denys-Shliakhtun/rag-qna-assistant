from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import requests

app = FastAPI()
client = genai.Client(api_key="YOUR_API_KEY_HERE")

class Query(BaseModel):
    question: str

@app.post("/query")
def qa_assistant(q: Query):
    response = requests.post("http://127.0.0.1:8000/query", json={"question": q.question, "k": 3})
    retrieved_data = response.json()
    context = "\n\n".join(retrieved_data["top_chunks"])
    
    prompt = f"""
    You are a helpful Q&A assistant. Answer the user's question based ONLY on the provided context.
    
    Context:
    {context}
    
    User Question: {q.question}
    """
    
    ai_response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt
    )
    
    return {"answer": ai_response.text, "context_used": retrieved_data["top_chunks"]}