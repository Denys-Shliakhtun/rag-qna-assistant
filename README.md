# RAG-based Q&A Assistant

A Retrieval-Augmented Generation (RAG) backend system that processes documents, stores their vector representations, and provides context-aware answers to user queries using Large Language Models.

## Tech Stack
- **Backend API:** FastAPI
- **Vector Database:** PostgreSQL with the `pgvector` extension
- **Embeddings:** `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Text Processing:** `langchain-text-splitters`
- **LLM:** Google Gemini (`gemini-2.5-flash-lite`) via the `google-genai` SDK

## Architecture
The project is split into two main microservices:

1. **Index Management API:** Handles receiving text, splitting it into logical overlapping chunks using `RecursiveCharacterTextSplitter`, generating 384-dimensional vector embeddings, and storing them in PostgreSQL. It also provides a similarity search endpoint using the cosine distance operator.
2. **Q&A Assistant API:** Receives user queries, requests the top-K closest context chunks from the Index Management API, and constructs a strict prompt for the Gemini model to generate a grounded answer.

## How to Run

**1. Start the Vector Database**
Initialize the PostgreSQL container using Docker:
`docker-compose up -d`

**2. Start the APIs**
Before running the Q&A service, ensure you have set your Gemini API key as an environment variable (e.g., `GEMINI_API_KEY`).

Run the Index Management API:
`uvicorn api_task1:app --port 8000 --reload`

Run the Q&A Assistant API:
`uvicorn api_task2:app --port 8001 --reload`

**3. Populate the Database**
Place your PDF files in the `library/` folder and run the document processing script to extract text and populate the vector database via the `/add_document` endpoint.

**4. Query the System**
Send a POST request to `http://127.0.0.1:8001/query` with a JSON payload containing your question.