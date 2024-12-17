# test_fastapi_rag

1. Setup and Prerequisites
Technologies: Install required libraries via pip install fastapi uvicorn langchain pypdf sentence-transformers azure-cognitiveservices-search.
Azure Setup: Ensure the OpenAI resource is configured in Azure with the deployment name (embedding-ada002) and the API key is available.
Vector DB: Use a vector database like FAISS or Pinecone for storing embeddings.
2. Plan the Architecture
You will implement five core components as separate FastAPI routes:

Extract text from PDF (/extract-pdf).
Chunk the extracted text (/chunk-text).
Store chunks in the vector database (/store-embeddings).
Query the vector database and provide context (/query-db).
Integrate with GPT-4o-mini for a response (/get-response).
3. Code Implementation
Below is the detailed implementation of each component.

a. Import Necessary Libraries
python
Copy code
from fastapi import FastAPI, UploadFile, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import os
b. Initialize FastAPI and Configuration
python
Copy code
app = FastAPI()

# Azure OpenAI Configuration
AZURE_API_KEY = "your_api_key_here"
EMBEDDING_DEPLOYMENT_NAME = "embedding-ada002"
LLM_DEPLOYMENT_NAME = "gpt-4o-mini"

# Placeholder for Vector Database
vector_db = None
4. Implement API Routes
1. Extract Text from PDF
This route extracts text from an uploaded PDF.

python
Copy code
@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return {"text": text}
2. Chunk the Extracted Text
Divide the text into manageable chunks for embedding.

python
Copy code
@app.post("/chunk-text")
async def chunk_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return {"chunks": chunks}
3. Store Chunks in the Vector Database
Store the chunks in FAISS (or another vector database) using Azure OpenAI embeddings.

python
Copy code
@app.post("/store-embeddings")
async def store_embeddings(chunks: list):
    global vector_db  # To persist the database across requests
    embeddings = OpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT_NAME,
        openai_api_key=AZURE_API_KEY
    )
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    return {"message": "Chunks stored successfully in the vector database."}
4. Query the Vector Database
Retrieve the most relevant chunks based on the user’s query.

python
Copy code
@app.post("/query-db")
async def query_db(query: str):
    if vector_db is None:
        raise HTTPException(status_code=400, detail="Vector database is not initialized.")
    
    retrieved_docs = vector_db.similarity_search(query, k=5)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    return {"context": context}
5. Generate Response Using GPT-4o-mini
Provide the query and context to the GPT-4o-mini model for the response.

python
Copy code
@app.post("/get-response")
async def get_response(query: str):
    # Retrieve context from the vector database
    context_result = await query_db(query)
    context = context_result["context"]
    
    # Combine query and context
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    llm = OpenAI(
        deployment=LLM_DEPLOYMENT_NAME,
        api_key=AZURE_API_KEY,
        temperature=0.7,
        max_tokens=200
    )
    response = llm(prompt)
    return {"response": response}
5. Run the Application
Save the code to a file, e.g., app.py. Start the FastAPI application with:

bash
Copy code
uvicorn app:app --reload
Access the API at http://127.0.0.1:8000.

6. Testing the API
Extract Text from PDF:
Use a tool like Postman or curl to upload a PDF file to /extract-pdf.
Chunk the Text:
Send the extracted text to /chunk-text.
Store Chunks in the Vector DB:
Pass the chunks to /store-embeddings.
Query the Vector DB:
Provide a question to /query-db.
Generate GPT-4o Response:
Pass the query to /get-response and receive the response.
7. Enhancements
Vector DB Persistence: Replace FAISS with a persistent solution like Pinecone for production.
Error Handling: Add robust error handling for API calls and edge cases.
Authentication: Secure API routes with authentication if needed.
Testing: Write unit and integration tests for each route.
8. Folder Structure
bash
Copy code
project/
├── app.py                # Main FastAPI application
├── requirements.txt      # Python dependencies
├── data/                 # Directory for uploaded PDFs (optional)
└── tests/                # Unit and integration tests



#FULL RAG USING LAMGHCHAIN
# import os
# import json
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# import numpy as np

# # Paths for FAISS index and metadata
# VECTOR_STORE_PATH = "vector_store.index"
# METADATA_STORE_PATH = "metadata.json"

# # Initialize embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Load or create FAISS vector store
# def load_or_create_faiss():
#     if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(METADATA_STORE_PATH):
#         with open(METADATA_STORE_PATH, "r", encoding="utf-8") as f:
#             metadata = json.load(f)
#         db = FAISS.load_local(
#             VECTOR_STORE_PATH, 
#             embeddings=embedding_model, 
#             allow_dangerous_deserialization=True  # Ensure you trust the source
#         )
#     else:
#         # Add a default document to initialize FAISS
#         default_docs = [Document(page_content="This is a placeholder document.")]
#         db = FAISS.from_documents(default_docs, embedding_model)
#         metadata = [{"content": doc.page_content} for doc in default_docs]
#     return db, metadata

# db, metadata = load_or_create_faiss()

# # Save FAISS vector store and metadata
# def save_faiss_and_metadata():
#     db.save_local(VECTOR_STORE_PATH)
#     with open(METADATA_STORE_PATH, "w", encoding="utf-8") as f:
#         json.dump(metadata, f, indent=4)

# # Add documents to FAISS
# def add_documents_to_faiss(docs):
#     db.add_documents(docs)
#     metadata.extend([{"content": doc.page_content, **doc.metadata} for doc in docs])
#     save_faiss_and_metadata()
# if _name_ == "_main_":
#     # File paths for processing
#     files_to_process = ["Processed_ASPICE_BP_Dynamic_Alignment.json", "automotive_chatbot_data.json"]
#     # process_files_backend(files_to_process)

#     # Test FAISS query
#     test_query = "BP ID: SWE.1.BP1?"
#     results = query_faiss(test_query)
#     print("Query Results:")
#     for doc in results:
#         print(doc.page_content)

 # Load FAISS index and metadata
def load_faiss_and_metadata():
    if os.path.exists(VECTOR_STORE_PATH):
        faiss_index = faiss.read_index(VECTOR_STORE_PATH)
    else:
        faiss_index = faiss.IndexFlatL2(dimension) index, metadata = load_faiss_and_metadata()

# Save FAISS index and metadata
def save_faiss_and_metadata():
    faiss.write_index(index, VECTOR_STORE_PATH) embeddings = embedding_model.embed_documents(valid_entries)

    # Add embeddings and metadata to FAISS
    for idx, content in enumerate(valid_entries):
        index.add(np.array([embeddings[idx]], dtype="float32"))
        metadata.append({
            "content": content,
            "type": "json",
            "source": file_path
        })

    save_faiss_and_metadata()
for open ai embeddings change the prompt
    retriever = db.as_retriever()
    retriever.search_kwargs.update({"k": top_k})
    docs = retriever.get_relevant_documents(query)
    return docs
