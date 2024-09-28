import os
import json
import numpy as np
from pydantic import BaseModel

from fastapi import FastAPI, Query
import uvicorn
from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
from groq import Groq
from fastapi import FastAPI, Query, HTTPException
from urllib.parse import unquote
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

groq_key="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"
groq_client = Groq(api_key=groq_key)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Initialize models
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def load_embeddings(self):
        embeddings = []
        
        # Check if the embeddings directory exists
        if not os.path.exists(self.embeddings_path):
            print(f"Embeddings directory not found: {self.embeddings_path}")
            return np.array([])

        embedding_files = [f for f in os.listdir(self.embeddings_path) if f.endswith('.npy')]
        
        for file in embedding_files:
            file_path = os.path.join(self.embeddings_path, file)
            try:
                embedding = np.load(file_path)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if not embeddings:
            print("No embeddings were loaded.")
            return np.array([])

        # Combine all embeddings into a single numpy array
        return np.concatenate(embeddings, axis=0)

import os
import json
import numpy as np

class LocalEmbeddingsStorage:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(self.path, 'documents.json'), 'r') as f:
            self.documents = json.load(f)
        with open(os.path.join(self.path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        embeddings = []
        embeddings_dir = os.path.join(self.path, 'embeddings')
        
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

        for i in range(12):  # Assuming there are 12 heads
            embedding_path = os.path.join(embeddings_dir, f'embeddings_head_{i}.npy')
            if os.path.exists(embedding_path):
                embeddings.append(np.load(embedding_path))
            else:
                print(f"Warning: Embedding file not found: {embedding_path}")
                break  # Stop if we don't find the next numbered file
        
        if not embeddings:
            raise FileNotFoundError("No embedding files found")
        
        return np.array(embeddings)

    def query(self, query_embeddings, n_results=3):
        similarities = np.zeros(len(self.documents))
        
        for head_idx, head_query_embedding in enumerate(query_embeddings):
            head_embeddings = self.embeddings[head_idx]
            dot_product = np.dot(head_embeddings, head_query_embedding)
            norm_product = np.linalg.norm(head_embeddings, axis=1) * np.linalg.norm(head_query_embedding)
            head_similarity = dot_product / norm_product
            similarities += head_similarity
        
        similarities /= len(query_embeddings)  # Normalize by number of heads
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        
        results = {
            'documents': [self.documents[idx] for idx in top_indices],
            'metadatas': [self.metadata[idx] for idx in top_indices],
            'distances': [1 - similarities[idx] for idx in top_indices]
        }
        return results

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.embeddings[idx]

# Initialize local embeddings storage
persist_directory = "./local_embeddings"
collection = LocalEmbeddingsStorage(persist_directory)

def get_embeddings(text: str) -> List[np.ndarray]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # We'll use the last layer's attention outputs
    last_layer_attentions = outputs.attentions[-1]
    last_hidden_state = outputs.last_hidden_state[0]
    
    # Initialize list to store embeddings from all heads of the last layer
    all_head_embeddings = []
    
    for head in range(last_layer_attentions.size(1)):
        head_attention = last_layer_attentions[0, head, :, :]
        head_embedding = torch.matmul(head_attention, last_hidden_state)
        head_embedding = head_embedding.mean(dim=0)
        all_head_embeddings.append(head_embedding.cpu().numpy())
    
    return all_head_embeddings
response=""
query_res=""

class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
@app.post("/query")
async def query_documents(request: QueryRequest):

    # Decode the URL-encoded query and remove any trailing parameters
    decoded_query = request.query
    n_results = 5
    global query_res
    query_res=decoded_query
    print(f"Decoded query: {decoded_query}")
    query_embeddings = get_embeddings(decoded_query)
    results = collection.query(query_embeddings=query_embeddings, n_results=n_results)

    formatted_results = []
    for doc, metadata, distance in zip(results['documents'], results['metadatas'], results['distances']):
        formatted_results.append({
            "content": doc[:200] + "..." if len(doc) > 200 else doc,
            "source": metadata.get("source", "Unknown"),
            "similarity_score": round(1 - distance, 4)
        })
    
    response = generate_response(decoded_query, results=formatted_results)
    
    print(f"Generated response: {response}")
    
    return {
        "query": decoded_query,
        "results": formatted_results,
        "response": response
    }
def generate_response(query: str, results: list) -> str:
    # Prepare the context from the search results
    context = "\n\n".join([f"Content: {r['content']}\nSource: {r['source']}" for r in results])
    
    # Prepare the prompt for Groq
    print(query,"query124")
    prompt = f"""Analyze the following query and the provided context carefully. Synthesize a comprehensive, coherent answer based on the information from the top 3 most relevant context pieces. Combine important details from all sources to create a well-rounded response. Do not use bullet points or numbered lists. Ensure the answer flows naturally and addresses the user's intent.
    Query: {query}
    
    Context:
    {context}

    Answer:"""

    # Query Groq API
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context. If the query cannot be answered from the context, clearly state that."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-70B-versatile",
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


@app.get("/answer")
async def get_answer(query: str = Query(..., description="The query string to search for")):
    try:
        # Perform the document search
        query_embeddings = get_embeddings(query)
        results = collection.query(query_embeddings=query_embeddings, n_results=3)

        formatted_results = []
        for doc, metadata, distance in zip(results['documents'], results['metadatas'], results['distances']):
            formatted_results.append({
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "source": metadata.get("source", "Unknown"),
                "similarity_score": round(1 - distance, 4)
            })

        # Generate the response using Groq
        answer = await generate_response(query_res, formatted_results)

        # Return the generated answer
        return {"answer": answer}

    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)