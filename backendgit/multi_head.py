# # Import necessary libraries
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import faiss
# import numpy as np
# import torch
# from fastapi import FastAPI, Query
# import uvicorn
# import groq
# from typing import List, Dict
# import os
# import json
# from documentapi import get_embeddings
# # Initialize Groq client
# groq_api_key = "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"  # Replace with your actual Groq API key
# groq_client = groq.Groq(api_key=groq_api_key)

# # Initialize FastAPI app
# app = FastAPI()

# # Step 1: Load pre-trained models
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# lm_model = AutoModelForCausalLM.from_pretrained("gpt2")

# # Step 2: Initialize local embeddings storage
# class LocalEmbeddingsStorage:
#     def __init__(self, path):
#         self.path = path
#         self.documents = []
#         self.embeddings = []
#         self.metadata = []
#         self.load_data()

#     def load_data(self):
#         with open(os.path.join(self.path, 'documents.json'), 'r') as f:
#             self.documents = json.load(f)
#         with open(os.path.join(self.path, 'metadata.json'), 'r') as f:
#             self.metadata = json.load(f)
        
#         embeddings_dir = os.path.join(self.path, 'embeddings')
#         layer_dirs = [d for d in os.listdir(embeddings_dir) if d.startswith('layer_')]
        
#         for layer_dir in sorted(layer_dirs):
#             layer_embeddings = []
#             layer_path = os.path.join(embeddings_dir, layer_dir)
#             head_files = [f for f in os.listdir(layer_path) if f.endswith('.npy')]
#             for head_file in sorted(head_files):
#                 head_embeddings = np.load(os.path.join(layer_path, head_file))
#                 layer_embeddings.append(head_embeddings)
#             self.embeddings.append(layer_embeddings)

#     def query(self, query_embeddings, n_results=3):
#         similarities = np.zeros(len(self.documents))
        
#         for layer_idx, layer_query_embeddings in enumerate(query_embeddings):
#             for head_idx, head_query_embedding in enumerate(layer_query_embeddings):
#                 head_embeddings = self.embeddings[layer_idx][head_idx]
#                 dot_product = np.dot(head_embeddings, head_query_embedding)
#                 norm_product = np.linalg.norm(head_embeddings, axis=1) * np.linalg.norm(head_query_embedding)
#                 head_similarity = dot_product / norm_product
#                 similarities += head_similarity
        
#         similarities /= (len(query_embeddings) * len(query_embeddings[0]))  # Normalize by total number of heads
#         top_indices = np.argsort(similarities)[-n_results:][::-1]
        
#         results = {
#             'documents': [self.documents[idx] for idx in top_indices],
#             'metadatas': [self.metadata[idx] for idx in top_indices],
#             'distances': [1 - similarities[idx] for idx in top_indices]
#         }
#         return results

# # Initialize local embeddings storage
# persist_directory = "./local_embeddings"
# collection = LocalEmbeddingsStorage(persist_directory)

# # Initialize sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Step 3: Process embeddings

# # Step 5: Implement multi-head attention for aspect-aware retrieval
# # class MultiHeadRetrieval(torch.nn.Module):
# #     def __init__(self, input_dim, num_heads):
# #         super().__init__()
# #         self.num_heads = num_heads
# #         self.attention = torch.nn.MultiheadAttention(input_dim, num_heads)
    
# #     def forward(self, query_embedding):
# #         # Ensure query_embedding is 3D: (seq_len, batch_size, input_dim)
# #         if query_embedding.dim() == 1:
# #             query_embedding = query_embedding.unsqueeze(0).unsqueeze(1)
# #         elif query_embedding.dim() == 2:
# #             query_embedding = query_embedding.unsqueeze(1)
        
# #         # Apply multi-head attention
# #         attn_output, _ = self.attention(query_embedding, query_embedding, query_embedding)
# #         return attn_output.mean(dim=0)

# # # Initialize multi-head retrieval model
# # multi_head_retrieval = MultiHeadRetrieval(dimension, num_heads=4)

# # Step 6: RAG function
# # def get_multi_head_embeddings(text):
# #     # Generate base embedding
# #     base_embedding = sentence_model.encode(text)
    
# #     # Apply multi-head attention
# #     with torch.no_grad():
# #         multi_head_emb = multi_head_retrieval(torch.tensor(base_embedding))
    
# #     # Convert to list of embeddings
# #     return [emb.cpu().numpy().tolist() for emb in multi_head_emb]

# # API endpoint for querying documents
# @app.post("/query")
# async def query_documents(
#     query: str = Query(..., description="The query string to search for"),
#     n_results: int = Query(3, description="Number of results to return")
# ):
#     # Generate embedding for the query
#     query_embeddings = get_embeddings(query)
    
#     # Query local embeddings storage
#     results = collection.query(
#         query_embeddings=query_embeddings,
#         n_results=n_results
#     )

#     # Format the results
#     formatted_results = []
#     for doc, metadata, distance in zip(results['documents'], results['metadatas'], results['distances']):
#         formatted_results.append({
#             "content": doc[:200] + "..." if len(doc) > 200 else doc,  # Show first 200 characters
#             "source": metadata.get("source", "Unknown"),
#             "similarity_score": round(1 - distance, 4)
#         })

#     return {
#         "query": query,
#         "results": formatted_results
#     }

# # Run the FastAPI app
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8005)
import os
import json
import numpy as np
from fastapi import FastAPI, Query
import uvicorn
from typing import List
from transformers import AutoModel, AutoTokenizer
import torch

app = FastAPI()

# Initialize models
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class LocalEmbeddingsStorage:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(self.path, 'documents.json'), 'r') as f:
            self.documents = json.load(f)
        with open(os.path.join(self.path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        self.embeddings_path = os.path.join(self.path, 'embeddings_.npy')
        self.embeddings = np.load(self.embeddings_path)

    def query(self, query_embeddings, n_results=3):
        similarities = np.zeros(len(self.documents))
        
        for head_idx, head_query_embedding in enumerate(query_embeddings):
            head_embeddings = self.embeddings[:, head_idx, :]
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

@app.post("/query")
async def query_documents(
    query: str = Query(..., description="The query string to search for"),
    n_results: int = Query(3, description="Number of results to return")
):
    query_embeddings = get_embeddings(query)
    results = collection.query(query_embeddings=query_embeddings, n_results=n_results)

    formatted_results = []
    for doc, metadata, distance in zip(results['documents'], results['metadatas'], results['distances']):
        formatted_results.append({
            "content": doc[:200] + "..." if len(doc) > 200 else doc,
            "source": metadata.get("source", "Unknown"),
            "similarity_score": round(1 - distance, 4)
        })

    return {"query": query, "results": formatted_results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)