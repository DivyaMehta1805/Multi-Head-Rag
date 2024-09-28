import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader

import torch
from typing import List
from transformers import AutoModel, AutoTokenizer

# Initialize paths and model
persist_directory = "./local_embeddings"
os.makedirs(persist_directory, exist_ok=True)

# Download NLTK data if needed
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

# Initialize the model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_embeddings(text: str) -> List[np.ndarray]:
    """
    Generate embeddings for a given text, capturing output from all attention heads in the last layer.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    last_layer_attentions = outputs.attentions[-1]
    last_hidden_state = outputs.last_hidden_state[0]
    
    all_head_embeddings = []
    
    for head in range(last_layer_attentions.size(1)):
        head_attention = last_layer_attentions[0, head, :, :]
        head_embedding = torch.matmul(head_attention, last_hidden_state)
        head_embedding = head_embedding.mean(dim=0)
        all_head_embeddings.append(head_embedding.cpu().numpy())
    
    return all_head_embeddings

def process_pdf_file(file_path):
    """Process a PDF file, split text into chunks, generate embeddings, and save locally."""
    # Read PDF file
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text = remove_stopwords(text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)  # Split text into chunks
    
    embeddings_list = []
    documents_list = []
    metadatas_list = []
    
    for i, chunk in enumerate(chunks):
        chunk_embeddings = get_embeddings(chunk)
        embeddings_list.append(chunk_embeddings)
        documents_list.append(chunk)
        metadatas_list.append({"source": file_path, "chunk_index": i})

    embeddings_dir = os.path.join(persist_directory, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)

    num_heads = len(embeddings_list[0])
    
    # Save embeddings for each attention head separately
    for head in range(num_heads):
        head_embeddings = [chunk_embeddings[head] for chunk_embeddings in embeddings_list]
        np.save(os.path.join(embeddings_dir, f'embeddings_head_{head}.npy'), np.array(head_embeddings))

    # Save documents and metadata
    with open(os.path.join(persist_directory, 'documents.json'), 'w') as doc_file:
        json.dump(documents_list, doc_file, indent=4)
    
    with open(os.path.join(persist_directory, 'metadata.json'), 'w') as meta_file:
        json.dump(metadatas_list, meta_file, indent=4)

    return (f"PDF file processed and embeddings from {len(embeddings_list)} chunks across "
            f"{num_heads} attention heads from the last layer stored locally in {embeddings_dir}.")

def get_valid_pdf_path():
    """Prompt the user for a valid PDF file path."""
    while True:
        pdf_file_path = input("Please enter the path to your PDF file: ").strip()
        
        if not pdf_file_path:
            print("Error: You must enter a file path.")
        elif not os.path.isfile(pdf_file_path):
            print(f"Error: The file '{pdf_file_path}' does not exist.")
        elif not pdf_file_path.lower().endswith('.pdf'):
            print(f"Error: The file '{pdf_file_path}' is not a PDF file.")
        else:
            return pdf_file_path

if __name__ == "__main__":
    pdf_file_path = get_valid_pdf_path()
    result = process_pdf_file(pdf_file_path)
    print(result)