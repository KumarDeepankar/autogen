import faiss
import ollama
import numpy as np
import pickle
from documents.fetch_sop import fetch_document


index = faiss.read_index("embeddings/faiss_index_file.idx")
model = 'nomic-embed-text'
query_response = ollama.embeddings(model=model, prompt="medical")
query_embedd = np.array([query_response['embedding']]).astype('float32')
# Normalize the query embedding
query_embedd = query_embedd / np.linalg.norm(query_embedd, axis=1, keepdims=True)
D, I = index.search(query_embedd, k=10)
distances, indices = D[0], I[0]
ret = []
for i in range(len(indices)):
    doc_id = indices[i]
    distance = distances[i]
    doc_data = fetch_document(doc_id)

    ret.append(
        {"doc_is": doc_id,
         "doc_path": doc_data['path'],
         "doc_content": doc_data['content']

         })

print(ret)