import faiss
import ollama
import numpy as np
import pickle

PICKLE_FILE = "../documents/documents.pkl"


def fetch_document():
    """Fetch document content and path by ID from pickle file."""
    try:
        with open(PICKLE_FILE, "rb") as f:
            documents = pickle.load(f)
            return documents
    except (FileNotFoundError, EOFError):
        return "Document store is empty"


index = faiss.IndexIDMap(faiss.IndexFlatL2(768))
# ollama model
model = 'nomic-embed-text'
docs = fetch_document()
for key, val in docs.items():
    # print(key, val.keys())
    doc_source = docs[key]["content"]
    doc_path = docs[key]["path"]
    print(key, doc_path)
    response = ollama.embeddings(model=model, prompt=doc_source)
    embedd_np_array = np.array([response['embedding']]).astype('float32')
    embedd_np_array_normalized = embedd_np_array / np.linalg.norm(embedd_np_array, axis=1, keepdims=True)
    doc_id= np.array(key).astype('int64')
    index.add_with_ids(embedd_np_array_normalized, doc_id)
# # Save the FAISS index to a file
faiss.write_index(index, 'faiss_index_file.idx')
print("Index saved to file.")
