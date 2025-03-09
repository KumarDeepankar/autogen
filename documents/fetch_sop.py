import pickle
import os

# Get the absolute path to the pickle file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of fetch_sop.py
PICKLE_FILE = os.path.join(BASE_DIR, "documents.pkl")


def fetch_document(doc_id):
    """Fetch document content and path by ID from pickle file."""
    try:
        with open(PICKLE_FILE, "rb") as f:
            documents = pickle.load(f)
        doc_data = documents.get(doc_id)
        if doc_data:
            return doc_data
        else:
            return "Document not found"
    except (FileNotFoundError, EOFError):
        print(FileNotFoundError)
        return "Document store is empty"

