import pickle
import os
from pdfminer.high_level import extract_text

PICKLE_FILE = "documents.pkl"


def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    return extract_text(pdf_path)


def save_document(documents):
    """Save all documents in a pickle file."""
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(documents, f)


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
        return "Document store is empty"


def process_pdf_folder(folder_path):
    """Process all PDF files in a folder and store them in a fresh pickle storage with numerical IDs."""
    if os.path.exists(PICKLE_FILE):
        os.remove(PICKLE_FILE)  # Delete old pickle file to start fresh

    documents = {}
    doc_id = 1  # Start numbering from 1

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            documents[doc_id] = {"content": text, "path": file_path}
            print(f"Stored {filename} as {doc_id}")
            doc_id += 1

    save_document(documents)


def main(folder_path):
    process_pdf_folder(folder_path)
    print("All PDFs have been processed and stored.")

    # Example usage of fetch_document
    doc_id = 1  # Example document ID to fetch
    doc_data = fetch_document(doc_id)
    if isinstance(doc_data, dict):
        print(
            f"Retrieved Document {doc_id} from {doc_data['path']}:\n{doc_data['content']}")  # Print first 500 characters
    else:
        print(doc_data)


if __name__ == "__main__":
    folder_path = "/Users/deepankar/Downloads/Clinical"  # Update with actual folder path containing PDFs
    main(folder_path)