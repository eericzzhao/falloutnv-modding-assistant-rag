import os
import pickle
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Set the directory containing the HTML files and the database
REPO_DIR = "./Viva-New-Vegas"
DB_DIR = "./vnv_chroma_db"

# Extract data from the HTML files and cleans the text (remove whitespace)
def extract_vnv_data():
    documents = []
    
    # This is to check if the cloned vnv repo is still there
    if not os.path.exists(REPO_DIR):
        print(f"ERROR: The directory '{REPO_DIR}' does not exist! Did you clone it here?")
        return documents

    # Iterate through all of the files in the cloned VNV repository
    for root, _, files in os.walk(REPO_DIR):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")

                    # Extract text, add newlines between elements
                    text = soup.get_text(separator="\n", strip=True)

                    # Create a LangChain document with the source file name
                    # IMPORTANT for filtering
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file}
                    ))

    return documents

def build_pipeline():
    print("Howdy pardner! First things first, gotta extract and clean the HTML data...")
    raw_documents = extract_vnv_data()
    print(f"Finished pardner, we've extracted {len(raw_documents)} pages of data.")

    print("\nMight I move on ahead and start chunking the text...")
    # Splitting the text makes it so that Kimi is easily digestbile during retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Finished pardner, we've chunked the data into {len(chunks)} pieces.")

    # Load the local embedding model
    # lowk should consider changing this model 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Generate the vectors and save them to the disk
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    # save the raw chunks into disk so our BM25 retriever can access them in app.py
    chunks_path = os.path.join(DB_DIR, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Woohoo pardner! The pipeline is now complete. I've gone ahead and had the Vector DB saved to {DB_DIR}")

if __name__ == "__main__":
    build_pipeline()


