import os
import requests
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

NEXUS_API_KEY = os.getenv("NEXUS_API_KEY")
GAME_DOMAIN = "newvegas"

headers = {
    "accept": "application/json",
    "apikey": NEXUS_API_KEY
}

# 1. The Hardcoded Essentials
# These are the absolute mandatory mods that the AI *must* know perfectly.
ESSENTIAL_MOD_IDS = [
    58277,  # JIP LN NVSE Plugin
    51664,  # YUP - Base Game and All DLC
    62552,  # JohnnyGuitar NVSE
    68714,  # NVTF - New Vegas Tick Fix
]

def clean_html(raw_html):
    """Strips HTML tags from the Nexus Mods description."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n").strip()

def fetch_and_parse_mod(mod_id):
    """Fetches mod data and formats it for LangChain."""
    url = f"https://api.nexusmods.com/v1/games/{GAME_DOMAIN}/mods/{mod_id}.json"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        page_content = (
            f"Mod Name: {data.get('name')}\n"
            f"Version: {data.get('version')}\n"
            f"Endorsements: {data.get('endorsement_count')}\n\n"
            f"Description:\n{clean_html(data.get('description'))}"
        )
        
        metadata = {
            "mod_id": mod_id,
            "name": data.get("name"),
            "version": data.get("version"),
            "source": url
        }
        
        return {"page_content": page_content, "metadata": metadata}
        
    elif response.status_code == 429:
        print(f"Rate limit hit on mod {mod_id}. Sleeping...")
        time.sleep(60)
        return None
    else:
        print(f"Error {response.status_code} fetching mod {mod_id}")
        return None

def fetch_latest_trending_mods():
    """
    Queries the API for recently updated or trending mods.
    This expands the database automatically over time.
    """
    url = f"https://api.nexusmods.com/v1/games/{GAME_DOMAIN}/mods/latest_updated.json"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # The API returns a list of recent mods
        recent_mods = response.json()
        # Extract the mod IDs from the list, capping it to avoid rate limits
        return [mod["mod_id"] for mod in recent_mods[:20]] 
    else:
        print("Failed to fetch trending mods.")
        return []

def run_ingestion_pipeline():
    documents = []
    
    print("Step 1: Ingesting Essential Mods...")
    for mod_id in ESSENTIAL_MOD_IDS:
        doc = fetch_and_parse_mod(mod_id)
        if doc:
            documents.append(doc)
        time.sleep(1) # Be gentle on the API
        
    print("Step 2: Discovering New/Trending Mods...")
    trending_ids = fetch_latest_trending_mods()
    
    for mod_id in trending_ids:
        # Prevent duplicating the essentials
        if mod_id not in ESSENTIAL_MOD_IDS:
            doc = fetch_and_parse_mod(mod_id)
            if doc:
                documents.append(doc)
            time.sleep(1) # Be gentle on the API
            
    print(f"Ingestion complete. Prepared {len(documents)} mod documents for the vector database.")
    return documents

if __name__ == "__main__":
    print("Initializing Nexus Mods Ingestion Pipeline...")
    
    # Check if the API key is actually loaded
    if not NEXUS_API_KEY:
        print("ERROR: NEXUS_API_KEY environment variable is not set. Please set it before running.")
    else:
        # Run the pipeline and capture the documents
        parsed_docs = run_ingestion_pipeline()
        
        print(f"Successfully processed {len(parsed_docs)} mods.")
        
        # Print out the names of the mods we successfully fetched as a test
        langchain_documents = []
        for doc in parsed_docs:
            mod_name = doc['metadata']['name'] if doc['metadata']['name'] else f"Unknown Mod (ID: {doc['metadata']['mod_id']})"
            print(f" - Preparing: {mod_name}")
            
            langchain_documents.append(
                Document(
                    page_content=doc['page_content'],
                    metadata=doc['metadata']
                )
            )
            
        # 3. Chunk the documents so they fit into the LLM context window cleanly
        print("\nChunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_chunks = text_splitter.split_documents(langchain_documents)
        print(f"Created {len(final_chunks)} text chunks from {len(langchain_documents)} mods.")
        
        # 4. Initialize Embeddings and save to local ChromaDB directory
        print("\nEmbedding and saving to ChromaDB...")
        
        # Replace this with your actual embedding setup (e.g., HuggingFaceEmbeddings() if offline)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        
        # This will create or update a local folder named 'chroma_db' in your project
        db = Chroma.from_documents(
            documents=final_chunks, 
            embedding=embeddings, 
            persist_directory="./vnv_chroma_db"
        )
        
        print("Database successfully updated! Your Streamlit app can now read these mods.")
# lang_chain_docs = run_ingestion_pipeline()
# chroma_db.add_documents(lang_chain_docs)