import os
import requests
import time
import pickle
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv() 

NEXUS_API_KEY = os.getenv("NEXUS_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GAME_DOMAIN = "newvegas"

# State tracking files
QUEUE_FILE = "mod_queue.txt"
PROCESSED_FILE = "processed_mods.txt"
DAILY_LIMIT = 50

headers = {
    "accept": "application/json",
    "apikey": NEXUS_API_KEY
}

def clean_html(raw_html):
    if not raw_html: return ""
    return BeautifulSoup(raw_html, "html.parser").get_text(separator="\n").strip()

def load_tracker(filepath):
    """Loads IDs from a text file into a list."""
    if not os.path.exists(filepath): return []
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip().isdigit()]

def save_tracker(filepath, data_list):
    """Saves a list of IDs back to a text file."""
    with open(filepath, "w") as f:
        for item in data_list:
            f.write(f"{item}\n")

def fetch_and_parse_mod(mod_id):
    """Fetches mod data from Nexus API and formats it."""
    url = f"https://api.nexusmods.com/v1/games/{GAME_DOMAIN}/mods/{mod_id}.json"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        page_content = (
            f"[NEXUS MODS LIVE DATA]\n"
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
        print(f"Rate limit hit on mod {mod_id}.")
    return None

def fetch_dynamic_discovery_mods():
    """Finds brand new and trending mods dynamically."""
    discovered_ids = []
    endpoints = ["trending.json", "latest_added.json", "latest_updated.json"]
    
    for ep in endpoints:
        url = f"https://api.nexusmods.com/v1/games/{GAME_DOMAIN}/mods/{ep}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            discovered_ids.extend([str(mod["mod_id"]) for mod in response.json()[:5]])
    return list(set(discovered_ids))

def run_drip_feed_pipeline():
    queue = load_tracker(QUEUE_FILE)
    processed = set(load_tracker(PROCESSED_FILE))
    documents = []
    
    print(f"Starting pipeline. {len(queue)} mods in queue. {len(processed)} mods already processed.")
    
    # 1. Take the top 50 from the queue
    batch_ids = queue[:DAILY_LIMIT]
    remaining_queue = queue[DAILY_LIMIT:]
    
    # 2. Add dynamically discovered new mods to the batch
    discovery_ids = fetch_dynamic_discovery_mods()
    for d_id in discovery_ids:
        if d_id not in processed and d_id not in batch_ids:
            batch_ids.append(d_id)
            
    print(f"Fetching data for {len(batch_ids)} mods...")
    
    successfully_processed = []
    
    for mod_id in batch_ids:
        if str(mod_id) in processed:
            continue
            
        doc = fetch_and_parse_mod(mod_id)
        if doc:
            documents.append(doc)
            successfully_processed.append(str(mod_id))
            processed.add(str(mod_id))
        time.sleep(1.5) # Anti-rate-limit buffer
        
    print(f"Successfully scraped {len(documents)} new documents.")
    
    # 3. Save the updated states back to disk
    save_tracker(QUEUE_FILE, remaining_queue)
    save_tracker(PROCESSED_FILE, list(processed))
    
    return documents

if __name__ == "__main__":
    if not NEXUS_API_KEY:
        print("ERROR: NEXUS_API_KEY environment variable is not set.")
        exit(1)
        
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("ERROR: QDRANT_URL or QDRANT_API_KEY environment variables are not set.")
        exit(1)
        
    parsed_docs = run_drip_feed_pipeline()
    
    if parsed_docs:
        langchain_documents = [
            Document(page_content=d['page_content'], metadata=d['metadata']) 
            for d in parsed_docs
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_chunks = text_splitter.split_documents(langchain_documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("Embedding and uploading to Qdrant Cloud...")
        db = QdrantVectorStore.from_documents(
            documents=final_chunks, 
            embedding=embeddings, 
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name="fnvma"
        )
        print("Database expansion complete!")
        chunks_path = "chunks.pkl"

        # load existing chunks if the file exists
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                existing_chunks = pickle.load(f)
        else:
            existing_chunks = []

        # appending the exisitng nexus mod chunks with the new ones
        existing_chunks.extend(final_chunks)

        # save the updated master list back to disk
        with open(chunks_path, "wb") as f:
            pickle.dump(existing_chunks, f)
        
        print(f"Successfully added {len(final_chunks)} new chunks to the BM25 database!")
    else:
        print("No new mods to process today. Let's doomscroll on Nexus Mods again tomorrow!")