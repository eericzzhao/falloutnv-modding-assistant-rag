import os 
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# HyDE (Hypotehetical Document Embeddings): Query transformation
from langchain_classic.chains import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
# re-ranking imports
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

# this will load the environment variable from .env automaticlly
load_dotenv()

DB_DIR = "./vnv_chroma_db"


@st.cache_resource
def load_rag_chain():
    # we also set up the embeddings for HyDE first
    base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm_for_hyde = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,max_retries=3)
    hyde_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are an expert Fallout: New Vegas modder. "
            "Please write a detailed paragraph answering the following question. "
            "Focus on technical details, file paths, and mod names. \n\n"
            "Question: {question}\n\nAnswer:"
        )
    )

    #wrap the base embeddings in the HyDE engine
    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=llm_for_hyde,
        base_embeddings=base_embeddings,
        custom_prompt=hyde_prompt
    )

    # 1. Hybrid search w/ HyDE: setting up the Dense Retriever (Chroma + k-Nearest Neighbors)

    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=hyde_embeddings)
    retriever_dense = vector_db.as_retriever(search_kwargs={"k": 15})

    # 2. setup the sparse retriever (BM25)
    with open(os.path.join(DB_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    retriever_sparse = BM25Retriever.from_documents(chunks)
    retriever_sparse.k = 15

    # 3. Combine into an Ensemble Retriever - weights tweakble 0.5/0.5 = equal split
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_sparse, retriever_dense],
        weights = [0.5, 0.5]
    )

    # setting up the cross-encoder and the contextual compressor
    # analyze 15 chunks --> return only the best 5
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder_model,top_n=5)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # LangChain and Gemini will automatically look for the api key
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=3)

    system_prompt = (
        "You are an expert Fallout: New Vegas modding assistant. \n"
        "Use the following pieces of retrieved context to answer the question. \n"
        "If you do not know the answer or if it's not in the contet, " \
        "say exactly:\n"
        "I cannot find that in the official modern modding guides. To prevent game instability, I won't guess\n"
        "Do not recommend outdated or broken mods like Project Nevada, New Vegas Stutter Remover, or Zan AutoPurge.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(compression_retriever, question_answer_chain)

# Streamlit UI 
st.set_page_config(page_title="FNVMA - Fallout: New Vegas Modding Assistant", page_icon="🎲", layout="centered")

st.title("🎲 FNVMA - Fallout: New Vegas Modding Assistant")
st.markdown("---")

# checking if the env variable is loaded to give the user a warning if it's missing
if not os.environ.get("GOOGLE_API_KEY"):
    st.error("Woah there pardner! It looks like youre missing your `GOOGLE_API_KEY`. Please make sure it is defined in your local `.env` file.")
else:
    # OTHERWISE, we can just initialize the pipeline with the key
    try:
        rag_chain = load_rag_chain()
        st.success("Well I might be fit as a fiddle pardner! The knowledge base has been uploaded. I am ready for your questions.")
    except Exception as e:
        st.error(f"Yikes! Something went wrong while loading the knowledge base. Please check your `GOOGLE_API_KEY` and make sure it is correct. Error details: {str(e)}")

user_question = st.text_input(
    "Howdy pardner! How can I help you with your Fallout: New Vegas modding experience?",
    placeholder="e.g., Why should I avoid the New Vegas Stutter Remover?"
)

if st.button("[Science 100] Answer my question robot!", type="primary"):
    if not user_question:
        st.warning("Hold your horses! Please enter a question before asking me.")
    elif os.environ.get("GOOGLE_API_KEY") == "YOUR_GEMINI_API_KEY_HERE":
        st.error("Hold it there pardner! Make sure to that your Google API key has been uploaded to the script or environment variables first!")
    else:
        with st.spinner("Patrolling the Mojave almost makes you wish for a nuclear winter..."):
            try:
                # invoking the chain
                response = rag_chain.invoke({"input": user_question})

                st.markdown("Wires beep and machines clang. It says that...")
                st.write(response["answer"])

                # displaying the actual sourced files + chunks
                with st.expander("🔍 View Retrieved Context (Developer Mode)"):
                    for i, doc in enumerate(response["context"]):
                        source_file = doc.metadata.get("source", "Unknown Source")
                        st.markdown(f"**Chunk {i+1} - From: `{source_file}`**")
                        st.caption(doc.page_content)
                        st.markdown("---")

            except Exception as e:
                st.error(f"An error occured during generation: {e}")
