import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from google import genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import GoogleGemini
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

#       Currently, what is needed is the current implementation but using pinecone's vector databases for embeddings for store and retrieval:
#       - Storage:
#           - Memory
#           - RAG
#       - retrieval supports semantic search retrieval as well
#
#       Also, I can use Google Cloud Storage (GCS) for the documents themselves,
#       The Google Cloud Storage (GCS) free tier allows a monthly free usage limit of 5,000 Class A operations
#       (write and metadata operations like PUT, POST, and LIST requests)
#       and 50,000 Class B operations, which is what its really needed for
#       (mostly read operations like GET requests).
#
#       Also, use Google Gemini embeddings free plan to create cloud-based embeddings, so that should be used instead of SentenceTransformerEmbeddings
#       (The same key is used for all Gemini AI capabilities, including embeddings, text generation, and image generation)
#
#       If the above things are done, it's fully cloud-run
#
#       Theres a vibed version of this supposedly which is essentially the whole backbone for the chatbot in fully_cloud.py, although idk if it works

# API key is needed for both gpt model and embeddings model, we need to request users for their api key,
# probably have a button to have a pop-up guide to show how to, and a link to the page

# using langchain
#
# semantic search, and for chunking context
# - possibly replace FAISS with a cloud vector store (e.g., Vertex AI Matching Engine or Pinecone) to make memory fully cloud-based.
#
# use semantic search to find chunks and then send it as a part of your query
# have memory

"""
Structure:
[System instructions]
[Memory retrieved from past conversation]
[Relevant documents from RAG]
[User question]
"""

"""
Gemini 2.5 Pro; ~5 req/min; 25-100 req/day; 1 million token context window; Complex reasoning; coding; prototyping

Gemini 2.5 Flash; 10 req/min; ~250 req/day; unspecified context window but prob smaller maybe about 100,000?; General purpose; balanced speed

Gemini 2.5 Flash-Lite; 15 req/min; 1,000 req/day; ~250,000 token throughput per min; unspecified context window but prob smaller maybe about 100,000?; Fast, simple, high-frequency calls
"""

# ==============================
# 1. Load API Key
# ==============================
os.environ["GOOGLE_API_KEY"] = "SOME_GOOGLE_API_KEY"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# ==============================
# 2. Load documents from files
# ==============================
DOCS_DIR = Path("./docs")


def load_documents(doc_dir: Path) -> List[str]:
    texts = []
    for file_path in doc_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


documents = load_documents(DOCS_DIR)

# ==============================
# 3. Split documents into chunks
# ==============================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_chunks = []
for doc in documents:
    doc_chunks.extend(splitter.split_text(doc))

# ==============================
# 4. Embed document chunks for RAG
# ==============================
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
doc_vectorstore = FAISS.from_texts(doc_chunks, embeddings)

# ==============================
# 5. Embed conversation memory
# ==============================
memory_vectorstore = FAISS(embeddings.dimension)  # empty initially


def store_memory(text: str):
    """Add conversation text to memory store."""
    vector = embeddings.embed_query(text)
    memory_vectorstore.add_texts([text], [vector])


def retrieve_memory(query: str, k: int = 3):
    """Retrieve relevant past conversation."""
    if memory_vectorstore.index is None:
        return []
    results = memory_vectorstore.similarity_search(query, k=k)
    return [res.page_content for res in results]


# ==============================
# 6. LangChain + Gemini setup
# ==============================
template = """
You are a helpful assistant. Use the following context to answer the question.

Context from documents:
{doc_context}

Context from conversation history:
{memory_context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["doc_context", "memory_context", "question"], template=template
)


class GeminiLangChainLLM(GoogleGemini):
    def __init__(self, model_name="gemini-2.5-flash"):
        super().__init__(model_name=model_name, client=client)


llm = GeminiLangChainLLM()
chain = LLMChain(llm=llm, prompt=prompt)

# ==============================
# 7. Chat loop
# ==============================
if __name__ == "__main__":
    print("Welcome to the Gemini RAG + Memory Chatbot! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Retrieve top 3 relevant document chunks
        doc_context_chunks = doc_vectorstore.similarity_search(query, k=3)
        doc_context = "\n\n".join([c.page_content for c in doc_context_chunks])

        # Retrieve top 3 relevant memory chunks
        memory_context_chunks = retrieve_memory(query, k=3)
        memory_context = "\n\n".join(memory_context_chunks)

        # Run LLM chain
        answer = chain.run(
            {
                "doc_context": doc_context,
                "memory_context": memory_context,
                "question": query,
            }
        )

        # Store this turn in memory
        store_memory(f"You: {query}\nAssistant: {answer}")

        print(f"Assistant: {answer}")
