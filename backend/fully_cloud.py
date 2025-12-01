import os
from dotenv import load_dotenv
from langchain.embeddings import VertexAIEmbeddings  # Gemini API for embeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import VectorStoreRetrieverMemory
import pinecone

load_dotenv()

# ==============================
# 1. Set up your Gemini API key (if applicable)
# ==============================
gemini_api_key = os.environ.get("GEMINI_API_KEY")  # Use .get() in case it's missing
# ==============================
# 2. Initialize Pinecone
# ==============================
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
)
index_name_docs = "rag-docs"
index_name_memory = "rag-memory"

# Create Pinecone indexes if they don't exist
for idx_name in [index_name_docs, index_name_memory]:
    if idx_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=idx_name, dimension=768
        )  # Assuming 768-dimensional embeddings

docs_index = pinecone.Index(index_name_docs)
memory_index = pinecone.Index(index_name_memory)

# ==============================
# 3. Load documents from your own storage solution (e.g., local, AWS S3, etc.)
# ==============================
# Example of loading documents locally (replace this with your storage method)
documents = []
file_paths = [
    "path/to/your/doc1.txt",
    "path/to/your/doc2.txt",
]  # Replace with actual paths

for path in file_paths:
    with open(path, "r") as file:
        content = file.read()
        documents.append(content)

# ==============================
# 4. Use Gemini API for embeddings (no Google Cloud needed)
# ==============================
embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001")

# Upsert documents into Pinecone
vectors_to_upsert = []
for i, doc in enumerate(documents):
    vector = embeddings.embed_query(doc)
    vectors_to_upsert.append((str(i), vector, {"text": doc}))

docs_index.upsert(vectors_to_upsert)

# Wrap in LangChain vector store
doc_vectorstore = Pinecone(
    docs_index, embedding_function=embeddings.embed_query, text_key="text"
)

# ==============================
# 5. Set up cloud memory (using Pinecone to store chat history)
# ==============================
memory_vectorstore = Pinecone(
    memory_index, embedding_function=embeddings.embed_query, text_key="text"
)
memory = VectorStoreRetrieverMemory(
    retriever=memory_vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    memory_key="chat_history",
)

# ==============================
# 6. Create Conversational RAG Chain
# ==============================
chat_model = ChatVertexAI(model_name="gemini-2.5-flash")

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=doc_vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    memory=memory,
)

# ==============================
# 7. Example Conversation
# ==============================
query1 = "How does RAG improve Gemini's responses?"
answer1 = conv_chain.run(query1)
print("User:", query1)
print("Bot:", answer1)

# Next query will remember the previous one
query2 = "Can you summarize what we discussed before?"
answer2 = conv_chain.run(query2)
print("User:", query2)
print("Bot:", answer2)
