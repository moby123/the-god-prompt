import os
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Qdrant setup
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize embedding model
embedding = OpenAIEmbeddings()

# Connect to Qdrant
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# File paths
FILES = {
    "gita": "gita_sample.txt",
    "bible": "bible_sample.txt",
    "quran": "quran_sample.txt"
}

# Chunking setup
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for collection, filepath in FILES.items():
    print(f"📖 Processing: {collection}")

    # Delete collection if exists
    if client.collection_exists(collection):
        client.delete_collection(collection)

    # Recreate the collection
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # Read file content
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # ✅ Upload documents using correct constructor
    vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embedding,
        url=QDRANT_URL,
        prefer_grpc=False,
        api_key=QDRANT_API_KEY,
        collection_name=collection,
    )

    print(f"✅ Uploaded {len(docs)} chunks to collection '{collection}'")
