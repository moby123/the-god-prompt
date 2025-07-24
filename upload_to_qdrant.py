import os
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load Qdrant credentials from environment
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Initialize Qdrant Client
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# Embedding model
embedding = OpenAIEmbeddings()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# SCRIPTURES = {collection_name: file_path}
SCRIPTURES = {
    "gita": "gita_sample.txt",
    "bible": "bible_sample.txt",
    "quran": "quran_sample.txt"
}

for collection, file_path in SCRIPTURES.items():
    print(f"📖 Processing: {collection}")

    # Load and split
    loader = TextLoader(file_path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)

    # Create collection if it doesn't exist
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    )

    # Upload to Qdrant
    Qdrant.from_documents(
        docs,
        embedding,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection
    )

    print(f"✅ Uploaded {len(docs)} chunks to collection '{collection}'")
