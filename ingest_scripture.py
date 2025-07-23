import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your OpenAI key from Replit Secrets (you named it "openai")
openai_key = os.getenv("openai")

# Validate key
if not openai_key:
    raise ValueError(
        "❌ OPENAI API key not found. Please set it in Replit Secrets with key = 'openai'"
    )

# Load text from Gita sample file
loader = TextLoader("gita_sample.txt")
documents = loader.load()

# Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                               chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create embeddings and store them in a ChromaDB vector database
embedding = OpenAIEmbeddings(openai_api_key=openai_key)
db = Chroma.from_documents(texts, embedding, persist_directory="./gita_db")
db.persist()

print("✅ Gita embedded and stored successfully!")
