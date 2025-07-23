from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

loader = TextLoader("quran_sample.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
texts = splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embedding, persist_directory="./quran_db")
db.persist()

print("✅ Quran embedded!")
