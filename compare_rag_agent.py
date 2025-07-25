import os
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient

# Map collection names to friendly names
SCRIPTURES = {
    "gita": "gita",
    "bible": "bible",
    "quran": "quran"
}

# Get Qdrant credentials
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Set up embedding and chat models
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_context_and_answer(collection_name, question, age=None, country=None, sex=None):
    try:
        vectorstore = Qdrant(client=client, collection_name=collection_name, embeddings=embedding)
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        raise RuntimeError(f"❌ Qdrant similarity search failed for collection '{collection_name}': {str(e)}")

    if not docs:
        raise ValueError(f"❌ No documents found in collection '{collection_name}'")

    prompt = PromptTemplate.from_template("""
You are a wise and thoughtful scholar. Based on the excerpts below from a sacred text, answer the question truthfully and respectfully.

The person asking is:
- Age: {age}
- Country: {country}
- Sex: {sex}

Excerpts:
{context}

Question:
{question}

Answer:
""")

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "age": age or "Unknown",
        "country": country or "Unknown",
        "sex": sex or "Prefer not to say"
    })

    return context, response.content

def run_filter_agent(context, answer, question):
    validation_prompt = PromptTemplate.from_template("""
Given the question:
{question}

And the following answer:
{answer}

Check if the answer is:
- Relevant to the question
- Respectful and non-judgmental
- Consistent with the context below

Context:
{context}

Reply with a short verdict like:
✅ Valid and helpful  
⚠️ Relevant but not fully supported  
❌ Off-topic or inappropriate
""")
    chain = validation_prompt | llm
    result = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context
    })
    return result.content
