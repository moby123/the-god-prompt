import os
import requests
import openai
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ------------------- Configuration -------------------
SCRIPTURES = {
    "gita": "gita",
    "bible": "bible",
    "quran": "quran"
}

QDRANT_URL = os.getenv("QDRANT_URL")  # e.g. https://your-qdrant-url
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

HEADERS = {
    "Content-Type": "application/json",
    "api-key": QDRANT_API_KEY
}

openai.api_key = OPENAI_API_KEY
llm = ChatOpenAI(temperature=0)
# ------------------------------------------------------

def get_embedding(text: str):
    """Generate embedding for a given text using OpenAI API."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]

def search_qdrant(collection_name, query_vector, top_k=5):
    """Search similar documents from Qdrant collection using REST API."""
    url = f"{QDRANT_URL}/collections/{collection_name}/points/search"

    payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True
    }

    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code != 200:
        raise RuntimeError(f"❌ REST API Search failed: {response.text}")

    return response.json()["result"]

def get_context_and_answer(collection_name, question):
    """Retrieve context from Qdrant and generate answer using LLM."""
    try:
        query_vector = get_embedding(question)
        search_results = search_qdrant(collection_name, query_vector)
        docs = [hit["payload"]["text"] for hit in search_results if "text" in hit.get("payload", {})]
    except Exception as e:
        raise RuntimeError(f"❌ Qdrant similarity search failed for collection '{collection_name}': {str(e)}")

    if not docs:
        raise ValueError(f"❌ No documents found in collection '{collection_name}'")

    context = "\n\n".join(docs)

    prompt = PromptTemplate.from_template("""
You are a wise and thoughtful scholar. Based on the excerpts below from a sacred text, answer the question truthfully and respectfully.

Excerpts:
{context}

Question:
{question}

Answer:
""")

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return context, response.content

def run_filter_agent(context, answer, question):
    """Validate that the answer is coherent and appropriate."""
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
