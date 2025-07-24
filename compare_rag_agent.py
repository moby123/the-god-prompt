import os
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient

# 🌍 Map collection names to friendly display names
SCRIPTURES = {
    "gita": "gita",
    "bible": "bible",
    "quran": "quran"
}

# 🔐 Load Qdrant credentials from environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 💡 Set up OpenAI embeddings and LLM
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# 🔌 Connect to Qdrant
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def get_context_and_answer(collection_name, question):
    """🔍 Search Qdrant collection and generate LLM response from retrieved context."""
    print(f"🔍 Querying collection: {collection_name} for question: {question}")

    # Create LangChain vector store from remote Qdrant collection
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding,
    )

    try:
        docs = vectorstore.similarity_search(question, k=5)
        print(f"📚 Retrieved {len(docs)} documents.")
    except Exception as e:
        raise RuntimeError(f"❌ Qdrant similarity search failed for collection '{collection_name}': {str(e)}")

    if not docs:
        raise ValueError(f"❌ No documents found in collection '{collection_name}' for query: {question}")

    context = "\n\n".join(doc.page_content for doc in docs)

    # 🧠 Prompt Template for Answer Generation
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
    """🧪 Evaluate if answer is respectful, relevant, and scripture-aligned."""
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
