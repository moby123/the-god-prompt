import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load OpenAI key from Replit Secrets
openai_key = os.getenv("openai")
if not openai_key:
    raise ValueError(
        "❌ OpenAI API key not found. Add it in Secrets as 'openai'.")

# Load vector store (Gita-only)
embedding = OpenAIEmbeddings(openai_api_key=openai_key)
vectordb = Chroma(persist_directory="./gita_db", embedding_function=embedding)

# GPT Model
llm = ChatOpenAI(model_name="gpt-4",
                 temperature=0.3,
                 openai_api_key=openai_key)

# Strict + Flexible Prompt Template
prompt_template = """
You are a spiritual assistant grounded in the teachings of the Bhagavad Gita.

A user has asked: "{question}"

Below are selected verses from the Gita that relate to this question.
You must base your response on these verses.

- You may offer clarifications or simple explanations, but do not introduce ideas that are not present in the Gita.
- Always cite the verses when relevant.
- Begin your answer with: "According to the Bhagavad Gita..."
- If the verses do not clearly answer the question, say: "The Bhagavad Gita does not directly address this question."

{context}
"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT})

# Interactive Q&A Loop
while True:
    user_question = input("\n🧘 Ask the Gita (or type 'exit'): ")
    if user_question.lower() == "exit":
        break

    # Get answer from LLM
    response = qa_chain.run(user_question)

    # Get matching verses
    docs = vectordb.similarity_search(user_question, k=5)

    print("\n📜 Gita’s response:")
    print(response)

    print("\n🔍 Based on the following verses:\n")
    for doc in docs:
        print(f"➡️ {doc.page_content.strip()}\n")
