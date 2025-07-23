import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain

# 🔐 Load API key
openai_key = os.getenv("openai")
if not openai_key:
    raise ValueError("❌ OPENAI API key not found in Replit Secrets")

# 🔎 Vector DB
embedding = OpenAIEmbeddings(openai_api_key=openai_key)
vectordb = Chroma(persist_directory="./gita_db", embedding_function=embedding)

# 🤖 Answer Agent — RetrievalQA
llm_answer = ChatOpenAI(model_name="gpt-4", temperature=0.3, openai_api_key=openai_key)

answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a spiritual assistant answering a user using Bhagavad Gita verses only.

A user asked: "{question}"

Here are relevant verses:
{context}

Respond based on these verses. Do not include ideas not grounded in the scripture. If unsure, say: "The Gita does not clearly address this."
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_answer,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": answer_prompt}
)

# 🧪 Filter Agent — LLMChain
llm_filter = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=openai_key)

filter_prompt = PromptTemplate(
    input_variables=["context", "answer"],
    template="""
You are a scripture verifier. Your job is to check whether the AI-generated answer is faithful to the provided Bhagavad Gita verses.

Verses:
{context}

Answer:
{answer}

Please respond with:
- ✅ FAITHFUL: [short reason]
OR
- ❌ UNFAITHFUL: [explanation of hallucination or deviation]
"""
)

filter_chain = LLMChain(llm=llm_filter, prompt=filter_prompt)

# 🧠 Main orchestration loop
while True:
    user_question = input("\n🧘 Ask the Gita (or type 'exit'): ")
    if user_question.lower() == "exit":
        break

    # Step 1: Get verses
    docs = vectordb.similarity_search(user_question, k=5)
    context_text = "\n".join([doc.page_content for doc in docs])

    # Step 2: Get initial answer
    answer = qa_chain.run(user_question)

    # Step 3: Filter agent verification
    verdict = filter_chain.run({
        "context": context_text,
        "answer": answer
    })

    # Step 4: Display everything
    print("\n📜 Gita’s Response:\n" + answer)
    print("\n🔍 Retrieved Verses:")
    for doc in docs:
        print(f"➡️ {doc.page_content.strip()}\n")

    print("\n🧪 Filter Agent Verdict:\n" + verdict)
