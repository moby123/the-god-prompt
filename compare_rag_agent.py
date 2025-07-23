import os
import random
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# 🌟 Load LLM and Embeddings
openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai")
if not openai_key:
    raise ValueError(
        "❌ OPENAI API key not found. Please set it in Replit Secrets as 'OPENAI_API_KEY' or 'openai'."
    )

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=openai_key)
embedding = OpenAIEmbeddings(api_key=openai_key)

# 🧠 Custom Answer Prompt with fallback wording
ANSWER_TEMPLATE = """Answer the question using ONLY the provided scripture context.

If the context does not contain the answer, say exactly:
"This question is not answered in this scripture."

Context:
{context}

Question: {question}
"""
answer_prompt = PromptTemplate.from_template(ANSWER_TEMPLATE)

# 🛡️ Filter Agent Prompt with fallback validation
FILTER_TEMPLATE = """You are a scripture guardian. Ensure the answer sticks to the context and does not hallucinate.

Context:
{context}

Answer:
{answer}

Question:
{question}

If the answer matches the context or says "This question is not answered in this scripture.", reply with:
✅ VALID - [brief reason]

If not, reply with:
❌ INVALID - [brief reason]
"""
filter_prompt = ChatPromptTemplate.from_template(FILTER_TEMPLATE)


# 📦 Retrieval + Answer Logic
def get_context_and_answer(db_path, question):
    db = Chroma(persist_directory=db_path, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": 8})  # Increased depth
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 🐛 DEBUG: Print what was retrieved
    print(
        f"\n📚 Context from {db_path}:\n{'-'*40}\n{context[:1000]}...\n{'-'*40}"
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": answer_prompt})
    answer = chain.run(question)
    return context, answer


# 🧠 Filter Agent Check
def run_filter_agent(context, answer, question):
    messages = filter_prompt.format_messages(context=context,
                                             answer=answer,
                                             question=question)
    response = llm(messages).content
    return response


# 📖 Map of Scriptures
SCRIPTURES = {
    "Gita": "./gita_db",
    "Bible": "./bible_db",
    "Quran": "./quran_db"
}

# 👂 Input from User
question = input("❓ Enter your ethical/spiritual question:\n")
anonymize = input("🔒 Anonymize responses? (y/n): ").strip().lower() == "y"

# 🧪 Run for Each Scripture
results = {}
for name, path in SCRIPTURES.items():
    context, answer = get_context_and_answer(path, question)
    verdict = run_filter_agent(context, answer, question)
    results[name] = {"answer": answer, "verdict": verdict}

# 🎭 Shuffle Labels if Anonymous
labels = list(results.keys())
if anonymize:
    random.shuffle(labels)
    print("\n📜 Anonymous Scripture Responses (with Validation):")
else:
    print("\n📖 Scripture Responses (with Validation):")

# 🖥️ Output the Results
for idx, name in enumerate(labels, 1):
    label = f"Response {idx}" if anonymize else name
    print(f"\n🔹 {label}")
    print(f"📝 Answer: {results[name]['answer']}")
    print(f"🧪 Check: {results[name]['verdict']}")
