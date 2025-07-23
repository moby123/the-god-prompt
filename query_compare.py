from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import random

# Custom prompt
template = """Answer the question as truthfully as possible using ONLY the provided context (which is from scripture).
If the answer is not in the context, say "Not found in scripture."

Context:
{context}

Question: {question}
"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

def make_chain(db_path):
    vectordb = Chroma(persist_directory=db_path, embedding_function=llm.embedding)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

# Load chains
gita_chain = make_chain("./gita_db")
bible_chain = make_chain("./bible_db")
quran_chain = make_chain("./quran_db")

# Ask question
question = input("❓ Enter your ethical/spiritual question:\n")

# Toggle anonymize
anonymize = input("🔒 Anonymize responses? (y/n): ").strip().lower() == "y"

# Get answers
answers = {
    "Gita": gita_chain.run(question),
    "Bible": bible_chain.run(question),
    "Quran": quran_chain.run(question)
}

# Randomize display if anonymized
labels = list(answers.keys())
if anonymize:
    random.shuffle(labels)
    print("\n📜 Anonymous Responses:")
else:
    print("\n📖 Responses by Scripture:")

# Display
for idx, label in enumerate(labels, 1):
    print(f"\n🔹 Response {idx if anonymize else label}:")
    print(answers[label])
