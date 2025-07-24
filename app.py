import streamlit as st
import os
from compare_rag_agent import SCRIPTURES, get_context_and_answer, run_filter_agent

# 🔑 API Key check
openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai")
if not openai_key:
    st.error("❌ OPENAI API key not found. Please set it in Streamlit Secrets as 'OPENAI_API_KEY'.")
    st.stop()

# 🧾 Page setup
st.set_page_config(page_title="The God Prompt", layout="centered")

# 📌 Header
st.title("🕊️ The God Prompt")
st.markdown("""
Ask your ethical or spiritual question and receive wisdom drawn from multiple scriptures.
Compare insights across the **Gita**, **Bible**, and **Quran**, or evaluate how closely each response reflects the original scripture.
""")

# 📋 Demographic inputs
st.sidebar.header("👤 About You (Optional)")
age = st.sidebar.number_input("Age", min_value=5, max_value=100, step=1)
country = st.sidebar.text_input("Country")
sex = st.sidebar.selectbox("Sex", ["Prefer not to say", "Male", "Female", "Other"])

# 🔍 Question input
question = st.text_area("❓ Enter your ethical or spiritual question:")

# 🔄 Submit button
if st.button("Get Wisdom"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown(f"### 🙋 Your Question:\n> *{question.strip()}*")

        with st.spinner("Seeking divine wisdom..."):
            answers = {}
            for name, db_path in SCRIPTURES.items():
                context, answer = get_context_and_answer(db_path, question, age, country, sex)
                verdict = run_filter_agent(context, answer, question)
                answers[name] = {"answer": answer, "verdict": verdict}

        st.markdown("## 📖 Responses from Scriptures")
        for scripture, result in answers.items():
            st.subheader(f"📜 {scripture.capitalize()}")
            st.markdown(f"**Answer:** {result['answer']}")
            st.markdown(f"**Validation:** {result['verdict']}")
