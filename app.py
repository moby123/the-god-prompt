import streamlit as st
from compare_rag_agent import SCRIPTURES, get_context_and_answer, run_filter_agent
import os

# Set page title and layout
st.set_page_config(page_title="The God Prompt", layout="centered")

# App header
st.title("🕊️ The God Prompt")
st.markdown("""
Ask your ethical or spiritual question and receive wisdom drawn from multiple scriptures.
Choose to view all responses, compare across sources, or get a filtered insight.
""")

# API Key check
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "❌ OPENAI API key not found. Please set it in Replit Secrets or as an environment variable named 'OPENAI_API_KEY'."
    )
    st.stop()

# Input section
question = st.chat_input("Enter your ethical or spiritual question here...")

if question:
    st.markdown(f"### 🙋 Your Question:\n> *{question}*")

    try:
        with st.spinner("Seeking divine wisdom..."):
            answers = {}
            for name, db_path in SCRIPTURES.items():
                context, answer = get_context_and_answer(db_path, question)
                validation = run_filter_agent(context, answer, question)
                answers[name] = {"answer": answer, "verdict": validation}

        if answers:
            st.markdown("## 📖 Responses from Scriptures:")
            for scripture, result in answers.items():
                st.subheader(f"📜 {scripture}")
                st.markdown(f"**Answer:** {result['answer']}")
                st.markdown(f"**Validation:** {result['verdict']}")
        else:
            st.warning(
                "⚠️ No response was generated. Try rephrasing your question.")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
