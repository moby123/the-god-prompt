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

# Check API key availability
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai")
if not api_key:
    st.error(
        "❌ OPENAI API key not found. Please set it in Streamlit secrets as 'OPENAI_API_KEY' or 'openai'."
    )
    st.stop()

# Debug: Show that API key is loaded
# st.success(f"✅ API Key Loaded: {bool(api_key)}")

# Input section (using text_input instead of chat_input)
question = st.text_input("❓ Enter your ethical or spiritual question:")

if question:
    st.markdown(f"### 🙋 Your Question:\n> *{question}*")

    try:
        with st.spinner("🔍 Seeking divine wisdom..."):
            answers = {}
            for name, db_path in SCRIPTURES.items():
                context, answer = get_context_and_answer(db_path, question)
                verdict = run_filter_agent(context, answer, question)
                answers[name] = {"answer": answer, "verdict": verdict}

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
