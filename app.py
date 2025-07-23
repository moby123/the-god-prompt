import streamlit as st
from compare_rag_agent import SCRIPTURES, get_context_and_answer, run_filter_agent

st.set_page_config(page_title="The God Prompt", layout="centered")

st.title("🌟 The God Prompt")
st.markdown(
    "**Ask your deepest spiritual, ethical, or social questions.**\nAnswers are drawn from different scriptures without bias."
)

# 📝 User Input
question = st.text_area("What is your question?", height=100)
anonymize = st.checkbox("Anonymize scripture names", value=True)
submit = st.button("🙏 Ask the Scriptures")

if submit and question.strip():
    with st.spinner("Consulting the scriptures..."):
        results = {}
        for name, path in SCRIPTURES.items():
            context, answer = get_context_and_answer(path, question)
            verdict = run_filter_agent(context, answer, question)
            results[name] = {"answer": answer, "verdict": verdict}

        # Shuffle if anonymized
        display_keys = list(results.keys())
        if anonymize:
            import random
            random.shuffle(display_keys)

        st.subheader("📖 Responses")
        for idx, key in enumerate(display_keys, 1):
            label = f"Response {idx}" if anonymize else key
            with st.expander(f"🔹 {label}"):
                st.markdown(f"**Answer**: {results[key]['answer']}")
                st.markdown(f"**Check**: {results[key]['verdict']}")
else:
    st.info("👆 Enter a question and press 'Ask the Scriptures'")
