import streamlit as st

from rag_graph import MAX_RETRIES, build_app, build_retriever

DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

TEST_QUESTIONS = [
    "Explain how the different types of agent memory work?",
    "Explain how chain of thought prompting works?",
]

st.set_page_config(page_title="Chat with your PDF", page_icon=":material/chat:")
st.title("Chat with your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []


def source_label(d):
    src = d.metadata.get("source", "unknown")
    if "page" in d.metadata:  # PDF chunk
        return f"{src}, page {d.metadata['page'] + 1}"
    return d.metadata.get("title") or src  # web page


@st.cache_resource(max_entries=3, show_spinner="Indexing sources…")
def get_graph(pdf_data, urls):
    retriever, failed = build_retriever(pdf_data, urls)
    return build_app(retriever), failed


# ---- Sidebar: sources (URLs are pre-filled for every new session)
with st.sidebar:
    pdfs = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
    urls_text = st.text_area(
        "Web pages (one URL per line)",
        value="\n".join(DEFAULT_URLS),
        height=180,
    )
    if st.button("Clear chat"):
        st.session_state.messages = []

urls = tuple(u.strip() for u in urls_text.splitlines() if u.strip())
pdf_data = tuple((f.name, f.getvalue()) for f in pdfs)

# ---- Input right below the title (inside a container so it is not pinned to the bottom)
with st.container():
    typed = st.chat_input("Ask about your sources")
    cols = st.columns(len(TEST_QUESTIONS))
    clicked = None
    for col, q in zip(cols, TEST_QUESTIONS):
        if col.button(q, key=f"test_{q}", width="stretch"):
            clicked = q

question = typed or clicked

if not pdf_data and not urls:
    st.info("Upload a PDF or add a URL in the sidebar to start.")
    st.stop()

try:
    graph, failed_urls = get_graph(pdf_data, urls)
except ValueError as e:
    st.error(str(e))
    st.stop()

for u in failed_urls:
    st.sidebar.warning(f"Could not load: {u}")

# ---- Conversation (below the input)
chat_area = st.container()
with chat_area:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            final = {}
            with st.status("Thinking…", expanded=False) as status:
                for step in graph.stream({"question": question}, stream_mode="values"):
                    final = step
                    status.update(label=f"Working… (generation attempts: {step.get('retries', 0)})")
                status.update(label="Done", state="complete")

            answer = final.get("generation") or "I couldn't find an answer in the sources."
            st.markdown(answer)
            if final.get("retries", 0) >= MAX_RETRIES:
                st.warning("Retry limit reached; this answer may not be fully supported by the sources.")
            docs = final.get("documents", [])
            if docs:
                with st.expander(f"Sources ({len(docs)})"):
                    for d in docs:
                        st.caption(source_label(d))
                        st.write(d.page_content[:300] + "…")

        st.session_state.messages.append({"role": "assistant", "content": answer})
