"""Module 5: Streamlit UI (duck.ai-style)."""
import time

import streamlit as st

from chatbot.compare import compare
from chatbot.config import get_backend, get_timeout
from chatbot.llm_client import LLMError, LLMTimeout, split_thinking, stream_chat
from chatbot.memory import Conversation

st.set_page_config(page_title="Compare Chat", page_icon="💬", layout="centered")
st.markdown("""<style>
.block-container {max-width: 860px; padding-top: 2rem;}
[data-testid="stChatInput"] {border-radius: 24px;}
[data-testid="stChatMessage"] {background: transparent;}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("conv", Conversation())
DEPLOYED_NIM = get_backend().name == "nim"  # BACKEND=nim -> hide the LM Studio option
ss.setdefault("backend_name", get_backend().name)
ss.setdefault("model_idx", 1)  # llama by default (faster)
ss.setdefault("compare", False)
ss.setdefault("notice", None)
ss.setdefault("last_upload", None)


def short(m):
    return m.split("/")[-1]


def render_msg(msg):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            think, answer = split_thinking(msg["content"])
            if think:
                with st.expander("Thinking"):
                    st.markdown(think)
            st.markdown(answer or "_(no final answer)_")
            if "latency" in msg:
                st.caption(f"{msg['latency']}s")
        else:
            st.markdown(msg["content"])


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("💬 Compare Chat")
    if st.button("➕ New chat", use_container_width=True):
        ss.conv.clear()
        ss.notice = None
        st.rerun()

    if DEPLOYED_NIM:
        ss.backend_name = "nim"
        st.caption("Backend: NVIDIA NIM")
    else:
        ss.backend_name = st.radio("Backend", ["local", "nim"], horizontal=True,
                                   index=["local", "nim"].index(ss.backend_name),
                                   format_func=lambda x: "LM Studio" if x == "local" else "NVIDIA NIM")
    backend = get_backend(ss.backend_name)
    if backend.name == "nim" and not backend.api_key:
        st.error("NIM_API_KEY is not set.")

    ss.compare = st.toggle("Compare both models", value=ss.compare)
    if not ss.compare:
        ss.model_idx = st.radio("Model", [0, 1], index=ss.model_idx,
                                format_func=lambda i: short(backend.models[i]))
    st.divider()
    st.subheader("Memory")
    st.download_button("⬇ Save memory (JSON)", ss.conv.to_json(),
                       file_name="chat_memory.json", mime="application/json",
                       use_container_width=True)
    up = st.file_uploader("Load memory", type="json", label_visibility="collapsed")
    if up is not None and ss.last_upload != (up.name, up.size):
        try:
            ss.conv = Conversation.from_json(up.getvalue().decode("utf-8"))
            ss.last_upload = (up.name, up.size)
            st.rerun()
        except Exception as e:
            st.error(f"Could not load: {e}")

backend = get_backend(ss.backend_name)
timeout = get_timeout()
models = list(backend.models) if ss.compare else [backend.models[ss.model_idx]]

# ---------------- history ----------------
if not any(ss.conv.get(m) for m in backend.models):
    st.markdown("<h2 style='text-align:center;margin-top:20vh'>How can I help you today?</h2>",
                unsafe_allow_html=True)

cols = st.columns(len(models)) if len(models) > 1 else [st.container()]
for col, m in zip(cols, models):
    with col:
        if len(models) > 1:
            st.markdown(f"**{short(m)}**")
        for msg in ss.conv.get(m):
            render_msg(msg)

if ss.notice:
    kind, text, switch_to = ss.notice
    st.warning(text) if kind == "timeout" else st.error(text)
    if switch_to is not None and st.button(f"Switch to {short(backend.models[switch_to])}"):
        ss.model_idx, ss.notice = switch_to, None
        st.rerun()

# ---------------- input ----------------
prompt = st.chat_input("Ask anything…")
if prompt:
    ss.notice = None
    if ss.compare:
        with st.spinner("Asking both models (local runs one after the other)…"):
            results = compare(backend, ss.conv, prompt, timeout)
        errs = [e for _, e in results.values() if e]
        if errs:
            ss.notice = ("error", " | ".join(errs), None)
    else:
        m = models[0]
        ss.conv.add(m, "user", prompt)
        with cols[0]:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                box, text, t0 = st.empty(), "", time.time()
                try:
                    for piece in stream_chat(backend, m, ss.conv.for_llm(m), timeout):
                        text += piece
                        box.markdown(text + "▌")
                    ss.conv.add(m, "assistant", text, latency=time.time() - t0)
                except LLMTimeout as e:
                    ss.conv.get(m).pop()
                    other = 1 - ss.model_idx
                    ss.notice = ("timeout", f"⏱ {e} (waited {timeout:.0f}s)", other)
                except LLMError as e:
                    ss.conv.get(m).pop()
                    ss.notice = ("error", str(e), None)
    st.rerun()
