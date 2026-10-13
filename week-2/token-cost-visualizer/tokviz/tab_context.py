"""Context window tab: see exactly what fits, and what the model never gets to read."""
import pandas as pd
import streamlit as st

from .context import split_window, text_of
from .pricing import load_models
from .render import escape_md
from .tokenizer import tokenize

CUSTOM = "Custom (tiny demo window)"
KEEP_LABELS = {"Keep the most recent (chat apps)": "recent", "Keep the beginning": "start"}
PREVIEW_CHARS = 2500

DEMO_TEXT = (
    "SECRET CODE: tangerine-42. Remember it, you will be asked about it later.\n\n"
    "Notes from the meeting. We talked about the quarterly roadmap, the new onboarding flow, and how "
    "support tickets are trending. The design team shared three mockups, and engineering raised concerns "
    "about the migration timeline. Marketing wants the launch moved to spring, while finance asked for a "
    "revised budget. Everyone agreed to meet again next week to review the open questions, assign owners, "
    "and decide which features can be cut. Lunch was catered and the coffee was surprisingly good.\n\n"
    "Question: What was the secret code?"
)


def _preview(text: str, n: int = PREVIEW_CHARS) -> str:
    if len(text) <= n:
        return text
    half = n // 2
    return text[:half] + "\n\n[... middle not shown ...]\n\n" + text[-half:]


def render():
    st.markdown("A model can only read so many tokens at once: its **context window**. Your prompt and the "
                "reply must **share** that space. Anything beyond it is not remembered poorly, it is simply "
                "**not seen**. Paste text below, pick a window, and watch what gets cut.")

    models, _ = load_models()
    windows = {m.label: m.context_window for m in models if m.context_window}
    c1, c2, c3 = st.columns(3)
    choice = c1.selectbox("Model's window", list(windows) + [CUSTOM], index=len(windows),
                          help="Real windows are huge, so use the custom demo window to see truncation "
                               "with short text. Paid models' windows aren't filled in yet.")
    if choice == CUSTOM:
        window = c2.number_input("Window size (tokens)", 20, 1_000_000, 120, step=10)
        reserve_default = 20
    else:
        window = windows[choice]
        c2.metric("Window size (tokens)", f"{window:,}")
        reserve_default = 500
    reserve = c3.number_input("Reserve for the reply (tokens)", 0, int(window), min(reserve_default, int(window)), step=10)

    keep_label = st.segmented_control("When it doesn't fit", list(KEEP_LABELS),
                                      default=list(KEEP_LABELS)[0], selection_mode="single")
    keep = KEEP_LABELS[keep_label or list(KEEP_LABELS)[0]]

    with st.form("ctx_form", border=False):
        text = st.text_area("Your text", value=DEMO_TEXT, height=220)
        st.form_submit_button("Check the window", type="primary", icon=":material/fit_screen:")

    tokens = tokenize(text)
    s = split_window(tokens, int(window), int(reserve), keep)
    n, kept = len(tokens), len(s.kept)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Your text", f"{n:,} tokens")
    m2.metric("Room for input", f"{s.budget:,}", help="Window minus the tokens reserved for the reply")
    m3.metric("Fits", f"{kept:,}")
    m4.metric("Cut off", f"{s.n_dropped:,}", delta=f"-{s.n_dropped / n:.0%}" if s.n_dropped else None,
              delta_color="off")

    if s.n_dropped:
        st.warning(f"{s.n_dropped:,} tokens don't fit. The model never sees them, and has no way to know "
                 f"they existed.", icon=":material/content_cut:")
    elif n:
        st.success("Everything fits.", icon=":material/check_circle:")

    unused = max(s.budget - kept, 0)
    bar = pd.DataFrame({"Kept input": [kept], "Reserved for reply": [int(reserve)],
                        "Unused room": [unused], "Cut off": [s.n_dropped]}, index=["tokens"])
    st.bar_chart(bar, horizontal=True, stack=True, height=130, x_label="Tokens", y_label="",
                 color=["#2ca02c", "#1f77b4", "#bdbdbd", "#d62728"],
                 alt="Context window usage: kept input, reply reserve, unused room and cut-off tokens")

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**What the model sees**")
        st.text(_preview(text_of(s.kept)) or "(nothing)")
    with right.container(border=True):
        st.markdown("**Cut off, never seen**")
        cut = text_of(s.dropped_start) + text_of(s.dropped_end)
        if cut:
            flat = _preview(cut).replace("\n", " ")
            st.markdown(f":red-background[{escape_md(flat)}]")
        else:
            st.caption("Nothing was cut.")

    st.caption("Bigger windows let you include more, but they cost more and are slower, because you pay for "
               "every token you send. Even when text fits, models can pay less attention to the middle of "
               "a very long prompt. Token counts here use the o200k tokenizer (exact for gpt-oss, approximate "
               "for other models).")
