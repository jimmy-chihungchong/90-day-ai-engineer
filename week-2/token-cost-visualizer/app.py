import pandas as pd
import streamlit as st

from tokviz import ui_common
from tokviz.pricing import estimate, load_models
from tokviz import tab_context, tab_embeddings, tab_hallucination, tab_temperature
from tokviz.render import chips_markdown, visible
from tokviz.tokenizer import ENCODINGS, tokenize

st.set_page_config(page_title="Token & cost visualizer", page_icon=":material/token:", layout="wide")
st.title("Token & cost visualizer")
st.caption("Paste text, see how a model sees it, and what it would cost.")

ui_common.backend_selector()

SAMPLE = "Tokenization is unbelievably fun! Numbers like 1234567 and emoji 🙂 split in surprising ways."
tab_tokens, tab_ctx, tab_temp, tab_hall, tab_emb = st.tabs(
    ["Tokens & cost", "Context window", "Temperature", "Hallucination", "Embeddings"]
)

with tab_tokens:
    with st.form("text_form", border=False):
        text = st.text_area("Your text", value=SAMPLE, height=160)
        st.form_submit_button("Tokenize", type="primary", icon=":material/token:")
    c1, c2 = st.columns(2)
    enc_label = c1.selectbox("Tokenizer to visualize", list(ENCODINGS))
    out_tokens = c2.slider("Expected output length (tokens)", 0, 8000, 500, step=50)
    encoding = ENCODINGS[enc_label][0]
    cost_slot = st.container()  # filled once costs are computed, so it sits right under the slider

    tokens = tokenize(text, encoding)
    chars = len(text)
    m1, m2, m3 = st.columns(3)
    m1.metric("Tokens", f"{len(tokens):,}")
    m2.metric("Characters", f"{chars:,}")
    m3.metric("Chars per token", f"{chars / len(tokens):.2f}" if tokens else "–")

    st.subheader("How the model sees your text")
    if tokens:
        st.markdown(chips_markdown(tokens))
        st.caption("Each colored chip is one token. · = space, ↵ = newline.")
        with st.expander("Token IDs"):
            st.dataframe(
                pd.DataFrame({"token": [visible(t.text) for t in tokens],
                              "id": [t.id for t in tokens],
                              "bytes": [t.n_bytes for t in tokens]}),
                hide_index=True, alt="Token text, id and byte length",
            )
    else:
        st.info("Type something above.")

    st.subheader("Estimated cost across models")
    models, as_of = load_models()
    rows = estimate(text, out_tokens, models)
    df = pd.DataFrame(rows)
    paid = [r for r, m in zip(rows, models) if m.kind == "paid"]
    lo, hi = min(paid, key=lambda r: r["total"]), max(paid, key=lambda r: r["total"])
    cost_slot.info(
        f"**Estimated cost:** $0 on local models · "
        f"${lo['total']:.6f} ({lo['model']}) to ${hi['total']:.6f} ({hi['model']}) "
        f"for {len(tokens):,} input + {out_tokens:,} output tokens",
        icon=":material/payments:",
    )
    df["model"] = [r["model"] if r["exact"] else f"≈ {r['model']}" for r in rows]
    st.dataframe(
        df[["model", "tokens", "input_cost", "output_cost", "total", "context_used_pct"]],
        hide_index=True,
        alt="Token count and cost per model",
        column_config={
            "model": "Model",
            "tokens": st.column_config.NumberColumn("Input tokens"),
            "input_cost": st.column_config.NumberColumn("Input $", format="$%.6f"),
            "output_cost": st.column_config.NumberColumn("Output $", format="$%.6f"),
            "total": st.column_config.NumberColumn("Total $", format="$%.6f"),
            "context_used_pct": st.column_config.ProgressColumn(
                "Context used", format="%.2f%%", min_value=0, max_value=100),
        },
    )
    st.caption(f"≈ = token count approximated with o200k_base (that model's own tokenizer isn't bundled). "
               f"Prices as of {as_of}; edit data/pricing.json. Paid models' context windows are not filled in yet.")

with tab_temp:
    tab_temperature.render()

with tab_hall:
    tab_hallucination.render()

with tab_emb:
    tab_embeddings.render()

with tab_ctx:
    tab_context.render()
