"""Temperature & sampling tab: same prompt at several temperatures + how the odds get reshaped."""
import pandas as pd
import streamlit as st

from . import ui_common
from .config import timeout_seconds
from .llm import LLMError, LLMTimeout, complete
from .sampling import reshape, to_probs

DEFAULT_PROMPT = "Finish this sentence with a few words: The best thing about autumn is"
KEY = "temp_results"
MODEL_KEY = "temp_model"


def render():
    st.markdown("**Temperature** controls how adventurous the model is when picking the next token. "
                "**top_p** cuts off the unlikely tail. Run the same prompt at different settings and compare.")
    picked = ui_common.model_picker(MODEL_KEY)
    if not picked:
        return
    backend, model, models = picked
    with st.form("temp_form", border=False):
        prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=80)
        cols = st.columns(4)
        temps = [cols[i].number_input(f"Temperature {i + 1}", 0.0, 2.0, v, step=0.1)
                 for i, v in enumerate([0.0, 0.7, 1.5])]
        top_p = cols[3].slider("top_p", 0.05, 1.0, 1.0, step=0.05)
        go = st.form_submit_button("Generate", type="primary", icon=":material/play_arrow:")

    if go:
        ui_common.clear_timeout(MODEL_KEY)
        outs = []
        try:
            with st.spinner(f"Asking {model}…"):
                for t in temps:
                    outs.append(complete(backend, model, prompt, temperature=t, top_p=top_p,
                                         max_tokens=60, logprobs=True, top_logprobs=8,
                                         timeout=timeout_seconds(), available=models))
            st.session_state[KEY] = {"outs": outs, "temps": temps, "top_p": top_p, "model": model}
        except LLMTimeout as e:
            ui_common.record_timeout(MODEL_KEY, e)
        except LLMError as e:
            st.error(str(e), icon=":material/error:")

    ui_common.show_timeout(MODEL_KEY)


    res = st.session_state.get(KEY)
    if not res:
        st.info("Press Generate to see the outputs.")
        return

    st.subheader("Outputs")
    cols = st.columns(len(res["temps"]))
    for col, t, o in zip(cols, res["temps"], res["outs"]):
        with col.container(border=True):
            st.markdown(f"**Temperature {t:g}**")
            st.write(o.text.strip() or "*(empty - model spent its token budget thinking)*")
            st.caption(f"{o.completion_tokens} tokens · {o.latency:.1f}s")

    first = next((o.tokens[0] for o in res["outs"] if o.tokens and o.tokens[0].top), None)
    st.subheader("What the model was choosing between (first word)")
    if not first:
        st.info("This model didn't return token probabilities, so the chart is unavailable.")
        return
    words = [w.replace(" ", "·").replace("\n", "↵") or "(empty)" for w, _ in first.top]
    base = to_probs([lp for _, lp in first.top])
    df = pd.DataFrame({f"T={t:g}": reshape(base, t, res["top_p"]) for t in res["temps"]},
                      index=pd.Index(words, name="candidate token"))
    st.bar_chart(df, stack=False, sort=False, alt="Next-token probabilities at each temperature", x_label="Candidate first token", y_label="Probability",
                 horizontal=False)
    st.caption("Probabilities come from the model (raw, before temperature); the bars show how each "
               "temperature and top_p would reshape them. Only the top 8 candidates are shown, so they are "
               "renormalized among those. Low temperature piles probability onto one token; high temperature "
               "spreads it out, so unlikely tokens get picked more often.")
