"""Hallucination tab: shade each generated token by how confident the model was in it."""
import math

import pandas as pd
import streamlit as st

from . import ui_common
from .config import timeout_seconds
from .confidence import SHAKY, SURE, band, prob, summarize
from .llm import LLMError, LLMTimeout, complete
from .render import escape_md

MODEL_KEY = "hall_model"
RESULT_KEY = "hall_result"
FORCE_SYSTEM = ("You are a confident trivia assistant. Always give one specific answer. "
                "Never say you don't know or can't verify.")
PRESETS = {
    "Well-known fact": "What is the capital of Australia?",
    "Made-up person": "Who was the 14th mayor of Springfield, Oregon?",
    "Made-up number": "What was the population of Little Wenlock, Shropshire in the 1851 census?",
    "Custom": "",
}
BAND_COLOR = {"sure": "green", "unsure": "orange", "shaky": "red"}


def _pick_preset():
    q = PRESETS[st.session_state["hall_preset"]]
    if q:
        st.session_state["hall_question"] = q


def render():
    st.markdown("A model doesn't look facts up. It predicts the next token, and each prediction has a "
                "probability. Below, every word is shaded by how sure the model was when it picked it. "
                "**Made-up details tend to show up as low-confidence words**, but the model sounds "
                "just as certain either way.")
    picked = ui_common.model_picker(MODEL_KEY)
    if not picked:
        return
    backend, model, models = picked

    st.session_state.setdefault("hall_preset", "Well-known fact")
    st.session_state.setdefault("hall_question", PRESETS["Well-known fact"])
    st.segmented_control("Try a question about", list(PRESETS), key="hall_preset",
                         on_change=_pick_preset, selection_mode="single")
    with st.form("hall_form", border=False):
        question = st.text_area("Question", key="hall_question", height=80)
        force = st.checkbox("Pressure the model to always answer (never say \"I don't know\")", value=True,
                            help="Well-trained models usually admit they don't know. This instruction "
                                 "removes that escape hatch, like a confident-sounding product prompt might.")
        go = st.form_submit_button("Ask", type="primary", icon=":material/help:")

    if go and question.strip():
        ui_common.clear_timeout(MODEL_KEY)
        try:
            with st.spinner(f"Asking {model}…"):
                r = complete(backend, model, question + " Answer in one short sentence.",
                             system=FORCE_SYSTEM if force else None, temperature=0, max_tokens=100,
                             logprobs=True, top_logprobs=3, timeout=timeout_seconds(), available=models)
            st.session_state[RESULT_KEY] = {"r": r, "q": question}
        except LLMTimeout as e:
            ui_common.record_timeout(MODEL_KEY, e)
        except LLMError as e:
            st.error(str(e), icon=":material/error:")
    ui_common.show_timeout(MODEL_KEY)

    res = st.session_state.get(RESULT_KEY)
    if not res:
        st.info("Press Ask to see the answer with confidence shading.")
        return
    r = res["r"]
    if not r.text.strip():
        st.warning("The model used its whole token budget on hidden reasoning and gave no answer, so there is "
                   "nothing to shade. Pick a non-reasoning model (for NIM: the first in the list).",
                   icon=":material/psychology:")
        return
    s = summarize(r.tokens)
    if not s:
        st.info("This model didn't return token probabilities (or answered with nothing).")
        st.write(r.text)
        return

    st.subheader("Answer, shaded by confidence")
    chips = " ".join(f":{BAND_COLOR[band(prob(t))]}-background[{escape_md(t.token.strip())}]"
                     for t in r.tokens if t.token.strip())
    st.markdown(chips)
    st.caption(f":green-background[sure ≥ {SURE:.0%}]  :orange-background[unsure {SHAKY:.0%}–{SURE:.0%}]  "
               f":red-background[shaky < {SHAKY:.0%}]  ·  greedy decoding (temperature 0)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Average confidence", f"{s.mean:.0%}")
    m2.metric("Shakiest word", f"{s.lowest_token.strip() or '(space)'}", f"{s.lowest_p:.0%} sure", delta_color="off")
    m3.metric("Shaky words", f"{s.n_shaky} of {s.n_tokens}")

    with st.expander("What else could it have said? (the 3 least confident words)"):
        worst = sorted(range(len(r.tokens)), key=lambda i: prob(r.tokens[i]))[:3]
        rows = [{"word": r.tokens[i].token.strip() or "(space)", "chosen": f"{prob(r.tokens[i]):.0%}",
                 "alternatives": ", ".join(f"{w.strip() or '(space)'} {math.exp(lp):.0%}"
                                           for w, lp in r.tokens[i].top if w != r.tokens[i].token)}
                for i in sorted(worst)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Least confident words and their alternatives")
    st.caption("Confidence is not truth: a model can be confidently wrong, and a low-confidence word can still "
               "be right. But low confidence on names, dates and numbers is a good reason to double-check.")
