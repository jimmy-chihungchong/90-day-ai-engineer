"""Embeddings tab: sentences become vectors; similar meaning = nearby. Map + search by meaning."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .config import get_backend, timeout_seconds
from .ui_common import current_backend
from .embeddings import embed, project_2d, rank_by_similarity
from .llm import LLMError, LLMTimeout

DEFAULT_TEXT = """\
animals | The cat sat on the sofa.
animals | A kitten is sleeping on the couch.
animals | The dog chased the ball across the park.
animals | A puppy played fetch in the garden.
money | The stock market fell sharply today.
money | Shares dropped as investors panicked.
money | The bank raised interest rates again.
money | Mortgage payments are getting more expensive.
space | Rockets launch satellites into orbit.
space | The astronaut floated outside the space station.
space | A telescope spotted a distant galaxy.
space | The probe landed on the surface of Mars.
food | She baked fresh bread this morning.
food | The soup needs a little more salt.
food | He grilled vegetables for dinner.
food | The pasta was cooked al dente."""
SEARCH_EXAMPLE = "a pet resting at home"
RESULT_KEY = "emb_result"
SEARCH_KEY = "emb_search"


def parse_lines(raw: str) -> list[tuple[str, str]]:
    """'topic | sentence' per line (topic optional). Blank lines are skipped."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        topic, sep, text = line.partition("|")
        out.append((topic.strip() or "other", text.strip()) if sep and text.strip() else ("other", line))
    return out


@st.cache_data(ttl=600, show_spinner=False, max_entries=20)
def _embed_cached(backend_name: str, texts: tuple[str, ...], kind: str = "passage") -> np.ndarray:
    return embed(get_backend(backend_name), list(texts), timeout=timeout_seconds(), kind=kind)


def _short(s: str, n: int = 30) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def render():
    backend = current_backend()
    st.markdown("An **embedding** turns a piece of text into a long list of numbers that captures its "
                "*meaning*. Sentences about similar things get similar numbers, so they land close together. "
                "That is how search by meaning works, and how an AI can find relevant text without "
                "matching exact words.")
    if not backend.embed_model:
        st.warning(f"No embedding model is set for the {backend.name} backend, so this tab is unavailable.")
        return

    with st.form("emb_form", border=False):
        raw = st.text_area("Sentences (one per line; optional `topic | sentence` to colour the map)",
                           value=DEFAULT_TEXT, height=260)
        go = st.form_submit_button("Embed and map", type="primary", icon=":material/scatter_plot:")

    if go:
        rows = parse_lines(raw)
        if len(rows) < 3:
            st.error("Enter at least 3 sentences.")
        else:
            try:
                with st.spinner(f"Embedding {len(rows)} sentences with {backend.embed_model}…"):
                    vecs = _embed_cached(backend.name, tuple(t for _, t in rows))
                st.session_state[RESULT_KEY] = {"rows": rows, "vecs": vecs}
                st.session_state.pop(SEARCH_KEY, None)
            except LLMTimeout as e:
                st.warning(f"{e} (embedding model: {backend.embed_model})", icon=":material/hourglass_top:")
            except LLMError as e:
                st.error(str(e), icon=":material/error:")

    res = st.session_state.get(RESULT_KEY)
    if not res:
        st.info("Press Embed and map to see your sentences as points.")
        return
    rows, vecs = res["rows"], res["vecs"]

    st.subheader("Map of meaning")
    xy = project_2d(vecs)
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "topic": [t for t, _ in rows],
                       "sentence": [s for _, s in rows], "label": [_short(s) for _, s in rows]})
    base = alt.Chart(df).encode(x=alt.X("x", axis=None), y=alt.Y("y", axis=None),
                                color=alt.Color("topic:N", legend=alt.Legend(title="Topic"),
                                                scale=alt.Scale(scheme="category10")),
                                tooltip=["sentence", "topic"])
    chart = (base.mark_circle(size=140) +
             base.mark_text(align="left", dx=9, fontSize=11).encode(text="label")).properties(height=460)
    st.altair_chart(chart, alt=f"2D map of {len(rows)} sentences; similar meanings are close together")
    st.caption(f"Each sentence became a list of **{vecs.shape[1]} numbers**. This map squashes those down to 2 "
               "so you can see them, so distances are approximate. Hover a point for the full sentence.")

    with st.expander("See the raw numbers for the first sentence"):
        st.write(f"*{rows[0][1]}*")
        st.code("[" + ", ".join(f"{x:.3f}" for x in vecs[0][:12]) + f", … {vecs.shape[1] - 12} more]")
        st.caption("Individual numbers mean nothing on their own. Only the overall pattern, "
                   "compared with other sentences, carries meaning.")

    st.subheader("Search by meaning")
    with st.form("emb_search_form", border=False):
        query = st.text_input("Describe something", value=SEARCH_EXAMPLE)
        found = st.form_submit_button("Find closest sentences", icon=":material/search:")
    if found and query.strip():
        try:
            with st.spinner("Embedding your query…"):
                q = _embed_cached(backend.name, (query.strip(),), "query")[0]
            st.session_state[SEARCH_KEY] = {"q": query.strip(), "ranked": rank_by_similarity(q, vecs)}
        except LLMTimeout as e:
            st.warning(str(e), icon=":material/hourglass_top:")
        except LLMError as e:
            st.error(str(e), icon=":material/error:")
    s = st.session_state.get(SEARCH_KEY)
    if s:
        top = pd.DataFrame([{"sentence": rows[i][1], "topic": rows[i][0], "similarity": sim}
                            for i, sim in s["ranked"][:6]])
        st.dataframe(top, hide_index=True, alt=f"Sentences closest in meaning to: {s['q']}",
                     column_config={"similarity": st.column_config.ProgressColumn(
                         "Similarity", format="%.2f", min_value=0.0, max_value=1.0)})
        st.caption("Notice the top matches share *meaning* with your query, not necessarily any of its words.")
