import pytest
"""Headless UI tests (AppTest). Live LLM calls are only made if LM Studio is running."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parent.parent / "app.py"


def _app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_tokenize_button_applies_text():
    at = _app()
    assert not at.exception
    at.text_area[0].set_value("Hello world, hello tokens!")
    assert at.metric[0].value == "20"          # unchanged until the button is pressed
    at.button[0].click().run()
    assert at.metric[0].value == "6"


def test_slider_changes_cost_banner():
    at = _app()
    before = at.info[0].value
    at.slider[0].set_value(4000).run()
    assert at.info[0].value != before and not at.exception


def test_temperature_tab_renders_without_server_crash():
    at = _app()
    assert not at.exception          # either shows the model picker or a friendly error


def _live():
    from tokviz.config import get_backend
    from tokviz.llm import LLMError, list_models
    try:
        return list_models(get_backend("local"), timeout=2)
    except LLMError:
        return None


@pytest.mark.skipif(_live() is None, reason="LM Studio not running")
def test_temperature_generate_live():
    at = _app()
    charts_before = len(at.get("vega_lite_chart"))
    next(b for b in at.button if b.label == "Generate").click().run()
    assert not at.exception and not at.error
    assert any(m.value.startswith("**Temperature") for m in at.markdown)
    assert len(at.get("vega_lite_chart")) == charts_before + 1   # the probability chart appeared


@pytest.mark.skipif(_live() is None, reason="LM Studio not running")
def test_hallucination_ask_live():
    at = _app()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert "Average confidence" in labels and "Shaky words" in labels


@pytest.mark.skipif(_live() is None, reason="LM Studio not running")
def test_embeddings_map_and_search_live():
    at = _app()
    next(b for b in at.button if b.label == "Embed and map").click().run()
    assert not at.exception and not at.error
    next(b for b in at.button if b.label == "Find closest sentences").click().run()
    assert not at.exception and not at.error
    df = at.dataframe[-1].value
    assert df.iloc[0]["topic"] == "animals"      # "a pet resting at home" -> cat/dog sentences


def test_context_tab_cuts_secret_with_tiny_window():
    at = _app()
    assert not at.exception
    # default custom 120-token window with the demo text: the secret code at the start is cut off
    cut = [m for m in at.metric if m.label == "Cut off"][0]
    assert cut.value != "0"
    assert any("don't fit" in w.value for w in at.warning)
    assert any("tangerine" in m.value for m in at.markdown)   # shown in the red "cut off" box
