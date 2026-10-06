import pytest

from chatbot.compare import ask, compare
from chatbot.config import get_backend
from chatbot.memory import Conversation


@pytest.mark.live
def test_memory_across_turns():
    b, c = get_backend("local"), Conversation()
    m = b.models[1]
    ask(b, c, m, "My favourite colour is teal. Remember it.", 90)
    r, err = ask(b, c, m, "What is my favourite colour?", 90)
    assert err is None and "teal" in r.text.lower()


@pytest.mark.live
def test_compare_both():
    b, c = get_backend("local"), Conversation()
    out = compare(b, c, "Say hello.", 120)
    for m in b.models:
        assert out[m][1] is None, out[m][1]
        assert len(c.get(m)) == 2
