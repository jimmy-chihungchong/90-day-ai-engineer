"""Module 4: send one prompt to both models in parallel; each keeps its own history."""
from concurrent.futures import ThreadPoolExecutor

from .llm_client import LLMError, LLMTimeout, chat


def ask(backend, conv, model, prompt, timeout):
    """Add the user turn, call the model, record the reply. Returns (Result|None, error|None)."""
    conv.add(model, "user", prompt)
    try:
        r = chat(backend, model, conv.for_llm(model), timeout)
    except (LLMTimeout, LLMError) as e:
        conv.get(model).pop()  # don't leave a dangling user turn
        return None, str(e)
    conv.add(model, "assistant", r.text, latency=r.latency)
    return r, None


def compare(backend, conv, prompt, timeout):
    """Returns {model: (Result|None, error|None)}.

    LM Studio returns 500 on simultaneous requests, so local runs sequentially;
    NIM is a hosted API and runs in parallel.
    """
    if backend.name == "local":
        return {m: ask(backend, conv, m, prompt, timeout) for m in backend.models}
    with ThreadPoolExecutor(max_workers=len(backend.models)) as ex:
        futs = {m: ex.submit(ask, backend, conv, m, prompt, timeout) for m in backend.models}
        return {m: f.result() for m, f in futs.items()}
