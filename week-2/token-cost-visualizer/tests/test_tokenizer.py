from tokviz.tokenizer import count_tokens, tokenize


def test_roundtrip_ascii():
    toks = tokenize("Hello, world!")
    assert "".join(t.text for t in toks) == "Hello, world!"
    assert len(toks) == count_tokens("Hello, world!")


def test_empty():
    assert tokenize("") == [] and count_tokens("") == 0


def test_emoji_and_cjk_do_not_crash_and_use_more_tokens():
    assert count_tokens("你好世界🙂") > 0
    assert count_tokens("internationalization") >= 1


def test_special_token_text_is_plain_text():
    assert count_tokens("<|endoftext|>") > 1


def test_other_encoding():
    assert count_tokens("Hello, world!", "cl100k_base") == 4
