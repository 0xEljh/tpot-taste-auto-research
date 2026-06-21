"""TDD for the factored deopt helper (Phase 6c, D32) — pure, no model."""
from __future__ import annotations

from tpot_taste.data.degrade import PLATITUDE_STYLES, STYLES, clean_gen


def test_clean_gen_strips_quotes_and_whitespace():
    assert clean_gen('  "hello   world"  ') == "hello world"
    assert clean_gen("a normal post.") == "a normal post."


def test_clean_gen_drops_assistant_preamble():
    assert clean_gen("Sure, here is the rewrite: the actual post") == "the actual post"
    assert clean_gen("Okay: do the thing") == "do the thing"


def test_clean_gen_keeps_colon_in_body_when_no_preamble():
    # no leading prefix -> the colon is content, not a preamble boundary
    assert clean_gen("ratio of x:y matters") == "ratio of x:y matters"


def test_platitude_styles_are_subset_and_length_matched():
    assert set(PLATITUDE_STYLES) <= set(STYLES)
    assert all("same length" in s for s in STYLES)  # every degradation holds length (v5/D16)
