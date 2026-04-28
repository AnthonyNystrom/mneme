"""Phase-6 scoring tests: default_confidence and default_validator."""

from __future__ import annotations

import itertools
import math

import pytest

from mneme._scoring import default_confidence, default_validator

# --- default_confidence ---


def test_confidence_at_zero_age_equals_similarity():
    assert default_confidence(1.0, 0, {}) == pytest.approx(1.0)
    assert default_confidence(0.85, 0, {}) == pytest.approx(0.85)
    assert default_confidence(0.0, 0, {}) == pytest.approx(0.0)


def test_confidence_at_one_half_life_halves():
    """24h decay: confidence at 86_400s of age = similarity * 0.5."""
    assert default_confidence(1.0, 86_400, {}) == pytest.approx(0.5)
    assert default_confidence(0.9, 86_400, {}) == pytest.approx(0.45)


def test_confidence_at_two_half_lives_quarters():
    assert default_confidence(1.0, 86_400 * 2, {}) == pytest.approx(0.25)


def test_confidence_negative_age_clamped_to_zero():
    """Defensive: negative age (clock skew) treated as zero."""
    assert default_confidence(1.0, -100, {}) == pytest.approx(1.0)


def test_confidence_metadata_ignored_by_default():
    a = default_confidence(0.9, 1_000, {"k": "v"})
    b = default_confidence(0.9, 1_000, {})
    assert a == b


def test_confidence_decays_monotonically_with_age():
    sims = [default_confidence(0.9, age, {}) for age in (0, 3600, 86_400, 172_800)]
    for a, b in itertools.pairwise(sims):
        assert a > b


def test_confidence_returns_float():
    out = default_confidence(0.9, 100, {})
    assert isinstance(out, float)
    assert math.isfinite(out)


# --- default_validator ---


@pytest.mark.parametrize("response", ["", " ", "   ", "\t", "\n", "\t\n  "])
def test_validator_rejects_empty_or_whitespace(response):
    assert default_validator(response) is False


@pytest.mark.parametrize(
    "response",
    [
        "[LLM Error] timeout",
        "[LLM Error]",
        "[ERROR] something broke",
        "[ERROR]",
    ],
)
def test_validator_rejects_documented_error_prefixes(response):
    assert default_validator(response) is False


@pytest.mark.parametrize(
    "response",
    [
        "hello",
        "  hello  ",
        "Hello, world!",
        '{ "json": true }',
        "[INFO] this is fine",  # not the rejected prefix
        "Error: but no bracket prefix",
    ],
)
def test_validator_accepts_normal_responses(response):
    assert default_validator(response) is True


def test_validator_returns_bool():
    assert default_validator("hello") is True
    assert default_validator("") is False
