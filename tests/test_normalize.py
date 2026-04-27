"""Phase-2 tests: query normalization and hashing."""

from __future__ import annotations

import hashlib

import pytest

from mneme._normalize import hash_query, normalize

# --- Whitespace ---


def test_normalize_strips_leading_and_trailing_whitespace():
    assert normalize("  hello  ") == "hello"


def test_normalize_collapses_internal_runs_of_whitespace():
    assert normalize("hello   world") == "hello world"


def test_normalize_collapses_mixed_whitespace_classes():
    # tab, newline, carriage return, form feed, vertical tab, regular space
    assert normalize("a\tb\nc\rd\fe\vf g") == "a b c d e f g"


def test_normalize_collapses_runs_of_mixed_whitespace():
    assert normalize("hello \t \n world") == "hello world"


def test_normalize_handles_only_whitespace_input():
    assert normalize("   \t\n   ") == ""


def test_normalize_handles_empty_string():
    assert normalize("") == ""


# --- Casefold ---


def test_normalize_casefolds_by_default():
    assert normalize("Hello World") == "hello world"


def test_normalize_casefold_off_preserves_case():
    assert normalize("Hello World", casefold=False) == "Hello World"


def test_normalize_casefold_handles_german_eszett():
    # German eszett (U+00DF) casefolds to "ss" — stronger than .lower().
    assert normalize("STRAßE") == "strasse"


def test_normalize_casefold_idempotent():
    once = normalize("MiXeD CaSe")
    twice = normalize(once)
    assert once == twice


# --- Order of operations ---


def test_normalize_strips_before_casefolding():
    assert normalize("  HELLO  ") == "hello"


def test_normalize_collapse_runs_after_strip():
    assert normalize("  a   b  ") == "a b"


# --- Unicode ---


def test_normalize_preserves_non_latin_scripts():
    assert normalize("Καλημέρα") == "καλημέρα"


def test_normalize_preserves_emoji():
    assert normalize("hello 👋 world") == "hello 👋 world"


def test_normalize_preserves_combining_characters():
    """NFC vs NFD: normalize() does not canonicalize Unicode forms."""
    nfc = "café"  # single precomposed 'é'
    nfd = "café"  # 'e' + combining acute
    assert normalize(nfc) != normalize(nfd)
    assert normalize(normalize(nfc)) == normalize(nfc)
    assert normalize(normalize(nfd)) == normalize(nfd)


def test_normalize_zero_width_space_left_intact():
    """ZWSP (U+200B) is not whitespace per the regex \\s class."""
    zwsp = chr(0x200B)
    s = f"hi{zwsp}there"
    assert normalize(s) == s.casefold()


def test_normalize_unicode_whitespace_collapsed():
    """Python regex \\s against ``str`` patterns matches Unicode whitespace by default.
    Documented contract: NBSP, em space, line separator are collapsed."""
    nbsp = chr(0x00A0)
    em_space = chr(0x2003)
    line_sep = chr(0x2028)
    assert normalize(f"a{nbsp}{nbsp}b") == "a b"
    assert normalize(f"a{em_space}{em_space}b") == "a b"
    assert normalize(f"a{line_sep}{line_sep}b") == "a b"


def test_normalize_rtl_text_unaffected():
    assert normalize("שלום עולם") == "שלום עולם"


# --- Hashing ---


def test_hash_query_returns_64_char_hex():
    h = hash_query("anything")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_query_stable_across_calls():
    assert hash_query("hello") == hash_query("hello")


def test_hash_query_distinct_for_distinct_inputs():
    assert hash_query("hello") != hash_query("Hello")
    assert hash_query("hello") != hash_query("hello ")
    assert hash_query("") != hash_query(" ")


def test_hash_query_matches_sha256_utf8():
    expected = hashlib.sha256(b"hello").hexdigest()
    assert hash_query("hello") == expected


def test_hash_query_handles_unicode_bytes_correctly():
    assert hash_query("café") != hash_query("cafe")
    assert hash_query("café") == hashlib.sha256("café".encode()).hexdigest()


def test_hash_query_empty_string_well_defined():
    expected = hashlib.sha256(b"").hexdigest()
    assert hash_query("") == expected


# --- Combined: same-after-normalize -> same hash ---


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Hello", "hello"),
        (" hello world ", "hello world"),
        ("HELLO  WORLD", "hello world"),
        ("hello\tworld", "hello world"),
    ],
)
def test_normalize_collisions_drive_hash_collisions(a, b):
    assert hash_query(normalize(a)) == hash_query(normalize(b))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("hello", "hello!"),
        ("hello", "Hello"),
        ("café", "cafe"),
    ],
)
def test_hashes_differ_when_normalized_differ(a, b):
    if normalize(a) == normalize(b):
        pytest.skip("inputs collide under normalize; tested elsewhere")
    assert hash_query(normalize(a)) != hash_query(normalize(b))
