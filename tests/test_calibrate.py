"""Phase-11 calibration tests + CLI smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mneme.tools.calibrate import (
    CalibrationResult,
    find_threshold,
    precision_recall_curve,
)
from mneme.tools.calibrate import (
    main as cli_main,
)

from .fakes import ParaphraseEmbedder

# --- Synthetic fixtures ---


def _paraphrases() -> list[tuple[str, str]]:
    """High-overlap pairs (should be positive)."""
    return [
        ("how do i reset my password", "how to reset password"),
        ("what is the refund policy", "what's the policy on refunds"),
        ("how do i delete account", "delete my account please"),
        ("contact customer support", "talk to support"),
        ("when does my subscription expire", "subscription expiration date"),
    ]


def _distractors() -> list[tuple[str, str]]:
    """Unrelated pairs (should be negative)."""
    return [
        ("how do i reset my password", "what is the weather today"),
        ("refund policy", "best pizza topping"),
        ("delete account", "stock market closing time"),
        ("customer support", "favorite color"),
        ("subscription", "earthquake magnitude"),
    ]


# --- find_threshold ---


def test_find_threshold_returns_calibration_result():
    e = ParaphraseEmbedder(dim=32)
    result = find_threshold(_paraphrases(), _distractors(), e)
    assert isinstance(result, CalibrationResult)
    assert 0.0 <= result.threshold <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert len(result.pr_curve) > 0


def test_find_threshold_target_metric_f1_picks_balanced():
    e = ParaphraseEmbedder(dim=32)
    result = find_threshold(_paraphrases(), _distractors(), e, target_metric="f1")
    # F1 of paraphrase-shaped data should be reasonable.
    assert result.f1 >= 0.5


def test_find_threshold_target_precision_prefers_strict():
    e = ParaphraseEmbedder(dim=32)
    p_result = find_threshold(_paraphrases(), _distractors(), e, target_metric="precision")
    f1_result = find_threshold(_paraphrases(), _distractors(), e, target_metric="f1")
    # Precision-optimizing threshold is at least as strict as F1's.
    assert p_result.threshold >= f1_result.threshold or p_result.precision >= f1_result.precision


def test_find_threshold_min_precision_constraint_filters():
    e = ParaphraseEmbedder(dim=32)
    result = find_threshold(
        _paraphrases(), _distractors(), e, min_precision=0.9, target_metric="recall"
    )
    assert result.precision >= 0.9


def test_find_threshold_min_recall_constraint_filters():
    """Use a wide grid (incl. low thresholds) so paraphrase recall is achievable."""
    e = ParaphraseEmbedder(dim=32)
    grid = [round(i * 0.05, 2) for i in range(2, 20)]  # 0.10 .. 0.95
    result = find_threshold(
        _paraphrases(),
        _distractors(),
        e,
        min_recall=0.5,
        target_metric="precision",
        grid=grid,
    )
    assert result.recall >= 0.5


def test_find_threshold_no_solution_raises():
    e = ParaphraseEmbedder(dim=32)
    with pytest.raises(ValueError, match="threshold"):
        find_threshold(
            _paraphrases(),
            _distractors(),
            e,
            min_precision=1.5,  # impossible
        )


def test_find_threshold_empty_paraphrases_raises():
    e = ParaphraseEmbedder(dim=32)
    with pytest.raises(ValueError, match="paraphrase"):
        find_threshold([], _distractors(), e)


def test_find_threshold_empty_distractors_raises():
    e = ParaphraseEmbedder(dim=32)
    with pytest.raises(ValueError, match="distractor"):
        find_threshold(_paraphrases(), [], e)


def test_find_threshold_custom_grid():
    e = ParaphraseEmbedder(dim=32)
    grid = [0.2, 0.5, 0.8]
    result = find_threshold(_paraphrases(), _distractors(), e, grid=grid)
    assert result.threshold in grid
    assert len(result.pr_curve) == 3


# --- vector_dtype ---


def test_calibrate_with_int8_dtype():
    """int8 should produce a valid result; threshold may differ from fp32."""
    e = ParaphraseEmbedder(dim=64)
    find_threshold(_paraphrases(), _distractors(), e, vector_dtype="float32")
    r_int8 = find_threshold(_paraphrases(), _distractors(), e, vector_dtype="int8")
    # Both finish; int8 result is a valid CalibrationResult.
    assert isinstance(r_int8, CalibrationResult)
    assert r_int8.f1 >= 0.0


def test_calibrate_with_float16_dtype():
    e = ParaphraseEmbedder(dim=64)
    result = find_threshold(_paraphrases(), _distractors(), e, vector_dtype="float16")
    assert isinstance(result, CalibrationResult)


# --- precision_recall_curve ---


def test_pr_curve_returns_per_threshold_points():
    e = ParaphraseEmbedder(dim=32)
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    curve = precision_recall_curve(_paraphrases(), _distractors(), e, grid=grid)
    assert len(curve) == 5
    for t, p, r in curve:
        assert t in grid
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0


def test_pr_curve_default_grid_is_50_points():
    e = ParaphraseEmbedder(dim=32)
    curve = precision_recall_curve(_paraphrases(), _distractors(), e)
    assert len(curve) == 50


# --- CLI ---


def _write_jsonl(path: Path, pairs: list[tuple[str, str]]) -> None:
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(json.dumps({"a": a, "b": b}) + "\n")


def test_cli_smoke_human_format(tmp_path: Path, capsys, monkeypatch):
    """Run the CLI with a synthetic fixture; expect exit code 0 and human output."""
    pp = tmp_path / "pp.jsonl"
    dd = tmp_path / "dd.jsonl"
    _write_jsonl(pp, _paraphrases())
    _write_jsonl(dd, _distractors())

    # Stash the embedder factory at a known import path the CLI can resolve.
    import sys as _sys

    fake_mod = type(_sys)("_test_calibrate_embedder")
    fake_mod.embedder = ParaphraseEmbedder(dim=32)  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "_test_calibrate_embedder", fake_mod)

    rc = cli_main(
        [
            "--paraphrases",
            str(pp),
            "--distractors",
            str(dd),
            "--embedder",
            "_test_calibrate_embedder:embedder",
            "--vector-dtype",
            "float32",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Optimal threshold" in out
    assert "Precision" in out
    assert "Recall" in out


def test_cli_json_format_to_file(tmp_path: Path, monkeypatch):
    pp = tmp_path / "pp.jsonl"
    dd = tmp_path / "dd.jsonl"
    out_path = tmp_path / "result.json"
    _write_jsonl(pp, _paraphrases())
    _write_jsonl(dd, _distractors())

    import sys as _sys

    fake_mod = type(_sys)("_test_calibrate_embedder2")
    fake_mod.embedder = ParaphraseEmbedder(dim=32)  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "_test_calibrate_embedder2", fake_mod)

    rc = cli_main(
        [
            "--paraphrases",
            str(pp),
            "--distractors",
            str(dd),
            "--embedder",
            "_test_calibrate_embedder2:embedder",
            "--format",
            "json",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert {"threshold", "precision", "recall", "f1", "pr_curve"} <= set(payload.keys())


def test_cli_csv_format(tmp_path: Path, capsys, monkeypatch):
    pp = tmp_path / "pp.jsonl"
    dd = tmp_path / "dd.jsonl"
    _write_jsonl(pp, _paraphrases())
    _write_jsonl(dd, _distractors())

    import sys as _sys

    fake_mod = type(_sys)("_test_calibrate_embedder3")
    fake_mod.embedder = ParaphraseEmbedder(dim=32)  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "_test_calibrate_embedder3", fake_mod)

    rc = cli_main(
        [
            "--paraphrases",
            str(pp),
            "--distractors",
            str(dd),
            "--embedder",
            "_test_calibrate_embedder3:embedder",
            "--format",
            "csv",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    # CSV header
    assert out.startswith("threshold,precision,recall,f1")


def test_cli_missing_file_exits_2(tmp_path: Path, capsys):
    rc = cli_main(
        [
            "--paraphrases",
            str(tmp_path / "nope.jsonl"),
            "--distractors",
            str(tmp_path / "nope2.jsonl"),
            "--embedder",
            "x:y",
        ]
    )
    assert rc == 2  # _EXIT_FILE_NOT_FOUND


def test_cli_bad_jsonl_exits_5(tmp_path: Path, capsys):
    pp = tmp_path / "bad.jsonl"
    dd = tmp_path / "dd.jsonl"
    pp.write_text("this is not json\n")
    _write_jsonl(dd, _distractors())
    rc = cli_main(
        [
            "--paraphrases",
            str(pp),
            "--distractors",
            str(dd),
            "--embedder",
            "x:y",
        ]
    )
    assert rc == 5  # _EXIT_BAD_JSONL


def test_cli_embedder_import_failure_exits_3(tmp_path: Path, capsys):
    pp = tmp_path / "pp.jsonl"
    dd = tmp_path / "dd.jsonl"
    _write_jsonl(pp, _paraphrases())
    _write_jsonl(dd, _distractors())
    rc = cli_main(
        [
            "--paraphrases",
            str(pp),
            "--distractors",
            str(dd),
            "--embedder",
            "nonexistent_module_xyz:thing",
        ]
    )
    assert rc == 3  # _EXIT_EMBEDDER_IMPORT


def test_cli_no_threshold_meets_constraint_exits_4(tmp_path: Path, monkeypatch):
    pp = tmp_path / "pp.jsonl"
    dd = tmp_path / "dd.jsonl"
    _write_jsonl(pp, _paraphrases())
    _write_jsonl(dd, _distractors())

    import sys as _sys

    fake_mod = type(_sys)("_test_calibrate_embedder4")
    fake_mod.embedder = ParaphraseEmbedder(dim=32)  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "_test_calibrate_embedder4", fake_mod)

    rc = cli_main(
        [
            "--paraphrases",
            str(pp),
            "--distractors",
            str(dd),
            "--embedder",
            "_test_calibrate_embedder4:embedder",
            "--min-precision",
            "1.5",  # impossible
        ]
    )
    assert rc == 4  # _EXIT_NO_THRESHOLD


def test_cli_jsonl_skips_empty_and_comment_lines(tmp_path, capsys, monkeypatch):
    pp = tmp_path / "pp.jsonl"
    dd = tmp_path / "dd.jsonl"
    with open(pp, "w") as f:
        f.write("# this is a comment\n")
        f.write("\n")  # blank
        for a, b in _paraphrases():
            f.write(json.dumps({"a": a, "b": b}) + "\n")
    _write_jsonl(dd, _distractors())

    import sys as _sys

    fake_mod = type(_sys)("_test_calibrate_embedder5")
    fake_mod.embedder = ParaphraseEmbedder(dim=32)  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "_test_calibrate_embedder5", fake_mod)

    rc = cli_main(
        [
            "--paraphrases",
            str(pp),
            "--distractors",
            str(dd),
            "--embedder",
            "_test_calibrate_embedder5:embedder",
        ]
    )
    assert rc == 0


def test_cli_grid_override(tmp_path, capsys, monkeypatch):
    pp = tmp_path / "pp.jsonl"
    dd = tmp_path / "dd.jsonl"
    _write_jsonl(pp, _paraphrases())
    _write_jsonl(dd, _distractors())

    import sys as _sys

    fake_mod = type(_sys)("_test_calibrate_embedder6")
    fake_mod.embedder = ParaphraseEmbedder(dim=32)  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "_test_calibrate_embedder6", fake_mod)

    rc = cli_main(
        [
            "--paraphrases",
            str(pp),
            "--distractors",
            str(dd),
            "--embedder",
            "_test_calibrate_embedder6:embedder",
            "--grid",
            "0.2,0.5,0.8",
            "--format",
            "json",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    # 3 grid points
    assert len(payload["pr_curve"]) == 3
