"""Tests for the Kaggle runner's dataset-path resolution.

Not testing main()/argparse wiring here, consistent with the other thin CLI
wrappers in this project (ingest.py, dataset_card.py, run_local.py) -- only
the one piece of actual logic in this file is worth locking in directly.
"""

from pathlib import Path

import pytest

from chessmodel.training import run_kaggle
from chessmodel.training.run_kaggle import (
    KAGGLE_INPUT,
    find_dataset_path,
    resolve_kaggle_dataset_path,
)


def test_resolve_kaggle_dataset_path() -> None:
    resolved = resolve_kaggle_dataset_path("chessmodel-dataset", "lichess_2026-07_elo2200.pgn")

    assert resolved == KAGGLE_INPUT / "chessmodel-dataset" / "lichess_2026-07_elo2200.pgn"
    assert str(resolved).startswith(str(Path("/kaggle/input")))


def test_find_dataset_path_uses_conventional_layout_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_kaggle, "KAGGLE_INPUT", tmp_path)
    conventional = tmp_path / "chessmodel-dataset" / "data.pgn"
    conventional.parent.mkdir()
    conventional.touch()

    assert find_dataset_path("chessmodel-dataset", "data.pgn") == conventional


def test_find_dataset_path_falls_back_to_search_when_layout_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_kaggle, "KAGGLE_INPUT", tmp_path)
    nested = tmp_path / "datasets" / "macedo03" / "chessmodel-dataset" / "data.pgn"
    nested.parent.mkdir(parents=True)
    nested.touch()

    assert find_dataset_path("chessmodel-dataset", "data.pgn") == nested


def test_find_dataset_path_raises_with_listing_when_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_kaggle, "KAGGLE_INPUT", tmp_path)
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "other.txt").touch()

    with pytest.raises(FileNotFoundError, match="unrelated"):
        find_dataset_path("chessmodel-dataset", "data.pgn")


def test_find_dataset_path_raises_when_search_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_kaggle, "KAGGLE_INPUT", tmp_path)
    for owner in ("a", "b"):
        nested = tmp_path / "datasets" / owner / "data.pgn"
        nested.parent.mkdir(parents=True)
        nested.touch()

    with pytest.raises(FileNotFoundError, match="2 matches"):
        find_dataset_path("chessmodel-dataset", "data.pgn")
