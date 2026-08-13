"""Tests for the Kaggle runner's dataset-path resolution.

Not testing main()/argparse wiring here, consistent with the other thin CLI
wrappers in this project (ingest.py, dataset_card.py, run_local.py) -- only
the one piece of actual logic in this file is worth locking in directly.
"""

from pathlib import Path

from chessmodel.training.run_kaggle import KAGGLE_INPUT, resolve_kaggle_dataset_path


def test_resolve_kaggle_dataset_path() -> None:
    resolved = resolve_kaggle_dataset_path("chessmodel-dataset", "lichess_2026-07_elo2200.pgn")

    assert resolved == KAGGLE_INPUT / "chessmodel-dataset" / "lichess_2026-07_elo2200.pgn"
    assert str(resolved).startswith(str(Path("/kaggle/input")))
