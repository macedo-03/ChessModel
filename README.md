# ChessModel — Explainable Chess AI

An end-to-end, automated MLOps pipeline and serving infrastructure for an
explainable chess-playing bot: continuous training from Lichess game data,
statistical (SPRT) promotion gating, containerized serving on the Lichess Bot API,
and a dual-speed interpretability layer (linear concept probing, saliency maps,
and post-game LLM commentary).

See [`docs/DEVELOPMENT_PIPELINE.md`](docs/DEVELOPMENT_PIPELINE.md) for the full
phased build plan, scope ladder, and risk register. Architectural decisions are
recorded under [`docs/adr/`](docs/adr/).

## Status

Phase 00 — Foundations. No trained model yet.

## Development

```bash
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

## Project layout

```
src/chessmodel/
  data/       # ingestion, encoding, DVC pipeline stages   (Phase 01)
  training/   # PyTorch model, training loop, MLflow runs  (Phase 02)
  xai/        # concept probes, saliency, LLM commentary   (Phase 03, 08)
  serving/    # FastAPI app, ONNX inference, Lichess bot    (Phase 04, 05)
docs/
  DEVELOPMENT_PIPELINE.md
  adr/        # architectural decision records
infra/
  docker/     # container images
```
