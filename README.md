# ChessModel - Explainable Chess AI

An end-to-end MLOps pipeline and serving stack for an explainable chess-playing bot:
versioned data ingestion from Lichess game dumps, a supervised policy/value network,
containerized inference, a bot that plays real games.

Build plan, scope ladder and risk register: [`docs/DEVELOPMENT_PIPELINE.md`](docs/DEVELOPMENT_PIPELINE.md).
Current state, setup and open questions: [`docs/PROJECT_BLUEPRINT.md`](docs/PROJECT_BLUEPRINT.md).
Architectural decisions: [`docs/adr/`](docs/adr/).

## Status

Phases 00–02, 04 and 05 are complete. Phase 03 (the synchronous XAI layer) is next.

| | |
|---|---|
| **Dataset** | 200,000 Lichess games, both players ≥2200 Elo, bullet excluded — 600 MB filtered PGN, DVC-tracked on Cloudflare R2. [Dataset card](docs/datasets/lichess_2026-07_elo2200.md) |
| **Model** | ResNet trunk (128 channels × 8 residual blocks) + policy head over AlphaZero's 4,672-move action space + tanh value head. 2.7M parameters, trained 10 epochs on a Kaggle P100 |
| **Benchmark** | 51.23% top-1 and 79.45% top-3 move accuracy against 816,275 held-out positions; value Brier score 0.1942. [Benchmark card](docs/models/best.md) |
| **Serving** | FastAPI over the PyTorch checkpoint, CPU-only, 1.09 GB image; schema and a 200 ms latency budget asserted by a CI integration job that boots the container |
| **Bot** | Plays standard-variant games on Lichess as [XAIKnightmare](https://lichess.org/@/XAIKnightmare), run on demand rather than hosted 24/7 |

`/move`'s `explanation` field exists and returns `null` — Phase 03 fills it.

## Quickstart

```bash
uv sync --extra dev --extra model --extra training --extra serving --extra bot
uv run pre-commit install
uv run pytest                       # 121 tests
```

Play it in the terminal (needs a trained checkpoint — `*.pt` is not in git):

```bash
uv run python -m chessmodel.serving.play --checkpoint path/to/best.pt --side white
```

Serve it:

```bash
CHECKPOINT_PATH=path/to/best.pt uv run uvicorn chessmodel.serving.api:app --port 8000
curl -X POST localhost:8000/move -H 'Content-Type: application/json' \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

Or bring up the whole stack — inference API, Redis, and the Lichess bot
(needs `LICHESS_TOKEN` in a local `.env`):

```bash
docker compose up --build
```

Setting this up on a fresh machine, including the secrets and artifacts git does not
carry, is covered step by step in
[the blueprint](docs/PROJECT_BLUEPRINT.md#setup-on-a-new-machine-macos).

## How it fits together

```
Lichess monthly dump  →  data/ingest.py       stream-filter by Elo, headers only
filtered PGN (DVC)    →  data/dataset.py      IterableDataset, encodes on the fly
                         data/encode.py       board → (19,8,8); move → [0,4671]
                      →  training/train.py    MLflow-tracked, checkpoints best.pt
best.pt               →  training/evaluate.py top-k accuracy + Brier benchmark card
                      →  serving/api.py       /health · /model-info · /move
                      →  bot/lichess_bot.py   Lichess Bot API, HTTP-only client
                      →  Redis queue          finished games, awaiting Phase 08
```

Positions are encoded at training time rather than persisted as tensors: expanded
float32 shards would have cost ~93× the filtered PGN's size, and the CPU encode is
trivial next to per-batch GPU time. The bot talks to the API over HTTP and never imports
torch, which is why its image is 295 MB.

## Project layout

```
src/chessmodel/
  data/       ingestion, encoding, streaming dataset, dataset cards   (Phase 01)
  training/   ChessNet, training loop, evaluation, local/Kaggle runners (Phase 02)
  serving/    FastAPI app, terminal play, inference helpers           (Phase 04)
  bot/        Lichess client                                          (Phase 05)
  xai/        concept probes, saliency                                (Phase 03, planned)
docs/
  PROJECT_BLUEPRINT.md    current state, setup, action items
  DEVELOPMENT_PIPELINE.md the 11-phase plan
  adr/                    architectural decision records
  datasets/ models/       generated data and model cards
infra/docker/             dev, serving, and bot images
scripts/                  CI helpers (dummy checkpoint, container verification)
```

## License

MIT — see [`LICENSE`](LICENSE).
