# ChessModel — Project Blueprint

**Snapshot date:** 2026-11-19 · **Branch:** `main` · **HEAD:** `9a1e4ec`
**Purpose:** supplementary context for picking this project up on a different machine
(specifically macOS). Read alongside [`DEVELOPMENT_PIPELINE.md`](DEVELOPMENT_PIPELINE.md)
(the phase plan) and [`adr/`](adr/) (the binding decisions). This file describes *what
exists right now*; the pipeline doc describes *what the plan is*.

---

## 1. Global architecture

An explainable chess AI with the MLOps scaffolding around it. Five subsystems, each a
Python package under `src/chessmodel/`, connected by files (PGN, checkpoints), HTTP, and
one Redis queue — never by shared in-process state.

```
Lichess monthly dump (.pgn.zst, ~100GB, streamed)
        │  data/ingest.py — header-only filter, never decompresses to disk
        ▼
data/raw/lichess_2026-07_elo2200.pgn      600 MB · 200k games · DVC-tracked (R2)
        │  data/dataset.py — IterableDataset, replays games, encodes on the fly
        │  data/encode.py  — board→(19,8,8); move→[0,4671]; value from mover's POV
        ▼
training/train.py  ──MLflow(sqlite)──▶  runs, metrics, registered model "chessmodel"
        │  ChessNet (training/model.py): ResNet trunk 128ch × 8 blocks,
        │  policy head → 4,672 logits, value head → tanh scalar.  2,675,290 params
        ▼
checkpoints/best.pt (32 MB: weights + optimizer + scheduler + config dict)
        │                                   │
        │ training/evaluate.py              │ serving/play.py — load_model_for_inference
        │  → docs/models/best.md            │                   select_model_move
        ▼                                   ▼
benchmark card                     serving/api.py (FastAPI)
                                     /health · /model-info · /move
                                     explanation field present, always null (Phase 03)
                                            ▲ HTTP
                                            │
                                   bot/lichess_bot.py  ── Lichess Bot API (NDJSON)
                                            │  event stream on main thread,
                                            │  one thread per active game
                                            ▼
                                   Redis LPUSH "chessmodel:finished_games"
                                            ▼
                                   (Phase 08 consumer — does not exist yet)
```

### Module responsibilities

| Package | Files | Role |
|---|---|---|
| `chessmodel.data` | `ingest.py`, `dataset.py`, `encode.py`, `dataset_card.py` | Stream-filter the Lichess dump by Elo/time control (headers only, no python-chess); stream positions from the filtered PGN; encode board/move/value; generate the dataset card. |
| `chessmodel.training` | `model.py`, `train.py`, `evaluate.py`, `run_local.py`, `run_kaggle.py`, `kernel-metadata.json` | `ChessNet`; the environment-agnostic training loop (`train(config) -> Path`); the standalone benchmark harness; two thin environment wrappers (local CLI, Kaggle kernel). |
| `chessmodel.serving` | `api.py`, `play.py` | FastAPI app (lifespan-loaded model); terminal play loop + the shared inference helpers (`load_model_for_inference`, `select_model_move`, `_MODEL_SHAPE_KEYS`) that the API reuses. |
| `chessmodel.bot` | `lichess_bot.py` | Lichess client. Makes HTTP calls only — never imports torch. Needs the serving API to be up. |
| `chessmodel.xai` | `__init__.py` (empty) | Phase 03 placeholder. Nothing implemented. |

### Key interaction contracts

- **Encoding is the project's spine.** `encode.board_to_tensor` / `move_to_policy_index`
  are imported by the dataset, the model (plane-count constants), serving, and every
  test. The flat policy index order — `(row*8 + col)*73 + plane` (ADR-0003) — must agree
  with `model.flatten_policy_logits`'s `permute(0,2,3,1)`. If those two silently
  disagree, training still converges and every move is wrong. `tests/test_encode.py` and
  `tests/test_model.py` are the guardrail.
- **Value sign conventions.** Internally the value target/output is always from the
  **mover's** perspective (ADR-0002's fix for the prior attempt's bug). Both human-facing
  surfaces (`play.py`, `/move`'s `eval`) negate it for Black so the user always reads a
  White-perspective score.
- **Checkpoint format is ours, not MLflow's.** `save_checkpoint`/`load_checkpoint` in
  `train.py` define the artifact everything else loads. `mlflow.pytorch.log_model` is
  called purely so the MLflow **Registry** has a flavored model to version against.
- **Bot ↔ API boundary is HTTP.** The bot posts `{"fen": ...}` to `/move`; it never loads
  the model. This is what keeps the bot image at 295 MB with no torch.

---

## 2. Implementation status

Phases refer to `DEVELOPMENT_PIPELINE.md`. Eleven phases, 00–10.

### Complete and verified

| Phase | State | Evidence |
|---|---|---|
| **00 Foundations** | Done | 121 tests collected, `uv run pytest` green; ruff + mypy + pre-commit configured; CI green on GitHub; ADR-0001…0004 accepted; `tests/test_chess_harness.py` covers legal-move generation, FEN round-trip, PGN parsing, and the draw conditions. |
| **01 Data pipeline** | Done | `dvc.yaml` `ingest` stage + `dvc.lock` (md5 `8c8e67d9…`, 600,015,314 bytes); R2 remote configured; dataset card at `docs/datasets/lichess_2026-07_elo2200.md` (200,000 games, Elo 2200–3269, ~16.3M usable positions). |
| **02 Supervised model** | Done | Kaggle P100 run, 10 epochs, 601,650 steps, final `val_loss` 2.2837, `val_policy_top1` 0.5123. Benchmark card `docs/models/best.md`: **816,275** held-out positions, **top-1 51.23%**, **top-3 79.45%**, **Brier 0.1942**. Registry path exercised by a test in `tests/test_train.py`. |
| **04 Serving** | Done | `serving/api.py` + `tests/test_api.py` (6 tests); `infra/docker/Dockerfile.serving` → **1.09 GB** image (down from 5.29 GB before the extras split); `docker-compose.yml` (api + redis + bot); CI `container-integration` job boots the container and asserts schema + a 200 ms latency budget via `scripts/verify_container.py`. |
| **05 Lichess bot** | Done (closeout uncommitted) | Bot account **XAIKnightmare** played real games end-to-end as both colors; `tests/test_lichess_bot.py` 27 tests including every Lichess terminal status; challenge accept latency measured at 130–235 ms after the client split (was 3.2–6.6 s). |

### Known gaps inside "complete" phases

- **No held-out test split.** Only train/val (chronological, game-level, 95/5). Every
  reported number is validation-set. Phase 03's probe accuracies will land on the same
  split unless this is fixed — see Action Items.
- **No hyperparameter sweep.** Phase 02 listed a small Optuna/grid sweep as nested MLflow
  runs. One configuration was trained (128ch/8 blocks per ADR-0004). Not done, not
  blocking.
- **The MLflow store is not portable.** `mlruns/` is gitignored. The local stores
  (`mlruns/smoke_test/mlflow.db`, `mlruns/real_size_test/mlflow.db`) and the Kaggle run
  pulled into `src/chessmodel/training/output/mlruns/mlflow.db` each contain exactly one
  run, and **none currently contains a registered model version** — the registration code
  path works and is unit-tested, but there is no live registry state to inherit on a new
  machine. Treat the registry as something to re-establish, not restore.
- **`best.pt`'s stored `dataset_path` is a mangled Windows string**
  (`\kaggle\input\datasets\...`), a leftover of the PosixPath→WindowsPath repair done
  when the Kaggle checkpoint was first opened on Windows. `evaluate_checkpoint` falls
  back to that path, so on macOS **always pass `--dataset-path` explicitly**.

### Work in progress

- **Phase 05 closeout** — uncommitted, staged: `lichess_bot.py` (+28 lines: three-way
  status handling so `aborted`/`noStart` stop the loop without queueing, `created`
  treated as in-play, O(n²) gameState logging trimmed to move-count/last-move/status),
  `tests/test_lichess_bot.py` (+57 lines: parametrized terminal statuses, aborted/noStart
  not queued, created in-play), `run_kaggle.py` (1 line), `.devcontainer/devcontainer.json`
  (all five extras), `uv.lock`. **The user commits themselves — do not commit without an
  explicit request.**
- **`.claude/` is untracked.** It holds the project working agreements (`CLAUDE.md`) and
  the shared permission allowlist (`settings.json`) — neither reaches another machine
  until committed. See Action Items.

### Not started

| Phase | Planned shape |
|---|---|
| **03 Synchronous XAI** | `xai/` is an empty package. Design settled: concept dataset from python-chess heuristics (king exposure, center control, pawn integrity, material balance) over Phase 01 positions; forward hooks on residual blocks, activations mean-pooled to 128 dims; linear/logistic probes per layer logged to MLflow; **positive control** (material balance, must be highly separable) and **negative control** (random labels, must sit at chance) plus input-tensor and untrained-network baselines so a probe result means something; attribution via **gradient×input** (one backward pass, ~22.1 ms measured on CPU) rendered as an 8×8 heatmap; one `explain(board, model) -> ExplanationResult` entry point consumed by `/move`'s currently-null `explanation` field. Agreed next concrete step: `xai/concepts.py` + tests. |
| **06 MLOps automation** | Prefect flows over Phase 01/02; dynamic `month` in `dvc.yaml` (hardcoded `2026-07` today — a scheduled `dvc repro` would refetch the same month forever); scheduled `dvc repro && dvc push` committing `dvc.lock`; MLflow Registry stage transitions. |
| **07 SPRT arena + self-play** | Arena vs Production and vs Stockfish levels; Wald SPRT; capped self-play where the **policy target must be MCTS visit counts**, never temperature-sampled policy moves. |
| **08 Async XAI & commentary** | Celery worker on the Redis queue the bot already fills; Stockfish ACPL; probe timeline; grounded LLM recap. |
| **09 Observability** | Prometheus in FastAPI, Grafana, Evidently drift. |
| **10 Portfolio packaging** | Writeup, demo clip, architecture diagram. |

---

## 3. Technical decisions

### Binding ADRs (`docs/adr/`) — changing any of these invalidates the trained checkpoint

- **ADR-0001 — input tensor.** `(19, 8, 8)` float32: 12 piece planes, side-to-move,
  4 castling planes, en-passant target, normalized halfmove clock. Row 0 = rank 8,
  col 0 = file a, **absolute orientation, never flipped by side to move** — the encoder
  stays a pure function of the FEN and the net learns color symmetry from plane 12.
- **ADR-0002 — prior-attempt pitfalls checklist.** Distilled from an abandoned 2024–25
  attempt (`ChessBot`, separate repo, no code carried over). Two entries drive real code
  here: the value label must be **mover-perspective** (the prior bug was an absolute
  White-perspective label reused for every ply — a contradictory signal half the time),
  and the dataset must be **streamed, never materialized**.
- **ADR-0003 — policy output.** AlphaZero 8×8×73 → 4,672-way classification. Planes
  0–55 queen-type (incl. castling and auto-queen promotion), 56–63 knight, 64–72
  underpromotion. Flat index `(row*8 + col)*73 + plane`.
- **ADR-0004 — value head.** Shared trunk → `Conv2d(128→8, 1×1)` + BN + ReLU → flatten
  (512) → `Linear(512,256)` + ReLU → `Linear(256,1)` → tanh. Trunk fixed at 128 channels
  / 8 residual blocks — the cheap end of the plan's 6–10 × 128–256 range, chosen for the
  Kaggle free-tier GPU budget.

### Decisions made during implementation

| Decision | Rationale | Rejected alternative |
|---|---|---|
| Encode positions on the fly in the DataLoader | Expanded float32 tensors would be ~53 GB vs. the 600 MB filtered PGN (~93×), blowing the R2 free tier for nothing — CPU encode cost is trivial next to per-batch GPU time | Memory-mapped tensor shards (the original Phase 01 plan) |
| Elo ≥ 2200 both players | ≥2100 had a hit rate that would have needed ~14 h to reach a useful sample; ≥2200 took ~11 min | ≥2100 (original target) |
| Encoding/splitting are runtime ops, not DVC stages | They produce no files, so a stage would have no outs | Four separate `dvc.yaml` stages |
| **No ONNX** | A single forward pass at 2.7M params already clears any chess time control in eager mode; ONNX would add an export/version surface and a second runtime, and the XAI layer needs the live PyTorch graph for gradients anyway | ONNX Runtime serving (explicitly dropped; revisit only if the optional MCTS tier needs many evals/move) |
| Our own checkpoint dict is the load format; MLflow model logging only for the Registry | `mlflow.register_model` in MLflow 3.x requires a *logged model* with MLmodel metadata, which `log_artifact` doesn't produce | Registering the raw artifact |
| MLflow tracking = local SQLite, auto-resolved from a bare path | MLflow 3.x put the plain-filesystem backend in maintenance mode and it refuses to initialize without an opt-out env var. `_resolve_mlflow_tracking_uri` also treats a ≤1-char URI scheme as a Windows drive letter, not a scheme | `file://` store; a hosted tracking server (unjustified at this scale) |
| Dependency extras split: base / `model` / `training` / `serving` / `bot` | torch is not a base dependency, so the bot image never considers the multi-GB CUDA wheel; the serving image skips dvc/mlflow/kaggle entirely. Serving additionally excludes torch + 15 nvidia/triton packages from the frozen sync and installs the CPU wheel fresh | One flat dependency list (produced a 5.29 GB serving image with 2.7 GB of orphaned CUDA packages) |
| Write our own ~290-line Lichess client | Full control over the event loop, latency instrumentation, and the Redis hook; the upstream `lichess-bot` project is a large engine-oriented framework we'd mostly fight | `lichess-bot` upstream |
| Lichess `status` is authoritative for game over | Re-deriving threefold repetition / 50-move / insufficient material locally is a second source of truth that can disagree with the server | Local `board.is_game_over()` |
| Two separate httpx clients in the bot | The always-open NDJSON stream contended with time-sensitive one-off POSTs; accept latency dropped from 3.2–6.6 s to 130–235 ms | One shared client |
| Thread per active game, event stream on main thread | A long game must not block noticing/accepting new challenges; `httpx.Client` and `redis.Redis` are both documented thread-safe | asyncio rewrite |
| gradient×input for the sync XAI path | Measured: one forward+backward ≈ 22.1 ms CPU, so IG's usual 20–50 steps would be 440–1,100 ms against a sub-100 ms budget | Integrated gradients (kept for the Phase 08 async path) |
| Kaggle script kernel installing the package from GitHub, with cu126 torch preinstalled | The latest PyPI torch wheel targets newer architectures and dies on the P100 (CC 6.0) with `CUDA error: no kernel image is available`; pinning `torch==2.13.0+cu126` first stops the later install from touching it | Notebook kernel; letting pip resolve torch |

---

## 4. Environment and pipeline

### Dependencies

- **Python ≥ 3.11** (`requires-python`), CI and Docker images use **3.11/3.12**;
  ruff/mypy target `py312`.
- **uv** is the only package manager. `uv.lock` is committed; Docker builds use
  `--frozen`.
- Extras: `model` (torch ≥2.13.0), `training` (dvc[s3], requests, zstandard, tqdm,
  mlflow, kaggle), `serving` (fastapi, uvicorn[standard]), `bot` (httpx, redis),
  `dev` (pytest, pytest-cov, ruff, mypy, pre-commit, httpx). Base: python-chess, numpy.
- **External services:** Cloudflare R2 (DVC remote `s3://chessmodel-dvc`), Kaggle
  (dataset `macedo03/chessmodel-dataset`, kernel `macedo03/chessmodel-training`),
  Lichess (bot account XAIKnightmare), GitHub Actions, Docker.

### Secrets and machine-local state (none of this is in git)

| Item | Location | Notes |
|---|---|---|
| `LICHESS_TOKEN` | `.env` at repo root (gitignored) | Single key in the file. Bot scope token. The BOT account upgrade is **one-way**; XAIKnightmare is a throwaway account. **Never print the value.** |
| R2 keys | `.dvc/config.local` (`access_key_id`, `secret_access_key`) | Gitignored via `.dvc/.gitignore`. Needed for `dvc pull`/`dvc push`. |
| Kaggle API token | `~/.kaggle/kaggle.json` | Needed for `kaggle kernels push/output`. |
| Trained checkpoints | `src/chessmodel/training/output/checkpoints/{best,latest}.pt` | `*.pt` is gitignored — **32 MB each, must be copied to the new machine by hand** (or re-pulled with `kaggle kernels output macedo03/chessmodel-training`). |
| Dataset | `data/raw/lichess_2026-07_elo2200.pgn` (600 MB) | DVC-tracked. `dvc pull` if it was pushed to R2 (verify with `dvc status -c`), else `dvc repro` re-downloads and re-filters the monthly dump. |
| MLflow stores | `mlruns/`, `src/chessmodel/training/output/mlruns/` | Gitignored, local-only, not portable. |

### Local dev loop

```bash
uv sync --extra dev --extra model --extra training --extra serving --extra bot
uv run pre-commit install
uv run pytest                 # 121 tests
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

Run things:

```bash
# terminal play
uv run python -m chessmodel.serving.play \
  --checkpoint src/chessmodel/training/output/checkpoints/best.pt --side white

# benchmark card (ALWAYS pass --dataset-path: the stored one is a mangled Windows string)
uv run python -m chessmodel.training.evaluate \
  --checkpoint src/chessmodel/training/output/checkpoints/best.pt \
  --dataset-path data/raw/lichess_2026-07_elo2200.pgn

# API without Docker
CHECKPOINT_PATH=src/chessmodel/training/output/checkpoints/best.pt \
  uv run uvicorn chessmodel.serving.api:app --port 8000

# local training smoke run
uv run python -m chessmodel.training.run_local \
  --dataset-path data/raw/lichess_2026-07_elo2200.pgn \
  --checkpoint-dir /tmp/ckpt --mlflow-tracking-uri mlruns \
  --max-games 200 --num-epochs 1 --batch-size 32

# Kaggle GPU training (the user runs this themselves)
uv run kaggle kernels push -p src/chessmodel/training/
uv run kaggle kernels output macedo03/chessmodel-training -p ./pulled
```

### Deployment

- `docker-compose.yml`: **api** (`Dockerfile.serving`, port 8000, mounts
  `./src/chessmodel/training/output/checkpoints:/checkpoints:ro`,
  `CHECKPOINT_PATH=/checkpoints/best.pt`, health-checked) → **redis** (`redis:alpine`,
  health-checked) → **bot** (`Dockerfile.bot`, `env_file: .env`,
  `CHESSMODEL_API_URL=http://api:8000`, `REDIS_URL=redis://redis:6379/0`, waits for both
  to be healthy). `docker compose up --build`, then challenge XAIKnightmare on Lichess.
- Both runtime images are multi-stage, non-root (`appuser`, uid 1000), `python:3.11-slim`.
  `Dockerfile.bot` sets `PYTHONUNBUFFERED=1` — without it `print()` output never reaches
  `docker logs`.
- `infra/docker/Dockerfile` is the older dev/CI-parity image (`CMD uv run pytest`),
  superseded for serving.
- Target production host per the cost analysis: Oracle Cloud Always Free **ARM** VM
  (2 OCPU / 12 GB). **Nothing is provisioned yet** — the bot has only ever run locally.
  The images have only been built for the host's own architecture.

### CI/CD (`.github/workflows/ci.yml`)

Two jobs, on push to `main` and on every PR:

1. **lint-and-test** — uv, Python 3.12, `uv sync` with all five extras, `ruff check`,
   `ruff format --check`, `mypy src`, `pytest --cov`.
2. **container-integration** (needs 1) — generate a dummy checkpoint
   (`scripts/make_dummy_checkpoint.py`), build `Dockerfile.serving`, `docker run` it with
   the checkpoint mounted, run `scripts/verify_container.py` (stdlib-only: polls
   `/health`, POSTs the starting FEN to `/move`, asserts the response schema, `eval` in
   `[-1, 1]`, and latency under 200 ms), dump container logs on failure, always clean up.

Pre-commit: ruff (+`--fix`), ruff-format, mypy (excluding `tests/`), trailing whitespace,
end-of-file, check-yaml, max-1 MB added files.

No CD yet: no image push to a registry, no deploy step, no Prefect (Phase 06).

### Setup on a new machine (macOS)

A clone gets the code and `uv.lock` and nothing else. Four categories of state never
travel through git: the secrets, the dataset, the checkpoint, and — until it is
committed — `.claude/`.

#### Step 0 — on the old machine, before you leave it

```bash
git add docs/PROJECT_BLUEPRINT.md .claude/
git add -u            # the Phase 05 closeout + devcontainer + uv.lock
git commit && git push
```

Then copy these onto a USB stick or private cloud drive — git carries none of them:

| File | Size | Without it |
|---|---|---|
| `.env` | tiny | the bot refuses to start (`LICHESS_TOKEN must be set`) |
| `.dvc/config.local` | tiny | `dvc pull`/`dvc push` cannot authenticate to R2 |
| `~/.kaggle/kaggle.json` | tiny | no `kaggle` CLI (it lives outside the repo) |
| `src/chessmodel/training/output/checkpoints/best.pt` | 32 MB | nothing that runs the model works |

The 600 MB PGN can stay behind — `dvc pull` should fetch it, if it was ever pushed.

#### Step 1 — toolchain

```bash
# Homebrew first, if the machine doesn't have it:
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install uv                # `brew upgrade uv` keeps it current afterwards
brew install --cask docker     # launch once, then Settings > General > start at login
```

#### Step 2 — clone, sync, and prove the code works

```bash
git clone git@github.com:macedo-03/ChessModel.git   # SSH remote: needs a key on this
cd ChessModel                                        # machine, or switch to https://…
uv sync --extra dev --extra model --extra training --extra serving --extra bot
uv run pre-commit install
uv run pytest                                        # expect 121 passed
```

uv downloads a suitable Python itself if the machine has none. A green `pytest` here means
the entire code side is working; everything after this is data and secrets.

#### Step 3 — put the carried files in place

```bash
cp /Volumes/stick/.env .
cp /Volumes/stick/config.local .dvc/
mkdir -p ~/.kaggle && cp /Volumes/stick/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
mkdir -p src/chessmodel/training/output/checkpoints
cp /Volumes/stick/best.pt src/chessmodel/training/output/checkpoints/
```

Alternative to carrying `best.pt`:
`uv run kaggle kernels output macedo03/chessmodel-training -p ./pulled` re-downloads it
from the kernel run (the pulled tree also contains that run's `mlruns/mlflow.db`).

#### Step 4 — the dataset, only for training or re-benchmarking

```bash
dvc status -c      # does R2 actually hold it?
dvc pull           # 600 MB
```

If R2 does not have it, `dvc repro` rebuilds it from the Lichess monthly dump — that
streams 100 GB+ and takes ~11 min of filtering on top of the download. Skip this step
entirely if you only want to play or serve.

#### Step 5 — verify the stack

```bash
# the model loads and plays
uv run python -m chessmodel.serving.play \
  --checkpoint src/chessmodel/training/output/checkpoints/best.pt --side white

# api + redis + bot together, then challenge XAIKnightmare on Lichess
docker compose up --build
```

**Minimum to just play against the model:** steps 1, 2, and `best.pt`. The dataset,
Kaggle token and R2 keys are only needed for training, re-benchmarking, and `dvc`.

macOS-specific differences from the Windows machine this was built on:

- The Git Bash workarounds do not apply: no `MSYS_NO_PATHCONV=1` for docker `-v` paths,
  no `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` for the Kaggle CLI's charmap crash.
- **Apple Silicon builds arm64 images.** `Dockerfile.serving` installs the CPU torch
  wheel from `download.pytorch.org/whl/cpu`; verify a linux/aarch64 wheel resolves, or
  build with `--platform linux/amd64` (slow, emulated). This is the same multi-arch gap
  that the eventual Oracle ARM VM will hit — see Action Items. Faster escape hatch if that
  wheel turns out not to exist: replace Docker Desktop with Colima running an x86_64 VM
  (`brew install colima docker docker-compose`, then
  `colima start --arch x86_64 --vm-type=vz --vz-rosetta --cpu 4 --memory 8`), which uses
  Rosetta translation instead of per-build QEMU. Same `docker` CLI either way.
- Torch on macOS is CPU/MPS only. `train.py`'s `_resolve_device` checks CUDA only, so a
  local run falls back to CPU; pass `--device mps` explicitly to try MPS (untested here).
- `Path` handling: the checkpoint's stored `dataset_path` is a Windows-mangled string;
  always override it on the CLI.

### graphify (evaluated, dropped 2026-11-19)

A knowledge graph of the repo was built with graphify (493 nodes, 949 edges, 26
communities) and then removed, along with its hooks and vendored skill. It correctly
identified the structural chokepoints — `move_to_policy_index()` bridges five
communities, which is exactly where a silent ADR-0003 encoding bug would hide — but at
5,000 lines of source the rest was restating import blocks, and the 230k-token build cost
bought little that this blueprint doesn't state directly. Worth revisiting around Phases
06–09, when Prefect flows, a Celery worker and observability config roughly double the
surface area.

### Working agreements carried over

Also encoded in `.claude/CLAUDE.md`, so a new Claude Code session picks them up without
having to read this far.

- The user **commits and pushes themselves**. List proposed commits; don't run them
  unasked.
- The user runs the Kaggle pushes themselves.
- `DEVELOPMENT_PIPELINE.md` is mirrored by a published artifact
  (`https://claude.ai/code/artifact/478d880d-b981-44a6-bcd0-3a4922f46759`). **Ask before
  syncing plan edits** to either copy; never keep a standalone HTML copy in the repo.
- Never paste the Lichess token into chat; inspect `.env` by key name only.

---

## 5. Action items

### P0 — do these first, they block a new machine

1. **Commit `.claude/`** (`CLAUDE.md` with the working agreements, `settings.json` with
   the shared permission allowlist). Without it a new session has no project
   instructions. `settings.local.json` is machine-local and now gitignored.
2. **Commit the Phase 05 closeout** (`lichess_bot.py`, `tests/test_lichess_bot.py`,
   `run_kaggle.py`, `.devcontainer/devcontainer.json`, `uv.lock`) so the aborted-game fix
   and the 27-test suite travel with the repo.
3. **Move the unversioned artifacts** — `best.pt`, the 600 MB PGN, `.env`,
   `.dvc/config.local`, `~/.kaggle/kaggle.json`. Verify `dvc status -c` tells you whether
   R2 actually has the dataset before relying on `dvc pull`. Full procedure:
   [Setup on a new machine](#setup-on-a-new-machine-macos).

### P1 — next real work

4. **Phase 03, step 1:** `xai/concepts.py` — python-chess heuristics for king exposure,
   center control, pawn integrity, material balance, plus tests. Then forward hooks and
   the probe fitting loop with the positive/negative controls and the two baselines.
5. **Add a held-out test split** before any probe number is reported. Three-way
   chronological game-level split (e.g. 90/5/5); `split_train_val` and
   `_build_datasets`/`_effective_total_games` are the two places to change, and
   `evaluate.py` reads the split from the checkpoint config, so old checkpoints keep
   evaluating against the split they were trained with.
6. **Refresh `README.md`** — it still says "Phase 00 — Foundations. No trained model yet."
   and advertises ONNX inference. It is the first thing a portfolio reader sees.

### P2 — queued, from the plan review

7. **Multi-arch images** (`docker buildx` linux/amd64 + linux/arm64) — needed both for
   Apple Silicon and for the Oracle ARM VM.
8. **Provision the serving host** and actually run api+redis+bot somewhere 24/7.
9. **Phase 06 prerequisites:** make `dvc.yaml`'s `month` dynamic; decide the
   Prefect↔Kaggle story; schedule `dvc repro && dvc push` with the `dvc.lock` commit-back.
10. **Re-establish MLflow registry state** (and decide where it lives) so Phase 06's
    promotion gate has something to promote.

### Open architectural questions

- **Where does MLflow actually live** once training is scheduled? Training runs on Kaggle,
  whose filesystem does not persist between runs, so each run's SQLite db is pulled down
  as an island. A registry spanning runs needs a real home (same free VM as serving? R2
  as artifact store?) before Phase 06's None→Staging→Production gate means anything.
- **How does Prefect trigger a Kaggle GPU run?** Prefect would orchestrate a push to a
  Kaggle kernel and poll it — an awkward indirection. The alternative is accepting manual
  GPU runs and automating only data refresh + evaluation.
- **Drift reference distribution for Phase 09.** The dataset covers 2026-07-01 →
  2026-07-05 only (the 200k-game cap filled in five days), so "monthly drift" isn't what
  this data measures. Decide what the reference window is.
- **Search-augmented tier placement.** PUCT/MCTS over the top-k policy moves is purely
  additive and would also feed Phase 08 commentary — but it is the one thing most likely
  to need ONNX (many evals per move), which was dropped.
- **Metric conflict enforcement.** The plan says gate promotion on SPRT Elo and report
  move-matching descriptively. Nothing in code enforces that yet.
- **Probing claims.** A linear probe shows a concept is *linearly decodable* from
  activations, not that the network *uses* it. The writeup needs to say that plainly;
  decide whether to go further (e.g. activation patching) or scope the claim.

---

*Living document. Regenerate or revise when a phase closes.*
