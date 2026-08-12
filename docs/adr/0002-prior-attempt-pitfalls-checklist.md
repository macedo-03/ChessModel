# ADR-0002: Pitfalls checklist from the prior attempt

**Status:** Accepted
**Date:** 2026-08-11

## Context

Before starting Phase 01/02, reviewed a prior personal attempt at this same
project (`ChessBot`, a separate repo at
`C:\Users\José Macedo\Documents\MEI\1ano\EXTRA\ChessBot`, built Nov 2024–Jan
2025, then abandoned — not part of this repo's history). That codebase reached
real milestones — 200k supervised positions trained in chunks, a full puzzle
training pass, and 20 self-play cycles of 500 games each — but had several
concrete bugs and anti-patterns that plausibly explain why it stalled and why
picking it back up wasn't worth it (see chat history 2026-08-11 for the full
review). This ADR turns that review into a checklist to verify against during
implementation, so the same mistakes aren't repeated silently.

No code was carried over from that project. This ADR exists purely to make the
lessons durable.

## Checklist

- [ ] **Value target uses the mover's perspective, not an absolute white-perspective.**
  Prior bug: the value label was fixed to the absolute game outcome (+1 white
  win / -1 black win / 0 draw) and reused unchanged for every ply, regardless of
  whose turn it was at that position — a contradictory training signal roughly
  half the time. Check: when generating `(position, value)` pairs in Phase
  01/02, confirm the value is sign-flipped by side-to-move, not just the raw
  final result. [ADR-0001](0001-bitboard-tensor-spec.md)'s tensor already
  encodes side-to-move (plane 12) precisely so this stays checkable.

- [ ] **Any evaluation attached to a position reflects that exact position, not
  a stale earlier one.** Prior bug: puzzle preprocessing computed one Stockfish
  evaluation for a puzzle's *starting* FEN and reused it as the label for every
  subsequent ply in the solution line. Check: label per position, not per
  game/puzzle.

- [ ] **Dataset encoding is lazy/streamed, never fully materialized in RAM.**
  Prior bug: the PGN→tensor conversion eagerly built a full Python list of
  `(16,8,8)` float32 tensors for every ply of every game in a chunk — no
  memory-mapping, no lazy per-item encoding. Almost certainly the direct cause
  of the Kaggle dataset becoming unmanageable and getting deleted locally.
  Check: Phase 01's pipeline stores compact representations (PGN/move lists,
  not expanded tensors) and encodes to tensor on-the-fly in the DataLoader's
  `__getitem__`.

- [ ] **Move selection at inference isn't silently policy-argmax-only if a
  value head exists.** Prior bug: the value head was computed at inference but
  never actually used to pick a move — only a legality-masked policy argmax,
  with the value just printed. If Phase 04 ships without search, that should be
  a stated MVP scope decision (it is — see the Search-augmented tier in the
  scope ladder), not an oversight discovered later.

- [ ] **No duplicate implementations of the same conversion logic.** Prior repo
  had two independent implementations of PGN→tensor conversion, and two
  near-duplicate self-play scripts with diverging checkpoint-naming schemes.
  Check: one canonical encoder, imported everywhere, covered by the Phase 0
  test harness.

- [ ] **Every call site matches the signature it calls.** Prior repo had a
  script calling a model constructor with a keyword argument that didn't exist
  on the class — it would crash if run. Check: this is exactly what mypy + CI
  (already wired up in Phase 0) exist to catch; don't let ad hoc scripts drift
  from the APIs they call untested.

- [ ] **Every training run is tracked in MLflow, not just filename-encoded.**
  Prior repo saved checkpoints as e.g. `phase_5_model_after_chunk_20_ep_15.pt`
  — decodable but not queryable, no structured record of hyperparameters or
  metrics beyond what's in the filename or a manually maintained roadmap doc.
  Check: covered by design in Phase 02's MLflow logging requirement.

- [ ] **The dataset itself is versioned (DVC), not just a local CSV.** Prior
  repo's Kaggle download was deleted locally with no remote copy, losing it
  entirely. Check: covered by design in Phase 01's DVC + R2 remote requirement.

## Consequences

This is a verification checklist for Phase 01/02 implementation and review, not
a design decision in itself — it doesn't block anything, but each item should
have a concrete "here's how this build avoids it" answer before Phase 02 is
considered done.
