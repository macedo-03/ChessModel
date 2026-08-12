# ADR-0001: Input tensor encoding (bitboard spec)

**Status:** Accepted
**Date:** 2026-08-11

## Context

The policy/value network (Phase 02) and everything downstream of it — ONNX export,
the concept probes (Phase 03), self-play (Phase 07) — depends on a fixed, stable
input tensor shape. Deciding this after the Phase 01 encoder is written means
re-encoding the whole dataset and retraining. Fixing it here, before any encoder
code exists, avoids that.

This ADR covers only the **board input tensor** (what goes into the network).
The policy *output* encoding (the 8×8×73 move-plane action space, including how
promotions — auto-queen vs. underpromotion — map to move indices) is a separate
concern, deferred to a future ADR (ADR-0003 onward — ADR-0002 is the prior-attempt
pitfalls checklist) when Phase 01/02 need it.

## Decision

Each position is encoded as a single `8 × 8 × 19` `float32` tensor, planes stacked
in this fixed order:

| Planes | Count | Content |
|---|---|---|
| 0–5 | 6 | White pieces, one plane per type: P, N, B, R, Q, K |
| 6–11 | 6 | Black pieces, one plane per type: p, n, b, r, q, k |
| 12 | 1 | Side to move (all-1 if white to move, all-0 if black) |
| 13–16 | 4 | Castling rights: white king-side, white queen-side, black king-side, black queen-side (each plane all-1 or all-0) |
| 17 | 1 | En-passant target square (1.0 at the target square's cell, 0.0 elsewhere; all-zero if none) |
| 18 | 1 | Halfmove clock, normalized to `[0, 1]` via `min(halfmove_clock, 100) / 100` (broadcast across the plane) |

Board orientation: rank 8 → row 0, file a → column 0, regardless of side to move
(the network learns color symmetry from the side-to-move plane rather than the
board being flipped — this keeps the encoder a pure, stateless function of the
FEN with no perspective-dependent branching).

A1: piece planes use algebraic square indexing consistent with `python-chess`
(`chess.SQUARES`, `chess.square_rank`, `chess.square_file`) so the encoder can be
implemented directly against `Board.piece_map()` without a custom coordinate
system.

## Rationale

Think of the 19 planes as 19 board-sized transparency sheets stacked on top of
each other and fed to the network's first conv layer as input channels
(`in_channels=19`). Planes 0–11 and 17 are genuinely spatial — a 1 sits at the
specific square the fact applies to (a piece, or the en passant target square).
Planes 12, 13–16, and 18 are **broadcast planes**: the fact they encode isn't
tied to a square (whose turn it is, castling rights, the halfmove clock), so
the entire 8×8 grid just repeats the same value. This is the standard way to
feed a scalar fact into a conv net without a separate input pathway — the
convolution treats a broadcast plane exactly like any other spatial channel.

Two choices are worth calling out explicitly:

- **Fixed board orientation, no flipping to the mover's perspective.** Rank
  8/file a is always row 0/column 0, whichever side is to move — the network
  learns "whose turn it is" from plane 12 rather than always seeing the board
  from its own side. The alternative (flip the board so the mover's pieces are
  always at the bottom) is common in other implementations, but it makes the
  encoder stateful — a perspective-dependent branch that's easy to get subtly
  wrong and harder to unit-test. A pure function of the FEN (same FEN in, same
  tensor out, no hidden branching) was worth more than the flip's minor
  inductive-bias benefit.
- **No move-history stack.** Real AlphaZero encodes the last 8 board states
  (~119 planes total) so the network can detect repetition itself. That's cut
  here — one halfmove-clock plane instead, with `python-chess`'s own
  repetition/50-move logic doing the actual draw detection explicitly in the
  bot layer (Phase 05) rather than asking the network to infer it from history.
  ~6x smaller input and a much simpler encoder, at the cost of the network
  losing built-in repetition-awareness.

For context: a prior personal attempt at this project (see
[ADR-0002](0002-prior-attempt-pitfalls-checklist.md)) used a 16-plane version of
roughly the same idea — 12 piece planes, but only 2 *combined* castling planes
(one per color, not split kingside/queenside), 1 turn plane, 1 en passant
plane, and no halfmove-clock plane at all. The 13–16 split here and the added
plane 18 are direct refinements over that.

## Consequences

- Fixed at 19 planes now; if a future phase needs move-count-since-start or
  repetition-count as an explicit input (e.g. to help the value head reason about
  repetition draws), that is a new ADR and a re-encode of the dataset, not a
  silent change here.
- The halfmove clock plane gives the network a signal for approaching the 50-move
  rule without needing repetition history in the input.
- Any change to this table invalidates every DVC-tracked processed shard,
  every trained checkpoint, and the ONNX export — treat it as a breaking change
  requiring a new dataset version and a new ADR, not an edit to this one.

## Alternatives considered

- **AlphaZero's full 8-step history stack** (~119 planes): rejected for this
  project's scope — it triples-plus the encoder/training complexity for a
  benefit (repetition/history awareness) that the halfmove-clock plane and
  explicit draw-rule handling in the bot layer (Phase 05) already cover well
  enough for a portfolio-scale project.
- **Flipping the board to the mover's perspective** instead of a side-to-move
  plane: rejected to keep the encoder stateless and simpler to unit-test
  (no perspective-dependent branch to get wrong).
