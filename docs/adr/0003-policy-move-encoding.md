# ADR-0003: Policy output encoding (move-plane action space)

**Status:** Accepted
**Date:** 2026-08-11

## Context

[ADR-0001](0001-bitboard-tensor-spec.md) fixed the board *input* tensor and
explicitly deferred the policy *output* encoding — the fixed-size discrete
action space the network's policy head predicts over, and the label paired
with each board tensor for the training loss. Phase 01's encoder needs this to
produce `(board_tensor, policy_target, value_target)` triples; Phase 02 needs
it to size the policy head's final layer; Phase 04/07 need the inverse mapping
to turn a predicted index back into a `chess.Move`.

This has to be fixed before the Phase 01 encoder is written, for the same
reason as ADR-0001: changing it later means re-encoding the dataset and
retraining, not editing a function.

## Decision

Follow the AlphaZero move-plane scheme: the policy head outputs an
**8 × 8 × 73** tensor — 73 "move-type" planes per origin square, flattened to
a single **4,672**-way (`8×8×73`) classification target for the loss.

Board orientation matches ADR-0001 exactly: row 0 = rank 8, column 0 = file a,
absolute and geometric — **not** flipped by side to move. Same coordinate
system for both the input and output tensors.

The 73 planes, indexed 0–72:

| Planes | Count | Move type |
|---|---|---|
| 0–55 | 56 | "Queen-type" moves: 8 compass directions × 7 distances (1–7 squares). Covers rook/bishop/queen slides, one-square king moves, pawn single/double pushes and captures, castling (see Consequences), and **auto-queen promotion**. |
| 56–63 | 8 | Knight jumps — the 8 possible L-shaped (Δrow, Δcol) offsets. |
| 64–72 | 9 | **Underpromotion**: 3 relative directions (Δcol ∈ {−1, 0, +1}) × 3 alternate pieces (knight, bishop, rook). Only used when `move.promotion` is knight/bishop/rook — queen promotion uses the queen-type planes above, not these. |

Final index = `(origin_row * 8 + origin_col) * 73 + plane_index`, an integer
in `[0, 4671]`, used directly as the `CrossEntropyLoss` target class.

Direction/plane assignment is purely geometric (based on absolute Δrow/Δcol on
the fixed board), never color-relative — consistent with ADR-0001's no-flip
decision. The network infers "which way is forward" from the side-to-move
input plane (ADR-0001, plane 12), not from the action-space geometry itself.

## Rationale

**Why 4,672 classes via 73 planes, rather than a flat `(from×to×promotion)`
scheme?** A prior attempt at this project (see
[ADR-0002](0002-prior-attempt-pitfalls-checklist.md)) used exactly that flat
alternative: `start_square*64 + end_square + promotion_bucket*4096`, with 5
promotion buckets (none/knight/bishop/rook/queen) → 20,480 classes. Two
reasons to prefer the plane scheme instead:

- **Size.** 4,672 vs. 20,480 — about 4.4x smaller output layer, which matters
  at this project's compute budget.
- **Spatial structure.** The policy head is a conv layer producing a
  `(73, 8, 8)` output, not a dense FC layer over a flat 20,480-way softmax
  (which is what the flat scheme requires, and is what the prior attempt
  actually did — see `fc_policy` in that codebase). Each of the 73 planes has
  a *fixed geometric meaning* at every square, so convolutional weight-sharing
  transfers "how a rook moves" learned at one square to every other square for
  free. The flat scheme has to learn 4,096 independent from→to associations
  with no shared structure between them. This is exactly why the Phase 02
  task list already specifies a conv policy head rather than a dense one.

**Why fold auto-queen into the queen-type planes instead of giving it its own
bucket, the way the prior attempt did?** A queen promotion moves exactly like
a queen to its destination square — no information is needed beyond "which
square did the pawn land on," which the queen-type planes already encode.
Giving it a separate bucket, as the flat scheme does, is redundant. Only
underpromotion (knight/bishop/rook) needs dedicated plane real estate, since
those pieces move fundamentally differently from a queen and can't be inferred
from the destination square alone.

**Masking is still required regardless of encoding.** This scheme produces
geometrically-defined logits for plenty of moves that are illegal in a given
position (blocked slides, wrong piece at the origin square, etc.). Phase 04/07
still need to restrict the softmax to `board.legal_moves` at inference — the
same pattern the prior attempt's `predict_move` already got right. This ADR
doesn't remove that requirement, it just fixes what the *legal* moves are
indexed as.

## Consequences

- `move_to_policy_index(move: chess.Move) -> int` and its inverse
  `policy_index_to_move(index: int, board: chess.Board) -> chess.Move` become
  one canonical function pair, covered by the Phase 0 test harness — every
  legal move across a large sample of positions should round-trip through
  encode→decode unchanged. This is a direct, concrete answer to ADR-0002's
  "no duplicate implementations" and "every call site matches what's defined"
  checklist items.
- Fixed at 4,672 outputs. Changing this later invalidates the policy head's
  final layer shape and any registered checkpoint — same severity as an
  ADR-0001 change, not a quick edit.
- **Castling needs no special case.** `python-chess` represents castling as a
  two-square king move (e.g. `e1g1` for white kingside) — geometrically a
  queen-type move (Δrow=0, Δcol=2, "east," distance 2), so it falls naturally
  into the queen-type planes. One less edge case to hand-code.

## Alternatives considered

- **Prior attempt's flat `(from×64 + to + promotion_bucket×4096)` scheme**:
  rejected — larger and spatially unstructured output, as discussed above.
- **A separate underpromotion classifier** (4,096-way move index + a small
  auxiliary 4-way promotion head) instead of folding underpromotion into the
  plane scheme: rejected as unnecessary complexity. Nine extra planes is cheap
  and keeps policy prediction as one tensor and one loss term, rather than two
  coupled heads to keep in sync.
