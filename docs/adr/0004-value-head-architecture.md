# ADR-0004: Value head architecture

**Status:** Accepted
**Date:** 2026-08-12

## Context

Phase 02's task list specifies "value head (conv → FC → tanh)" at a one-line
level — enough to know the shape of the idea, not enough to write
`model.py` against. Unlike the policy head (ADR-0003), the value head's
design wasn't forced by an encoding decision already made elsewhere, so it
was left open. It needs to be pinned down now, for the same reason ADR-0001
and ADR-0003 were written before their respective code: changing a network's
output-layer shape later invalidates any checkpoint trained against the old
one.

## Decision

Both heads read from the same ResNet trunk output: `(C, 8, 8)`, where
`C` = 128 channels, 8 residual blocks (the smaller/cheaper end of Phase 02's
stated 6–10 blocks / 128–256 channels range — chosen for Kaggle free-tier T4
budget, not because the range's upper end is wrong).

**Value head:**

```
trunk output (128, 8, 8)
  -> Conv2d(128, 8, kernel_size=1) + BatchNorm2d(8) + ReLU   # (8, 8, 8)
  -> flatten                                                  # (512,)
  -> Linear(512, 256) + ReLU
  -> Linear(256, 1)
  -> tanh                                                      # scalar in [-1, 1]
```

The 1×1 conv reduces 128 channels down to 8 before flattening — a global
position judgment doesn't need the trunk's full channel width the way the
policy head's spatial output does, so this keeps the head's parameter count
small relative to the trunk.

**Policy head** (restated here for contrast, unchanged from ADR-0003):

```
trunk output (128, 8, 8)
  -> Conv2d(128, 128, kernel_size=3, padding=1) + BatchNorm2d(128) + ReLU
  -> Conv2d(128, 73, kernel_size=1)                           # (73, 8, 8), logits
```

No flatten-to-FC step in the policy head — it stays a pure conv output,
per ADR-0003's whole rationale (weight-sharing across squares). This is the
point of contrast worth naming explicitly: the value head collapses spatial
structure on purpose (a position's value isn't "per square"), the policy
head preserves it on purpose (a move's identity *is* per square). Same trunk,
deliberately different treatment after it.

## Rationale

This mirrors AlphaZero's published value head shape (conv channel-reduction
→ FC → FC → tanh), scaled down for this project's compute budget — AlphaZero
reduces to 1 channel before its FC layers; 8 is used here as a slightly
richer bottleneck, since this project's trunk is already far smaller (8
blocks × 128 channels vs. AlphaZero's 19–40 blocks × 256 channels) and can
afford a marginally wider value head without meaningfully changing the
compute budget.

Two FC layers (512→256→1) rather than one (512→1) directly: a single linear
layer can only express a linear function of the pooled features: enough
capacity for the value head to combine "material feels imbalanced" and
"king safety feels off" into one number benefits from at least one
nonlinearity in between, at negligible parameter cost.

## Consequences

- Fixed output shapes: policy `(73, 8, 8)` (4,672 flat), value scalar in
  `[-1, 1]`. Changing either invalidates every checkpoint trained against
  this architecture — same severity as an ADR-0001/0003 change.
- **The policy head's raw output is channels-first** (`(N, 73, 8, 8)` in
  PyTorch's convention). Flattening it directly (`.view(N, -1)`) would
  produce index order `plane*64 + row*8 + col` — but ADR-0003's established
  formula is `(row*8 + col)*73 + plane`. These are *not* the same ordering.
  `model.py` must explicitly permute to `(N, 8, 8, 73)` before flattening, or
  every policy target from `move_to_policy_index` would silently point at
  the wrong logit — the network would still train (loss still decreases
  against whatever mapping it's given), but every move decoded at inference
  via `policy_index_to_move` would be wrong. This is exactly the kind of bug
  that doesn't announce itself in a loss curve, so it gets an explicit,
  dedicated test rather than being trusted to "look right."

## Alternatives considered

- **Global average pooling instead of a 1×1 conv + flatten**: rejected —
  pooling to `(128,)` before the FC layers is more parameter-efficient, but
  throws away *where* on the board features are strong, which for a value
  judgment (e.g., "is the king-side specifically weak") is informative. The
  1×1-conv-then-flatten keeps a coarse notion of position.
- **Single FC layer (512→1) directly**: rejected — see Rationale; the
  nonlinearity is cheap and the pure-linear version measurably underfits in
  the published AlphaZero-style architectures this is modeled on.
