"""Tests for ChessNet: shapes, the policy-flatten/ADR-0003 index alignment, and
a single-batch overfit sanity check that the whole thing actually learns."""

import chess
import torch

from chessmodel.data.encode import (
    NUM_INPUT_PLANES,
    NUM_POLICY_PLANES,
    board_to_tensor,
    move_to_policy_index,
)
from chessmodel.training.model import ChessNet, flatten_policy_logits


def _tiny_model() -> ChessNet:
    # Small enough to run instantly on CPU in a test.
    return ChessNet(trunk_channels=16, num_residual_blocks=2, value_channels=4, value_hidden=32)


def test_forward_shapes() -> None:
    model = _tiny_model()
    batch = torch.randn(4, NUM_INPUT_PLANES, 8, 8)

    policy_logits, value = model(batch)

    assert policy_logits.shape == (4, 8 * 8 * NUM_POLICY_PLANES)
    assert value.shape == (4,)


def test_value_output_is_bounded_by_tanh() -> None:
    model = _tiny_model()
    batch = torch.randn(4, NUM_INPUT_PLANES, 8, 8)

    _, value = model(batch)

    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_forward_on_real_encoded_position() -> None:
    model = _tiny_model().eval()  # eval mode: BatchNorm uses running stats, batch size 1 is fine
    tensor = torch.from_numpy(board_to_tensor(chess.Board())).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = model(tensor)

    assert policy_logits.shape == (1, 8 * 8 * NUM_POLICY_PLANES)
    assert not torch.isnan(policy_logits).any()
    assert not torch.isnan(value).any()


def test_flatten_policy_logits_matches_adr0003_index_formula() -> None:
    """The one place a silent, self-consistent-but-wrong bug could hide.

    Build a raw (N, 73, 8, 8) channels-first tensor with a single 1.0 marker
    at a known (plane, row, col), and confirm the flattened index lands
    exactly where ADR-0003 says a move at that (row, col) with that plane
    should be: (row*8 + col)*73 + plane.
    """
    logits = torch.zeros(1, NUM_POLICY_PLANES, 8, 8)

    cases = [
        (0, 0, 0),  # origin square a8, plane 0
        (5, 3, 2),  # plane 5, row 3, col 2
        (72, 7, 7),  # last plane, square h1
        (56, 4, 4),  # first knight-move plane, center square
    ]
    for plane, row, col in cases:
        logits.zero_()
        logits[0, plane, row, col] = 1.0

        flat = flatten_policy_logits(logits)
        expected_index = (row * 8 + col) * NUM_POLICY_PLANES + plane

        assert flat[0, expected_index] == 1.0
        assert flat[0].sum() == 1.0  # marker landed in exactly one slot


def test_flatten_policy_logits_agrees_with_move_to_policy_index() -> None:
    """End-to-end version of the same check, via a real move rather than raw indices."""
    move = chess.Move.from_uci("e2e4")
    expected_index = move_to_policy_index(move)  # 3797, per ADR-0003's worked example

    logits = torch.zeros(1, NUM_POLICY_PLANES, 8, 8)
    # e2e4: origin row 6, col 4; plane 1 (direction N, distance 2) -- see ADR-0003.
    logits[0, 1, 6, 4] = 1.0

    flat = flatten_policy_logits(logits)
    assert flat[0, expected_index] == 1.0


def test_overfits_a_single_tiny_batch() -> None:
    """Sanity check that the model can actually learn, not just run.

    Not a real training run -- a handful of random board tensors with fixed
    random targets, trained for enough steps that a correctly-wired model
    should drive the loss to near zero. If this fails, something structural
    is broken (e.g. the flatten bug ADR-0004 warns about), regardless of
    whether the shape tests above pass.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    model.train()

    batch_size = 8
    boards = torch.randn(batch_size, NUM_INPUT_PLANES, 8, 8)
    policy_targets = torch.randint(0, 8 * 8 * NUM_POLICY_PLANES, (batch_size,))
    value_targets = torch.empty(batch_size).uniform_(-1, 1)

    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(200):
        policy_logits, value = model(boards)
        loss = policy_loss_fn(policy_logits, policy_targets) + value_loss_fn(value, value_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.05  # loss collapsed by at least 20x
