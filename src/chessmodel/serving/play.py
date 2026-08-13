"""Interactive terminal play against a trained checkpoint.

Purely for informally trying a checkpoint out -- "does this look like it's
learning anything" -- not the production serving path. Phase 04 will wrap
similar model-loading logic in a proper API; load_model_for_inference is
written so that reuse is straightforward when that happens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch

from chessmodel.data.encode import board_to_tensor, move_to_policy_index
from chessmodel.training.model import ChessNet

_MODEL_SHAPE_KEYS = ("trunk_channels", "num_residual_blocks", "value_channels", "value_hidden")

_RESET = "\033[0m"
_LIGHT_BG = "\033[48;2;240;217;181m"
_DARK_BG = "\033[48;2;181;136;99m"
_WHITE_FG = "\033[1;38;2;255;255;255m"
_BLACK_FG = "\033[1;38;2;20;20;20m"


def _render_board(board: chess.Board) -> str:
    """A chessboard-shaped board, not just text.

    Two deliberate choices, both worth the extra bookkeeping over just
    calling board.unicode():

    - Each rank is drawn as *two* terminal lines, not one. Monospace
      terminal cells are usually taller than they are wide, so a
      single-line-per-rank square reads as flat/squashed; doubling the
      height gets each square closer to visually square.
    - Every cell is built with the same explicit fixed-width padding
      regardless of what glyph goes in it, rather than trusting the glyph's
      own rendered width -- board.unicode() doesn't control this, which is
      what broke grid alignment before.

    Alternating light/dark square colors plus white/black piece coloring
    are layered on top so it reads at a glance, not just technically
    correctly.
    """
    file_labels = "    " + "".join(f" {file}  " for file in "abcdefgh")
    lines = [file_labels]

    for rank in range(7, -1, -1):
        top_cells = []
        bottom_cells = []
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            bg = _LIGHT_BG if (rank + file) % 2 == 1 else _DARK_BG
            top_cells.append(f"{bg}    {_RESET}")
            if piece is None:
                bottom_cells.append(f"{bg}    {_RESET}")
            else:
                fg = _WHITE_FG if piece.color == chess.WHITE else _BLACK_FG
                glyph = piece.unicode_symbol()
                bottom_cells.append(f"{bg}{fg} {glyph}  {_RESET}")
        lines.append("    " + "".join(top_cells))
        lines.append(f" {rank + 1}  " + "".join(bottom_cells))

    lines.append(file_labels)
    return "\n".join(lines)


def load_model_for_inference(checkpoint_path: Path, device: torch.device) -> ChessNet:
    """Reconstruct the exact architecture a checkpoint was trained with, from
    its own saved config, then load its weights.

    Avoids requiring the caller to remember and pass matching
    --trunk-channels/--num-residual-blocks by hand -- save_checkpoint already
    stores the config that produced this checkpoint, so read it back instead
    of trusting the caller to get it right.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_shape = {key: checkpoint["config"][key] for key in _MODEL_SHAPE_KEYS}
    model = ChessNet(**model_shape).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def select_model_move(
    model: ChessNet, board: chess.Board, device: torch.device
) -> tuple[chess.Move, float]:
    """Legal-move-masked argmax over the policy head. Returns (move, self-assessed value)."""
    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(tensor)

    legal_moves = list(board.legal_moves)
    legal_indices = [move_to_policy_index(move) for move in legal_moves]
    legal_logits = policy_logits[0, legal_indices]
    best = int(torch.argmax(legal_logits).item())

    return legal_moves[best], value.item()


def _prompt_user_move(board: chess.Board) -> chess.Move | None:
    """Returns None if the user wants to quit."""
    while True:
        raw = input("Your move (SAN or UCI, 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None

        try:
            return board.parse_san(raw)
        except ValueError:
            pass

        try:
            move = chess.Move.from_uci(raw)
        except ValueError:
            move = None
        if move is not None and move in board.legal_moves:
            return move

        print(f"'{raw}' isn't a legal move here -- try again.")


def play(checkpoint_path: Path, user_plays_white: bool, device_name: str | None) -> None:
    device = (
        torch.device(device_name)
        if device_name
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model_for_inference(checkpoint_path, device)
    board = chess.Board()
    user_color = chess.WHITE if user_plays_white else chess.BLACK

    print(_render_board(board))
    while not board.is_game_over():
        if board.turn == user_color:
            move = _prompt_user_move(board)
            if move is None:
                print("Quitting.")
                return
            board.push(move)
        else:
            move, value = select_model_move(model, board, device)
            print(f"Model plays: {board.san(move)}  (self-assessed value: {value:+.2f})")
            board.push(move)

        print(_render_board(board))

    print(f"Game over: {board.result()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--side",
        choices=["white", "black"],
        default="white",
        help="Which side you play",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    play(args.checkpoint, user_plays_white=args.side == "white", device_name=args.device)


if __name__ == "__main__":
    main()
