from __future__ import annotations

import argparse
import sys
import numpy as np
from typing import TextIO
from ttt.engine import GameSpec, new_board, apply_move, winner, is_draw, render_2d
from ttt.policies import RandomPolicy, HeuristicPolicy, choose_move


def format_scores(scores: np.ndarray, dims: tuple[int, ...]) -> str:
    if len(dims) == 2:
        # The engine uses a column-major flat index mapping internally,
        # so reshape with order='F' to align score output with render_2d.
        return np.array2string(scores.reshape(dims, order='F'), formatter={'float_kind': lambda x: f"{x:6.1f}"})
    return np.array2string(scores, formatter={'float_kind': lambda x: f"{x:6.1f}"})


def choose_move_from_scores(scores: np.ndarray, board: list[int]) -> int:
    masked = scores.astype(float).copy()
    for i, v in enumerate(board):
        if v != 0:
            masked[i] = -np.inf
    return int(np.argmax(masked))


def play_game(pX, pO, spec: GameSpec, verbose: bool = False, out: TextIO = None) -> int:
    if out is None:
        out = sys.stdout

    board = new_board(spec)
    player = +1
    turn = 0

    while True:
        turn += 1
        policy = pX if player == +1 else pO
        policy_name = policy.__class__.__name__
        if verbose:
            scores = policy.scores(board, player, spec)
            print(f"\nTurn {turn}: {'X' if player == +1 else 'O'} ({policy_name})", file=out)
            print("Scores:", file=out)
            print(format_scores(scores, spec.dims), file=out)
            idx = choose_move_from_scores(scores, board)
        else:
            idx = choose_move(policy, board, player, spec)

        board = apply_move(board, idx, player)
        if verbose:
            print(f"{'X' if player == +1 else 'O'} plays {idx}", file=out)
            print(render_2d(board, spec.dims), file=out)

        w = winner(board, spec)
        if w is not None:
            if verbose:
                print("\nWinner:", "X" if w == +1 else "O", file=out)
            return w
        if is_draw(board, spec):
            if verbose:
                print("\nDraw.", file=out)
            return 0
        player *= -1


def play_demo(out: TextIO = None):
    spec = GameSpec.make(dims=(3, 3), k=3)
    pX = HeuristicPolicy()
    pO = RandomPolicy()
    if out is None:
        print("Playing Heuristic X vs Random O with verbose move-by-move output.")
    else:
        print("Playing Heuristic X vs Random O with verbose move-by-move output.", file=out)
    play_game(pX, pO, spec, verbose=True, out=out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Tic-Tac-Toe demo game.")
    parser.add_argument("--output", "-o", help="Path to write game output to.")
    args = parser.parse_args()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            play_demo(out=f)
        print(f"Game output written to {args.output}")
    else:
        play_demo()


if __name__ == "__main__":
    main()
