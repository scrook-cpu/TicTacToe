from __future__ import annotations

import argparse
import sys
import numpy as np
from typing import TextIO
from ttt.engine import GameSpec, render_2d, play_game
from ttt.policies import RandomPolicy, HeuristicPolicy, HeuristicExplorationPolicy
from ttt.nn import create_network, train_game, evaluate_policy, save_network


def format_scores(scores: np.ndarray, dims: tuple[int, ...]) -> str:
    if len(dims) == 2:
        # The engine uses a column-major flat index mapping internally,
        # so reshape with order='F' to align score output with render_2d.
        return np.array2string(scores.reshape(dims, order='F'), formatter={'float_kind': lambda x: f"{x:6.1f}"})
    return np.array2string(scores, formatter={'float_kind': lambda x: f"{x:6.1f}"})


def print_move_record(record: dict, spec: GameSpec, out: TextIO) -> None:
    player = record["player"]
    policy_name = record["policy_name"]
    scores = np.asarray(record["scores"], dtype=float)
    board_after = record["board_after"]

    print(f"\nTurn {record['turn']}: {'X' if player == +1 else 'O'} ({policy_name})", file=out)
    print("Scores:", file=out)
    print(format_scores(scores, spec.dims), file=out)
    print(f"{'X' if player == +1 else 'O'} plays {record['move']}", file=out)
    print(render_2d(board_after, spec.dims), file=out)


def play_demo(out: TextIO = None):
    spec = GameSpec.make(dims=(3, 3), k=3)
    pX = HeuristicPolicy()
    pO = RandomPolicy()
    if out is None:
        out = sys.stdout

    winner_id = play_game(
        pX,
        pO,
        spec,
        move_callback=lambda record: print_move_record(record, spec, out),
    )

    if winner_id == 0:
        print("\nDraw.", file=out)
    else:
        print("\nWinner:", "X" if winner_id == +1 else "O", file=out)


def train_nn_demo(out: TextIO = None) -> None:
    """
    Train the neural network to imitate HeuristicPolicy moves, then evaluate it.
    """
    if out is None:
        out = sys.stdout

    network = create_network(hidden_size=32)

    print("Training NN by imitating HeuristicPolicy vs HeuristicPolicy...", file=out)
    train_game(network, HeuristicExplorationPolicy(), num_games=10000)

    print("\nEvaluating trained NN vs RandomPolicy...", file=out)
    random_stats = evaluate_policy(network, RandomPolicy(), num_games=100)
    print(random_stats, file=out)

    print("\nEvaluating trained NN vs HeuristicPolicy...", file=out)
    heuristic_stats = evaluate_policy(network, HeuristicPolicy(), num_games=100)
    print(heuristic_stats, file=out)

    save_network(network, "model.npz")
    print("\nSaved trained network to model.npz", file=out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tic-Tac-Toe demos and NN training.")
    parser.add_argument("--output", "-o", help="Path to write game output to.")
    parser.add_argument(
        "--mode",
        choices=("demo", "train"),
        default="demo",
        help="Run a verbose heuristic-vs-random demo or train/evaluate the neural network.",
    )
    args = parser.parse_args()

    runner = train_nn_demo if args.mode == "train" else play_demo

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            runner(out=f)
        print(f"Output written to {args.output}")
    else:
        runner()


if __name__ == "__main__":
    main()
