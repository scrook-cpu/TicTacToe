from __future__ import annotations

import argparse
import csv
import sys
import numpy as np
from typing import TextIO
from ttt.engine import GameSpec, render_2d, play_game
from ttt.policies import RandomPolicy, HeuristicPolicy, HeuristicExplorationPolicy, ForkAwareHeuristicPolicy
from ttt.nn import create_network, train_game, evaluate_policy, save_network, load_network, train_with_checkpoints


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


def run_benchmark(csv_path: str = "training_curve.csv") -> None:
    """
    Train networks (vs Random, Heuristic, ForkAwareHeuristic) and evaluate at checkpoints.
    Writes results to a CSV and prints a summary table.
    """
    checkpoints = [0, 100, 300, 500, 1000, 2000, 5000, 10000]
    num_eval_games = 200
    eval_opponents = {
        "random": RandomPolicy(),
        "heuristic": HeuristicPolicy(),
        "fork_heuristic": ForkAwareHeuristicPolicy(),
    }
    training_runs = [
        ("random", RandomPolicy()),
        ("heuristic", HeuristicExplorationPolicy()),
        ("fork_heuristic", ForkAwareHeuristicPolicy()),
    ]

    all_rows: list[dict] = []

    for games_opp_name, games_opponent in training_runs:
        print(f"\nNN watches Heuristic(X) vs {games_opp_name}(O)  (checkpoints: {checkpoints})")
        network = create_network(hidden_size=32)
        rows = train_with_checkpoints(
            network,
            games_opponent,
            games_opp_name,
            eval_opponents,
            checkpoints,
            num_eval_games=num_eval_games,
        )
        all_rows.extend(rows)
        print(f"  {'games':>6}  {'eval vs':<10}  {'win':>5}  {'draw':>5}  {'loss':>5}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*5}  {'-'*5}  {'-'*5}")
        for row in rows:
            print(
                f"  {row['checkpoint_games']:>6}  {row['eval_opponent']:<10}"
                f"  {row['win_rate']:>4.0%}   {row['draw_rate']:>4.0%}   {row['loss_rate']:>4.0%}"
            )

    # Master CSV — one row per checkpoint/combination, all metadata included
    master_fieldnames = [
        "checkpoint_games", "teacher", "games_opponent", "eval_opponent",
        "wins", "losses", "draws", "win_rate", "draw_rate", "loss_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=master_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nMaster CSV written to {csv_path}")

    # Chart-ready CSVs — one file per combination, columns ordered for Excel stacked bar
    import os
    chart_dir = os.path.dirname(csv_path) or "."
    chart_fieldnames = ["checkpoint_games", "wins", "draws", "losses"]
    combos = {
        (r["games_opponent"], r["eval_opponent"])
        for r in all_rows
    }
    for games_opp, eval_opp in sorted(combos):
        subset = [
            r for r in all_rows
            if r["games_opponent"] == games_opp and r["eval_opponent"] == eval_opp
        ]
        fname = os.path.join(
            chart_dir,
            f"chart_teacher-heuristic_gamesopp-{games_opp}_eval-{eval_opp}.csv",
        )
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=chart_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)
        print(f"Chart CSV written to {fname}")


def run_diagnose(
    scenario: str,
    diag_mode: str,
    load_model: str | None = None,
    num_steps: int = 1,
    learning_rate: float = 0.01,
    teacher: str = "heuristic",
    snapshot_steps: list[int] | None = None,
    out_path: str | None = None,
) -> None:
    """Show how a gradient step shifts network scores on a specific board scenario."""
    from ttt.diagnostics import diagnose, generate_fork_snapshot_png, SCENARIOS

    if scenario not in SCENARIOS:
        print(f"Unknown scenario '{scenario}'. Available: {', '.join(SCENARIOS)}")
        sys.exit(1)

    teacher_policy = ForkAwareHeuristicPolicy() if teacher == "fork_heuristic" else HeuristicPolicy()

    if load_model:
        network = load_network(load_model)
        print(f"Loaded model from {load_model}")
    else:
        network = create_network(hidden_size=32)
        print("Using fresh (random) network")

    if snapshot_steps is not None:
        generate_fork_snapshot_png(
            scenario=scenario,
            network=network,
            teacher_policy=teacher_policy,
            snapshot_steps=snapshot_steps,
            learning_rate=learning_rate,
            out_path=out_path or "fork_snapshot.png",
        )
    else:
        diagnose(scenario, network, mode=diag_mode, num_steps=num_steps,
                 learning_rate=learning_rate, teacher_policy=teacher_policy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tic-Tac-Toe demos and NN training.")
    parser.add_argument("--output", "-o", help="Output file path (demo/train: text; benchmark: CSV).")
    parser.add_argument(
        "--mode",
        choices=("demo", "train", "benchmark", "diagnose"),
        default="demo",
        help=(
            "demo: verbose heuristic-vs-random game | "
            "train: train and evaluate the NN | "
            "benchmark: training-curve experiment → CSV | "
            "diagnose: single-scenario gradient step visualization"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("win_in_one", "block_or_lose", "fork_setup", "empty"),
        default="win_in_one",
        help="Board scenario for --mode diagnose.",
    )
    parser.add_argument(
        "--diag-mode",
        choices=("grid", "scores_weights", "full"),
        default="scores_weights",
        help="Detail level for --mode diagnose.",
    )
    parser.add_argument(
        "--load-model",
        metavar="FILE",
        help="Load saved .npz weights instead of random init (diagnose/train).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of gradient steps for --mode diagnose (default: 1).",
    )
    parser.add_argument(
        "--teacher",
        choices=("heuristic", "fork_heuristic"),
        default="heuristic",
        help="Teacher policy used in --mode diagnose (default: heuristic).",
    )
    parser.add_argument(
        "--snapshot-steps",
        metavar="STEPS",
        help="Comma-separated training steps for fork snapshot (e.g. 0,20,100,500,2000).",
    )
    args = parser.parse_args()

    snapshot_steps = None
    if args.snapshot_steps:
        snapshot_steps = [int(s) for s in args.snapshot_steps.split(",")]

    if args.mode == "benchmark":
        csv_path = args.output or "training_curve.csv"
        run_benchmark(csv_path=csv_path)
    elif args.mode == "diagnose":
        run_diagnose(
            scenario=args.scenario,
            diag_mode=args.diag_mode,
            load_model=args.load_model,
            num_steps=args.steps,
            teacher=args.teacher,
            snapshot_steps=snapshot_steps,
            out_path=args.output,
        )
    else:
        runner = train_nn_demo if args.mode == "train" else play_demo
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                runner(out=f)
            print(f"Output written to {args.output}")
        else:
            runner()


if __name__ == "__main__":
    main()
