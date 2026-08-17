import copy
import numpy as np
from typing import Dict, Any, List

from .engine import GameSpec, render_2d, legal_moves, ravel_index
from .policies import HeuristicPolicy
from .nn import forward_pass, backpropagate

# The engine uses column-major flat indexing: flat = row + rows*col
# Visual layout of flat indices on a 3×3 board:
#  0 | 3 | 6
# -----------
#  1 | 4 | 7
# -----------
#  2 | 5 | 8

_DIMS = (3, 3)


def _flat(row: int, col: int) -> int:
    return ravel_index((row, col), _DIMS)


def _make_board(**pieces: int) -> List[int]:
    """Build a board from {flat_index: player_value} kwargs."""
    board = [0] * 9
    for idx, val in pieces.items():
        board[int(idx)] = val
    return board


SCENARIOS: Dict[str, List[int]] = {
    # win_in_one: X has left column top+middle; wins by playing bottom-left (flat 2)
    # Visual:  X | . | .
    #          X | . | .
    #          _ | . | O
    "win_in_one": _make_board(**{
        str(_flat(0, 0)): +1,
        str(_flat(1, 0)): +1,
        str(_flat(2, 2)): -1,
    }),

    # block_or_lose: O has right column top+middle; X must block bottom-right (flat 8)
    # Visual:  X | . | O
    #          . | . | O
    #          . | . | _
    "block_or_lose": _make_board(**{
        str(_flat(0, 0)): +1,
        str(_flat(0, 2)): -1,
        str(_flat(1, 2)): -1,
    }),

    # fork_setup: X at top-left + center; O at bottom-right + top-middle.
    # It is X's turn (2X, 2O). O at top-middle blocks row 0 and col 1 but
    # has no immediate win threat. X can fork by playing mid-left (flat 1):
    # that creates two simultaneous threats — col 0 (X at 0,1) and row 1
    # (X at 1,4) — that O cannot both block.
    # Visual:  X | O | .
    #          _ | X | .
    #          . | . | O
    "fork_setup": _make_board(**{
        str(_flat(0, 0)): +1,
        str(_flat(1, 1)): +1,
        str(_flat(2, 2)): -1,
        str(_flat(0, 1)): -1,
    }),

    "empty": [0] * 9,
}


def _score_grid(board: List[int], scores: np.ndarray) -> str:
    """
    Format move scores as a 3×3 grid aligned with render_2d output.
    Uses the same column-major flat index as the engine.
    """
    rows = []
    for r in range(3):
        cells = []
        for c in range(3):
            idx = ravel_index((r, c), _DIMS)
            if board[idx] == +1:
                cells.append("  X   ")
            elif board[idx] == -1:
                cells.append("  O   ")
            else:
                cells.append(f"{scores[idx]:+6.2f}")
        rows.append(" | ".join(cells))
        if r < 2:
            rows.append("-" * 26)
    return "\n".join(rows)


def _weight_delta_summary(w_before: np.ndarray, w_after: np.ndarray, label: str, n: int = 8) -> str:
    delta = w_after - w_before
    flat = delta.ravel()
    top_idx = np.argsort(np.abs(flat))[::-1][:n]
    lines = [f"  {label}  top {n} weight changes:"]
    for i in top_idx:
        r, c = np.unravel_index(i, delta.shape)
        lines.append(
            f"    [{r:2d},{c:2d}]  {w_before.ravel()[i]:+.4f} → {w_after.ravel()[i]:+.4f}"
            f"  (Δ {flat[i]:+.4f})"
        )
    return "\n".join(lines)


def diagnose(
    scenario: str,
    network: Dict[str, Any],
    mode: str = "scores_weights",
    player: int = 1,
    learning_rate: float = 0.01,
    num_steps: int = 1,
    teacher_policy=None,
) -> None:
    """
    Show how gradient steps on a specific board scenario nudge network scores.

    mode:
        "grid"           – score grid before/after only
        "scores_weights" – score grid + top weight deltas (default)
        "full"           – score grid + weight deltas + full W1/W2 matrices

    teacher_policy: policy whose recommended move is used as training target.
        Defaults to HeuristicPolicy().
    """
    if teacher_policy is None:
        teacher_policy = HeuristicPolicy()

    spec = GameSpec.make(dims=(3, 3), k=3)
    board = list(SCENARIOS[scenario])
    player_label = "X" if player == +1 else "O"

    print(f"\n{'='*50}")
    print(f"Scenario: {scenario}   Player: {player_label}   Mode: {mode}")
    print(f"{'='*50}")
    print(render_2d(board, spec.dims))

    # Normalize board to current-player perspective before passing to NN
    board_for_nn = np.asarray(board, dtype=float) * player

    # Snapshot weights before gradient step
    net_before = copy.deepcopy(network)
    scores_before = forward_pass(board_for_nn, network)

    print(f"\n--- Scores BEFORE gradient step ---")
    print(_score_grid(board, scores_before))

    # Determine teacher's recommended move
    teacher_scores = teacher_policy.scores(board, player, spec)
    legal = legal_moves(board)
    teacher_move = max(legal, key=lambda i: teacher_scores[i])
    print(f"\n{teacher_policy.__class__.__name__} recommends: position {teacher_move}  "
          f"(teacher score {teacher_scores[teacher_move]:+.2f})")

    # Apply gradient steps
    for _ in range(num_steps):
        backpropagate(board_for_nn, teacher_move, 1.0, network, learning_rate=learning_rate)

    scores_after = forward_pass(board_for_nn, network)
    delta_at_move = scores_after[teacher_move] - scores_before[teacher_move]

    print(f"\n--- Scores AFTER {num_steps} gradient step{'s' if num_steps > 1 else ''} ---")
    print(_score_grid(board, scores_after))
    print(f"\nPosition {teacher_move} score:  {scores_before[teacher_move]:+.4f} → "
          f"{scores_after[teacher_move]:+.4f}   (Δ {delta_at_move:+.4f})")

    if mode in ("scores_weights", "full"):
        print()
        print(_weight_delta_summary(net_before["W1"], network["W1"], "W1"))
        print()
        print(_weight_delta_summary(net_before["W2"], network["W2"], "W2"))

    if mode == "full":
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        print("\n--- Full W1 before ---")
        print(net_before["W1"])
        print("\n--- Full W1 after  ---")
        print(network["W1"])
        print("\n--- Full W2 before ---")
        print(net_before["W2"])
        print("\n--- Full W2 after  ---")
        print(network["W2"])
        np.set_printoptions()  # reset to defaults


def _draw_score_board_on_ax(ax, board, scores, target_square, player, title):
    """Render a board with NN scores on empty squares onto a matplotlib axes."""
    import matplotlib.patches as mpatches
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.55, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    for i in range(1, 3):
        ax.plot([i, i], [0, 3], color="#333", lw=2, zorder=3)
        ax.plot([0, 3], [i, i], color="#333", lw=2, zorder=3)

    for r in range(3):
        for c in range(3):
            idx = ravel_index((r, c), _DIMS)
            cx, cy = c + 0.5, (2 - r) + 0.5

            if idx == target_square:
                ax.add_patch(mpatches.Rectangle(
                    (c, 2 - r), 1, 1, color="#FFF9C4", zorder=0))
                ax.add_patch(mpatches.Rectangle(
                    (c, 2 - r), 1, 1, fill=False,
                    edgecolor="#F57F17", lw=2.5, zorder=4))

            val = board[idx]
            if val == player:
                ax.text(cx, cy, "X", ha="center", va="center",
                        fontsize=22, fontweight="bold", color="#1565C0", zorder=5)
            elif val == -player:
                ax.text(cx, cy, "O", ha="center", va="center",
                        fontsize=22, fontweight="bold", color="#C62828", zorder=5)
            else:
                color  = "#E65100" if idx == target_square else "#555"
                weight = "bold"    if idx == target_square else "normal"
                ax.text(cx, cy, f"{scores[idx]:+.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight=weight, zorder=5)

    ax.text(1.5, -0.38,
            f"fork sq: {scores[target_square]:+.3f}",
            ha="center", va="top", fontsize=8.5,
            color="#E65100", fontweight="bold")


def generate_fork_snapshot_png(
    scenario: str,
    network: Dict[str, Any],
    teacher_policy=None,
    snapshot_steps: List[int] = None,
    learning_rate: float = 0.05,
    out_path: str = "fork_snapshot.png",
    player: int = 1,
) -> None:
    """
    Train on a single board scenario and capture score-grid snapshots at
    each step in snapshot_steps.  Saves a side-by-side PNG showing how the
    NN progressively learns to prefer the fork square.
    """
    import matplotlib.pyplot as plt

    if snapshot_steps is None:
        snapshot_steps = [0, 20, 100, 500, 2000]
    if teacher_policy is None:
        teacher_policy = HeuristicPolicy()

    spec = GameSpec.make(dims=(3, 3), k=3)
    board = list(SCENARIOS[scenario])
    board_for_nn = np.asarray(board, dtype=float) * player

    teacher_scores = teacher_policy.scores(board, player, spec)
    legal = legal_moves(board)
    teacher_move = max(legal, key=lambda i: teacher_scores[i])

    print(f"Scenario '{scenario}' — {teacher_policy.__class__.__name__} "
          f"targets position {teacher_move}")
    print(f"Snapshot steps: {snapshot_steps}  lr={learning_rate}")

    # Train incrementally and capture at each checkpoint
    net = copy.deepcopy(network)
    steps_done = 0
    snapshots = []

    for target in sorted(snapshot_steps):
        for _ in range(target - steps_done):
            backpropagate(board_for_nn, teacher_move, 1.0, net,
                          learning_rate=learning_rate)
        steps_done = target
        scores = forward_pass(board_for_nn, net)
        snapshots.append((target, scores.copy()))
        print(f"  step {target:>5}  fork sq score: {scores[teacher_move]:+.4f}")

    # Build figure
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Targeted fork training — scenario: '{scenario}'\n"
        f"Teacher: {teacher_policy.__class__.__name__}  |  "
        f"Fork square highlighted  |  lr={learning_rate}",
        fontsize=11, fontweight="bold", y=1.03,
    )

    for ax, (step, scores) in zip(axes, snapshots):
        _draw_score_board_on_ax(ax, board, scores, teacher_move, player,
                                f"Step {step}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
