from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import itertools

Player = int  # +1 or -1
Board = List[int]  # flat list length N = rows*cols*(levels...) ; values in {-1,0,+1}

MoveRecord = Dict[str, Any]
MoveCallback = Callable[[MoveRecord], None]


@dataclass(frozen=True)
class GameSpec:
    dims: Tuple[int, ...] = (3, 3)  # 2D default; later (4,4,4)
    k: int = 3                      # win length; later 4 for 4x4x4
    winning_lines: Tuple[Tuple[int, ...], ...] = ()

    @staticmethod
    def make(dims: Tuple[int, ...] = (3, 3), k: int = 3) -> "GameSpec":
        """
        Factory that precomputes winning lines once.
        """
        lines = tuple(tuple(line) for line in generate_winning_lines(dims, k))
        return GameSpec(dims=dims, k=k, winning_lines=lines)

    @property
    def size(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


# ---------- Coordinate helpers (future-friendly) ----------

def ravel_index(coord: Tuple[int, ...], dims: Tuple[int, ...]) -> int:
    """
    Convert N-D coordinate -> flat index.
    """
    idx = 0
    stride = 1
    for c, dim in zip(coord, dims):
        idx += c * stride
        stride *= dim
    return idx


def unravel_index(idx: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Convert flat index -> N-D coordinate.
    """
    coord = []
    for dim in dims:
        coord.append(idx % dim)
        idx //= dim
    return tuple(coord)


# ---------- Winning lines ----------

def generate_winning_lines(dims: Tuple[int, ...], k: int) -> List[List[int]]:
    """
    Generate all straight k-in-a-row lines in an N-D grid.

    For (3,3), k=3 -> classic tic-tac-toe lines.
    For (4,4,4), k=4 -> Atari-style 3D lines.

    TODO (optional): Optimize / cache, but for these sizes it's fine.
    """
    D = len(dims)

    # All direction vectors in {-1,0,1}^D except all-zeros.
    # Canonicalize to avoid duplicates by keeping only directions where
    # the first nonzero component is +1.
    directions = []
    for d in itertools.product([-1, 0, 1], repeat=D):
        if all(x == 0 for x in d):
            continue
        for x in d:
            if x != 0:
                if x == 1:
                    directions.append(d)
                break

    lines: List[List[int]] = []
    starts = itertools.product(*[range(n) for n in dims])

    for start in starts:
        for d in directions:
            end = tuple(start[i] + (k - 1) * d[i] for i in range(D))
            if all(0 <= end[i] < dims[i] for i in range(D)):
                coords = [
                    tuple(start[i] + t * d[i] for i in range(D))
                    for t in range(k)
                ]
                line = [ravel_index(c, dims) for c in coords]
                lines.append(line)

    return lines


# ---------- Core engine functions ----------

def new_board(spec: GameSpec) -> Board:
    return [0] * spec.size


def legal_moves(board: Sequence[int]) -> List[int]:
    """
    Return list of indices that are empty.
    """
    # TODO: If you later encode board differently, update this.
    return [i for i, v in enumerate(board) if v == 0]


def apply_move(board: Sequence[int], idx: int, player: Player) -> Board:
    """
    Return a new board with player applied at idx.
    """
    if player not in (+1, -1):
        raise ValueError("player must be +1 or -1")
    if idx < 0 or idx >= len(board):
        raise IndexError("move index out of range")
    if board[idx] != 0:
        raise ValueError("illegal move: square is occupied")

    b = list(board)
    b[idx] = player
    return b


def winner(board: Sequence[int], spec: GameSpec) -> Optional[Player]:
    """
    Return +1 if +1 has won, -1 if -1 has won, None if no winner yet.
    (Draw detection is separate: no winner + no legal moves.)
    """
    # TODO: If you want speed, precompute sums / bitboards; not needed yet.
    for line in spec.winning_lines:
        s = 0
        for idx in line:
            s += board[idx]
        if s == spec.k:
            return +1
        if s == -spec.k:
            return -1
    return None


def is_draw(board: Sequence[int], spec: GameSpec) -> bool:
    return winner(board, spec) is None and all(v != 0 for v in board)


def render_2d(board: Sequence[int], dims: Tuple[int, ...] = (3, 3)) -> str:
    """
    Minimal 2D renderer for debugging.
    TODO: Make a 3D renderer later if you want.
    """
    if len(dims) != 2:
        return "<render_2d only supports 2D dims>"
    rows, cols = dims
    symbols = {+1: "X", -1: "O", 0: "."}
    lines = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = ravel_index((r, c), dims)
            row.append(symbols[board[idx]])
        lines.append(" ".join(row))
    return "\n".join(lines)


def choose_move_from_scores(scores: Sequence[float], board: Sequence[int]) -> int:
    """
    Select the highest-scoring legal move from a score vector.

    This helper keeps legal-move masking in one reusable place for demos,
    heuristic policies, and neural-network training.
    """
    best_idx: Optional[int] = None
    best_score = float("-inf")

    for i, score in enumerate(scores):
        if board[i] != 0:
            continue
        if best_idx is None or score > best_score:
            best_idx = i
            best_score = float(score)

    if best_idx is None:
        raise ValueError("No legal moves available.")

    return best_idx


def choose_move(policy: Any, board: Sequence[int], player: Player, spec: GameSpec) -> int:
    """
    Ask a policy for move scores and return the best legal move.

    A policy is expected to expose:
        scores(board, player, spec) -> Sequence[float]
    """
    scores = policy.scores(board, player, spec)
    return choose_move_from_scores(scores, board)


def play_game(
    pX: Any,
    pO: Any,
    spec: GameSpec,
    move_callback: MoveCallback | None = None,
) -> int:
    """
    Play one headless game between two policies.

    Returns:
        +1 if X wins, -1 if O wins, 0 for draw.

    If move_callback is provided, it is called once per move with a record:
        {
            "turn": int,
            "player": +1 or -1,
            "policy": policy object,
            "policy_name": str,
            "board_before": list[int],
            "scores": list[float],
            "move": int,
            "board_after": list[int],
        }
    """
    board = new_board(spec)
    player: Player = +1
    turn = 0

    while True:
        turn += 1
        policy = pX if player == +1 else pO
        board_before = list(board)
        scores = policy.scores(board_before, player, spec)
        move = choose_move_from_scores(scores, board_before)
        board = apply_move(board_before, move, player)

        if move_callback is not None:
            move_callback(
                {
                    "turn": turn,
                    "player": player,
                    "policy": policy,
                    "policy_name": policy.__class__.__name__,
                    "board_before": board_before,
                    "scores": list(scores),
                    "move": move,
                    "board_after": list(board),
                }
            )

        w = winner(board, spec)
        if w is not None:
            return w
        if is_draw(board, spec):
            return 0

        player *= -1
