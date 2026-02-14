from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np

from .engine import GameSpec, Player, legal_moves


class Policy(Protocol):
    def scores(self, board: Sequence[int], player: Player, spec: GameSpec) -> np.ndarray:
        ...


@dataclass
class RandomPolicy:
    rng: np.random.Generator = np.random.default_rng()

    def scores(self, board: Sequence[int], player: Player, spec: GameSpec) -> np.ndarray:
        """
        Random scores for demo/testing.
        TODO: Replace with HeuristicPolicy.scores().
        """
        return self.rng.standard_normal(spec.size)


def choose_move(policy: Policy, board: Sequence[int], player: Player, spec: GameSpec) -> int:
    """
    Compute scores, mask illegal moves, return best legal move.
    """
    s = policy.scores(board, player, spec)
    if s.shape != (spec.size,):
        raise ValueError(f"policy must return shape ({spec.size},), got {s.shape}")

    # Mask illegal moves: set them to -inf so they never get chosen.
    masked = s.astype(float).copy()
    masked[:] = masked  # explicit copy already; keep readable

    # TODO: Consider using a boolean mask for speed; this is clear.
    for i, v in enumerate(board):
        if v != 0:
            masked[i] = -np.inf

    lm = legal_moves(board)
    if not lm:
        raise ValueError("No legal moves available.")

    return int(np.argmax(masked))


@dataclass
class HeuristicPolicy:
    """
    TODO: This is where your “value add” starts.
    Implement scores() using your heuristic matrices / line-based features.
    """
    def scores(self, board: Sequence[int], player: Player, spec: GameSpec) -> np.ndarray:
        # TODO:
        # 1) Start with base positional priors (optional) OR line-based priors.
        # 2) Add immediate-win bonuses (player has k-1 in a line, 1 empty).
        # 3) Add immediate-block bonuses (opponent has k-1 in a line, 1 empty).
        # 4) Add fork bonuses (optional but fun).
        # 5) Return a length-N vector of move desirability.
        return np.zeros(spec.size, dtype=float)
