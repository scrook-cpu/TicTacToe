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
        Just populates our numpy 1d array with random values; the masking of illegal moves happens in choose_move.
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
    Heuristic policy that prefers positions with more remaining winning-line potential,
    and that always prioritizes immediate wins and immediate blocks.
    """
    def _eligible_winning_lines(self, board_arr: np.ndarray, player: Player, spec: GameSpec):
        """Yield lines that are still winnable for the specified player.

        A line is eligible if it contains no opponent pieces and has at least one
        empty square remaining.
        """
        opponent = -player
        for line in spec.winning_lines:
            line_vals = board_arr[list(line)]
            if np.any(line_vals == opponent):
                continue
            yield line, line_vals

    def _immediate_win_moves(self, board_arr: np.ndarray, player: Player, spec: GameSpec):
        """Yield move indices that complete an immediate win for the player."""
        for line in spec.winning_lines:
            line_vals = board_arr[list(line)]
            empty_indices = np.nonzero(line_vals == 0)[0]
            if empty_indices.size != 1:
                continue

            empty_idx = line[empty_indices[0]]
            if int(line_vals.sum()) == (spec.k - 1) * player:
                yield empty_idx

    def _immediate_block_moves(self, board_arr: np.ndarray, player: Player, spec: GameSpec):
        """Yield move indices that block the opponent's immediate win."""
        for line in spec.winning_lines:
            line_vals = board_arr[list(line)]
            empty_indices = np.nonzero(line_vals == 0)[0]
            if empty_indices.size != 1:
                continue

            empty_idx = line[empty_indices[0]]
            if int(line_vals.sum()) == -(spec.k - 1) * player:
                yield empty_idx

    def scores(self, board: Sequence[int], player: Player, spec: GameSpec) -> np.ndarray:
        board_arr = np.asarray(board, dtype=int)
        if board_arr.shape != (spec.size,):
            raise ValueError(f"board must have shape ({spec.size},), got {board_arr.shape}")
        if player not in (+1, -1):
            raise ValueError("player must be +1 or -1")

        scores = np.zeros(spec.size, dtype=float)

        # Base positional heuristic: reward moves that appear in more possible winning lines.
        for line, line_vals in self._eligible_winning_lines(board_arr, player, spec):
            empty_indices = np.nonzero(line_vals == 0)[0]
            if empty_indices.size == 0:
                continue

            player_count = np.count_nonzero(line_vals == player)
            line_value = 1.0 + 0.5 * player_count
            for empty_pos in empty_indices:
                scores[line[empty_pos]] += line_value

        immediate_win_bonus = 1_000.0
        immediate_block_bonus = 900.0

        for idx in self._immediate_win_moves(board_arr, player, spec):
            scores[idx] += immediate_win_bonus
        for idx in self._immediate_block_moves(board_arr, player, spec):
            scores[idx] += immediate_block_bonus

        return scores
