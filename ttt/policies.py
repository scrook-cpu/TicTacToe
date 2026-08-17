from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence, Tuple
import numpy as np

from .engine import GameSpec, Player, legal_moves, winner


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


@dataclass
class ForkAwareHeuristicPolicy(HeuristicPolicy):
    """
    Extends HeuristicPolicy with an offensive fork kicker.

    For each line where this player has at least one mark and the opponent has
    none, a bonus is added to every empty square on that line. Squares that sit
    on multiple such developing lines accumulate bonuses, making fork-building
    moves naturally more attractive without overriding win/block priorities.
    """
    fork_kicker: float = 50.0

    def scores(self, board: Sequence[int], player: Player, spec: GameSpec) -> np.ndarray:
        scores = super().scores(board, player, spec)
        board_arr = np.asarray(board, dtype=int)

        for line, line_vals in self._eligible_winning_lines(board_arr, player, spec):
            if np.count_nonzero(line_vals == player) == 0:
                continue
            for empty_pos in np.nonzero(line_vals == 0)[0]:
                scores[line[empty_pos]] += self.fork_kicker

        return scores


_MINIMAX_WIN = 100  # magnitude of a terminal result; depth-adjusted below


@lru_cache(maxsize=None)
def _negamax_value(board: Tuple[int, ...], side: Player, spec: GameSpec) -> int:
    """Game-theoretic value of `board` for the side to move, via negamax.

    A terminal win is worth (WIN - plies) and a loss -(WIN - plies), so the search
    prefers to win in fewer plies and to lose in more — i.e. it takes the quickest
    forced win and drags out an unavoidable loss. Draws are 0. Memoized on the
    (board, side, spec) triple, which is safe because the number of plies played is
    fully determined by the board.
    """
    w = winner(board, spec)
    if w is not None:
        # The player who just moved (the opponent of `side`) has won, so the side
        # to move is in a lost position.
        plies = sum(1 for v in board if v != 0)
        return -(_MINIMAX_WIN - plies)
    if all(v != 0 for v in board):
        return 0  # board full, no winner → draw

    best = -(_MINIMAX_WIN + 1)
    for i in range(len(board)):
        if board[i] == 0:
            child = board[:i] + (side,) + board[i + 1:]
            val = -_negamax_value(child, -side, spec)
            if val > best:
                best = val
    return best


@dataclass
class MinimaxPolicy:
    """Perfect play via full-depth minimax (negamax).

    Scores every legal move by the game-theoretic value of the position it leads
    to, assuming both sides play optimally thereafter. On 3x3 this is cheap and,
    once memoized, effectively instant. Intended as a 'perfect teacher' that —
    unlike the heuristic — reasons about the opponent's replies and therefore
    sees fork threats before they are sprung.
    """

    def scores(self, board: Sequence[int], player: Player, spec: GameSpec) -> np.ndarray:
        board_t = tuple(int(v) for v in board)
        scores = np.full(spec.size, -np.inf, dtype=float)
        for i in range(spec.size):
            if board_t[i] == 0:
                child = board_t[:i] + (player,) + board_t[i + 1:]
                scores[i] = -_negamax_value(child, -player, spec)
        return scores


@dataclass
class HeuristicExplorationPolicy(HeuristicPolicy):
    """
    Heuristic policy with added noise for exploration during training.
    """
    rng: np.random.Generator = np.random.default_rng()
    noise_scale: float = 0.25
    exploration_bonus =  2

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
        tryRandom = True

        for idx in self._immediate_win_moves(board_arr, player, spec):
            scores[idx] += immediate_win_bonus
            tryRandom = False
        for idx in self._immediate_block_moves(board_arr, player, spec):
            scores[idx] += immediate_block_bonus
            tryRandom = False
        
        if tryRandom and self.rng.random() < self.noise_scale:
            legal = legal_moves(board)

            move = self.rng.choice(legal)

            scores[move] += self.exploration_bonus
          

        return scores