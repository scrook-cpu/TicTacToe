import unittest
import numpy as np

from ttt.engine import GameSpec
from ttt.policies import HeuristicPolicy


class TestHeuristicPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GameSpec.make((3, 3), 3)
        self.policy = HeuristicPolicy()

    def test_eligible_winning_lines_filters_opponent(self):
        board = [0] * self.spec.size
        board[0] = -1
        eligible = list(self.policy._eligible_winning_lines(np.array(board), +1, self.spec))
        self.assertGreater(len(eligible), 0)
        for line, line_vals in eligible:
            self.assertNotIn(-1, line_vals)

    def test_immediate_win_moves(self):
        board = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        wins = sorted(self.policy._immediate_win_moves(np.array(board), +1, self.spec))
        self.assertEqual(wins, [2])

    def test_immediate_block_moves(self):
        board = [-1, -1, 0, 0, 0, 0, 0, 0, 0]
        blocks = sorted(self.policy._immediate_block_moves(np.array(board), +1, self.spec))
        self.assertEqual(blocks, [2])

    def test_scores_prioritize_immediate_win_and_block(self):
        board = [1, 1, 0, -1, -1, 0, 0, 0, 0]
        scores = self.policy.scores(board, +1, self.spec)
        self.assertGreater(scores[2], scores[5])
        self.assertGreater(scores[2], scores[8])

        block_board = [-1, -1, 0, 1, 0, 0, 0, 0, 0]
        block_scores = self.policy.scores(block_board, +1, self.spec)
        self.assertGreater(block_scores[2], block_scores[8])
        self.assertGreater(block_scores[2], block_scores[4])

    def test_scores_returns_correct_shape(self):
        board = [0] * self.spec.size
        scores = self.policy.scores(board, +1, self.spec)
        self.assertEqual(scores.shape, (self.spec.size,))


if __name__ == '__main__':
    unittest.main()
