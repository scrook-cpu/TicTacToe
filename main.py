from __future__ import annotations

from ttt.engine import GameSpec, new_board, apply_move, winner, is_draw, render_2d
from ttt.policies import RandomPolicy, HeuristicPolicy, choose_move

def play_demo():
    spec = GameSpec.make(dims=(3, 3), k=3)
    board = new_board(spec)

    # TODO: Swap RandomPolicy -> HeuristicPolicy once implemented.
    pX = RandomPolicy()
    pO = RandomPolicy()

    player = +1  # X starts
    turn = 0

    while True:
        turn += 1
        policy = pX if player == +1 else pO
        idx = choose_move(policy, board, player, spec)
        board = apply_move(board, idx, player)

        print(f"\nTurn {turn}: {'X' if player==+1 else 'O'} plays {idx}")
        print(render_2d(board, spec.dims))

        w = winner(board, spec)
        if w is not None:
            print("\nWinner:", "X" if w == +1 else "O")
            break
        if is_draw(board, spec):
            print("\nDraw.")
            break

        player *= -1

if __name__ == "__main__":
    play_demo()
