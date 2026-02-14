# TicTacToe

A small headless Tic-Tac-Toe engine with swappable player policies.

## Quick start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies and run:

```bash
pip install -r requirements.txt
python main.py
```

3. The program runs a simple demo game and prints moves/winner.

## Project layout

ttt/
  - `__init__.py`
  - `engine.py` — game engine logic (winner detection, move application)
  - `policies.py` — player policies / AI implementations

`main.py` — example runner

## Next steps / TODOs

- Implement `HeuristicPolicy.scores()` in `ttt/policies.py` (if desired)
- Add unit tests for `winner()`, `apply_move()`, and `legal_moves()`
- Optionally add CI to run tests on push

## CI and License

No CI is configured yet. I can add a GitHub Actions workflow to run tests and linting.

If you want the project licensed, add a `LICENSE` file (MIT is a common choice).
