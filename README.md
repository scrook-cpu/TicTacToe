# Tic-Tac-Toe scaffold (headless engine + swappable policies)

## Run
python main.py

## Next TODOs
- Implement HeuristicPolicy.scores() in ttt/policies.py
- Add small “position unit tests” for winner(), apply_move(), legal_moves()
- Later: swap in LinearPolicy (learnable weights) and NNPolicy
- Future: set dims=(4,4,4), k=4 and re-use engine unchanged

PROJECT LAYOUT
ttt/
  __init__.py
  engine.py
  policies.py
main.py
README.md

