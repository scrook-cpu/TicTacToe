import numpy as np
from typing import Dict, Any
from .engine import GameSpec, legal_moves, play_game
from .policies import HeuristicPolicy, RandomPolicy


def initialize_weights(input_size: int, output_size: int) -> np.ndarray:
    """
    Initialize a weight matrix with random values.
    TODO: Implement random initialization using np.random.randn, scaled appropriately (e.g., * 0.1).
    """
    return np.random.randn(output_size, input_size) * 0.1


def initialize_bias(output_size: int) -> np.ndarray:
    """
    Initialize a bias vector with random values.
    TODO: Implement random initialization using np.random.randn, scaled appropriately.
    """
    return np.random.randn(output_size) * 0.1

def create_network(hidden_size: int = 10) -> Dict[str, Any]:
    """
    Create and initialize the neural network parameters.
    TODO: Optionally include hyperparameters like learning_rate in the dict.
    """
    d = dict()
    
    d[f"W1"] = initialize_weights(9, hidden_size)
    d[f"b1"] = initialize_bias(hidden_size)
    d[f"W2"] = initialize_weights(hidden_size, 9)
    d[f"b2"] = initialize_bias(9)
    return d

class NeuralNetworkPolicy:
    """
    Policy wrapper that lets a neural-network parameter dict plug into engine.play_game().

    The network itself is just learned weights/biases. This class adapts it to the
    policy interface expected by the engine:
        scores(board, player, spec) -> score vector
    """
    def __init__(self, network: Dict[str, Any]):
        self.network = network

    def scores(self, board, player: int, spec: GameSpec) -> np.ndarray:
        # Normalize to current-player perspective so +1 means "me" and -1 means "opponent".
        board_for_nn = np.asarray(board, dtype=float) * player
        return forward_pass(board_for_nn, self.network)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def forward_pass(board: np.ndarray, network: Dict[str, Any]) -> np.ndarray:
    """
    Compute the move scores from the board state.
    TODO: Implement the forward pass: h = relu(W1 · board + b1), then scores = W2 · h + b2.
    TODO: Use np.maximum(0, x) for ReLU activation.
    """
    # Extract W1, b1, W2, b2 from network.
    W1 = network["W1"]
    b1 = network["b1"]
    W2 = network["W2"]
    b2 = network["b2"]
    # TODO: Compute hidden layer with ReLU.
    h = relu(np.dot(W1, board) + b1)
    # TODO: Compute output scores (no activation).
    scores = np.dot(W2, h) + b2
    return scores


def choose_move(board: np.ndarray, network: Dict[str, Any]) -> int:
    """
    Select the best legal move based on network scores.
    TODO: Call forward_pass to get scores.
    TODO: Mask illegal moves (occupied squares) by setting their scores to -inf.
    TODO: Return the index of the highest-scoring legal move.
    """
    # TODO: Get scores from forward_pass.
    scores = forward_pass(board, network)
    # TODO: Identify legal moves using legal_moves(board).
    legal = legal_moves(board)
    # TODO: Mask scores for illegal moves.
    masked = scores.astype(float).copy()
    for i in range(len(board)):
        if i not in legal:
            masked[i] = -np.inf
    # TODO: Return np.argmax of masked scores.

    return np.argmax(masked)


def compute_loss(predicted_scores: np.ndarray, target_scores: np.ndarray) -> float:
    """
    Compute the loss between predicted and target scores.
    TODO: Implement mean squared error: np.mean((predicted_scores - target_scores)**2).
    """
    # TODO: Return the MSE loss.
    return np.mean((predicted_scores - target_scores)**2)
    


def backpropagate(board: np.ndarray, chosen_move: int, outcome: float, network: Dict[str, Any], learning_rate: float = 0.01) -> Dict[str, Any]:
    """
    Update the network parameters using backpropagation.
    TODO: Compute forward pass to get intermediate values.
    TODO: Compute gradients for W2, b2, W1, b1 using chain rule for MSE loss.
    TODO: Update parameters: W -= learning_rate * dW, etc.
    TODO: Return the updated network dict.
    """
    # TODO: Perform forward pass to cache h and scores.
    fp = forward_pass(board, network)
    # TODO: Compute target_scores (e.g., outcome at chosen_move index, 0 elsewhere).
    target_scores = np.zeros_like(fp)
    target_scores[chosen_move] = outcome
    # TODO: Compute dL/dscores, then backprop through W2 and b2.
    dL_dscores = 2 * (fp - target_scores) / len(fp)  # MSE gradient
    dL_dW2 = np.outer(dL_dscores, relu(np.dot(network["W1"], board) + network["b1"]))
    dL_db2 = dL_dscores
    # TODO: Backprop through hidden layer to W1 and b1.
    dh = np.dot(network["W2"].T, dL_dscores)
    dh_relu = dh * (relu(np.dot(network["W1"], board) + network["b1"]) > 0)  # ReLU backprop
    dL_dW1 = np.outer(dh_relu, board)
    dL_db1 = dh_relu    
    # TODO: Update all parameters in place.
    network["W2"] -= learning_rate * dL_dW2
    network["b2"] -= learning_rate * dL_db2
    network["W1"] -= learning_rate * dL_dW1
    network["b1"] -= learning_rate * dL_db1
    return network
    


def train_game(network: Dict[str, Any], opponent_policy, num_games: int = 1000) -> Dict[str, Any]:
    """
    Train the network by playing games against an opponent.
    TODO: For each game, play moves using choose_move, record board states and chosen moves.
    TODO: After game ends, backpropagate rewards (+1 for win, -1 for loss, 0 for draw).
    TODO: Return the trained network.
    """

    spec = GameSpec.make(dims=(3, 3), k=3)
    teacher = HeuristicPolicy()

    for _ in range(num_games):
        def train_from_record(record):
            # For imitation learning, only train on the teacher's moves.
            if record["policy"] is not teacher:
                return

            board_for_nn = np.asarray(record["board_before"], dtype=float) * record["player"]
            target_move = record["move"]

            # In imitation mode, outcome=1.0 means "raise the score for the teacher's move".
            backpropagate(board_for_nn, target_move, 1.0, network)

        play_game(teacher, opponent_policy, spec, move_callback=train_from_record)

    return network


def evaluate_policy(network: Dict[str, Any], opponent_policy, num_games: int = 100) -> Dict[str, int]:
    """
    Evaluate the network against an opponent.
    TODO: Play num_games against opponent_policy.
    TODO: Count wins, losses, draws.
    TODO: Return a dict with counts (e.g., {'wins': 50, 'losses': 30, 'draws': 20}).
    """
    spec = GameSpec.make(dims=(3, 3), k=3)
    nn_policy = NeuralNetworkPolicy(network)
    stats = {"wins": 0, "losses": 0, "draws": 0}

    for _ in range(num_games):
        result = play_game(nn_policy, opponent_policy, spec)
        if result == +1:
            stats["wins"] += 1
        elif result == -1:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

    return stats


def save_network(network: Dict[str, Any], filename: str):

    """
    Save the network parameters to a file.
    Uses np.savez to store arrays in compressed format.
    """
    np.savez(
        filename,
        W1=network["W1"],
        b1=network["b1"],
        W2=network["W2"],
        b2=network["b2"],
    )

def load_network(filename: str) -> Dict[str, Any]:

    """
    Load network parameters from a file.
    Returns a network dict identical to what was saved.
    """

    data = np.load(filename)
    network = {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
    }

    return network

def visualize_scores(board: np.ndarray, network: Dict[str, Any]):
    
    scores = forward_pass(board, network)
    grid = scores.reshape(3, 3)
    board_grid = board.reshape(3, 3)
    print("\nMove Scores:")
    for r in range(3):
        row_str = []
        for c in range(3):
            if board_grid[r, c] != 0:
                row_str.append("  X/O ")
            else:
                row_str.append(f"{grid[r, c]:6.2f}")
        print(" ".join(row_str))
