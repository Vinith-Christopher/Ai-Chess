import chess
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv("random_evals.csv")

#Initiating our FEN to input Vector Function, which will output a (64,1) vector of our chess position.

def FENtoVEC (FEN):
    pieces = {"r":5,"n":3,"b":3.5,"q":9.5,"k":20,"p":1,"R":-5,"N":-3,"B":-3.5,"Q":-9.5,"K":-20,"P":-1}
    FEN = list(str(FEN.split()[0]))
    VEC = []
    for i in range(len(FEN)):
        if FEN[i] == "/":
            continue
        if FEN[i] in pieces:
            VEC.append(pieces[FEN[i]])
        else:
            em = [VEC.append(0) for i in range(int(FEN[i]))]

    return VEC




#Preparing traing data
df = df.head(400000)
board = df["FEN"].tolist()
evl = np.asarray([float(i.strip("#+")) for i in df["Evaluation"].tolist()])
RFevl = np.asarray([float(i.strip("#+")) for i in df["Evaluation"].tolist()])
for i in range(len(evl)):
    if evl[i] > 1000:
        evl[i] = float(1000)
    if evl.item(i) < -1000:
        evl[i] = float(1000)
    evl[i] = evl[i]/1000
NNtrainX = np.asarray([FENtoVEC(state) for state in board])
NNtrainy = evl

# -------- Neural Network Model --------
model = Sequential([
    Dense(64, input_dim=64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='tanh')
])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(NNtrainX, NNtrainy, epochs=10, batch_size=64, verbose=1)  # Train NN

# -------- Board Evaluation --------
def evaluate_board():
    """Evaluates board position using the trained Neural Network."""
    if board.is_checkmate():
        return -9999 if board.turn else 9999  # Negative if AI loses, positive if wins
    else:
        FENp = board.epd()
        inVEC = np.asarray([FENtoVEC(FENp)])  # Convert FEN to vector
        evalu = model.predict(inVEC)[0] * 10
        return evalu

# -------- Alpha-Beta Pruning for AI --------
def alphabeta(alpha, beta, depthleft):
    """Alpha-Beta Pruning for selecting best AI move."""
    bestscore = -9999
    if depthleft == 0:
        return quiesce(alpha, beta)

    for move in board.legal_moves:
        board.push(move)
        score = -alphabeta(-beta, -alpha, depthleft - 1)
        board.pop()

        if score >= beta:
            return score
        if score > bestscore:
            bestscore = score
        if score > alpha:
            alpha = score

    return bestscore

def quiesce(alpha, beta):
    """Quiescence search to avoid horizon effect."""
    stand_pat = evaluate_board()
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiesce(-beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

    return alpha

def selectmove(depth):
    """Selects the best move for the AI using Alpha-Beta Pruning."""
    bestMove = chess.Move.null()
    bestValue = -99999
    alpha = -100000
    beta = 100000

    for move in board.legal_moves:
        board.push(move)
        boardValue = -alphabeta(-beta, -alpha, depth - 1)
        board.pop()

        if boardValue > bestValue:
            bestValue = boardValue
            bestMove = move
        if boardValue > alpha:
            alpha = boardValue

    return bestMove

# -------- Play Against AI --------
board = chess.Board()

while not board.is_game_over():
    print(board)

    # --- User Move ---
    move_uci = input("Enter your move (in UCI format): ")
    try:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
        else:
            print("Illegal move, try again.")
            continue
    except ValueError:
        print("Invalid move format. Use UCI notation (e.g., e2e4).")
        continue

    if board.is_game_over():
        break

    # --- AI Move ---
    print("AI is thinking...")
    ai_move = selectmove(2)  # AI selects a move with depth=2
    print(f"AI plays: {ai_move}")
    board.push(ai_move)

print("Game Over!")
print(f"Result: {board.result()}")
