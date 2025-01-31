
# Importing all relevant and necessary modules

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from tensorflow import keras
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import chess


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


# # Preparing Validation Data
# Edf = Edf.head(10000)
# Eboard = Edf["FEN"].tolist()
# Eevl = np.asarray([float(i.strip("#+")) for i in Edf["Evaluation"].tolist()])
# RFEevl = np.asarray([float(i.strip("#+")) for i in Edf["Evaluation"].tolist()])
# for i in range(len(Eevl)):
#     if Eevl[i] > 1000:
#         Eevl[i] = float(1000)
#     if Eevl.item(i) < -1000:
#         Eevl[i] = float(1000)
#     Eevl[i] = Eevl[i]/1000
# NNvalX = np.asarray([FENtoVEC(state) for state in Eboard])
# NNvaly = Eevl
# RRvalX = pd.DataFrame(NNvalX[:10000])
# RRvaly = RFEevl


# --------- model ------
model = Sequential()
model.add(Dense(64, input_dim=64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'],steps_per_execution = 32)

model.fit(NNtrainX, NNtrainy, epochs=100,batch_size = 64, verbose = 0.1)




RFtrain_X = pd.DataFrame(NNtrainX[:10000])
RFtrain_y = RFevl[:10000]
forest_model = RandomForestRegressor(n_estimators=200, max_depth= 32,random_state=1)
forest_model.fit(RFtrain_X, RFtrain_y)


# -------------------
def evaluate_board():
    if board.is_checkmate():
        if board.turn:
            return -9999
        else:
            return 9999
    else:
        FENp = board.epd()
        inVEC = np.asarray([FENtoVEC(FENp)])
        evalu = model.predict(inVEC)[0]*10
        return evalu
board = chess.Board()
print(evaluate_board())
def alphabeta( alpha, beta, depthleft ):
    bestscore = -9999
    if( depthleft == 0 ):
        return quiesce( alpha, beta )
    for move in board.legal_moves:
        board.push(move)
        score = -alphabeta( -beta, -alpha, depthleft - 1 )
        board.pop()
        if( score >= beta ):
            return score
        if( score > bestscore ):
            bestscore = score
        if( score > alpha ):
            alpha = score
    return bestscore
def quiesce( alpha, beta ):
    stand_pat = evaluate_board()
    if( stand_pat >= beta ):
        return beta
    if( alpha < stand_pat ):
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiesce( -beta, -alpha )
            board.pop()

            if( score >= beta ):
                return beta
            if( score > alpha ):
                alpha = score
    return alpha
def selectmove(depth):
    bestMove = chess.Move.null()
    bestValue = -99999
    alpha = -100000
    beta = 100000
    for move in board.legal_moves:
        board.push(move)
        boardValue = -alphabeta(-beta, -alpha, depth-1)
        if boardValue > bestValue:
            bestValue = boardValue;
            bestMove = move
        if( boardValue > alpha ):
            alpha = boardValue
        board.pop()
    return bestMove

def evaluate_boardRF():
    if board.is_checkmate():
        if board.turn:
            return -9999
        else:
            return 9999
    else:
        FENp = board.epd()
        inVEC = np.asarray([FENtoVEC(FENp)])
        evalu = forest_model.predict(inVEC)[0]
        return evalu
def alphabetaRF( alpha, beta, depthleft ):
    bestscore = -9999
    if( depthleft == 0 ):
        return quiesceRF( alpha, beta )
    for move in board.legal_moves:
        board.push(move)
        score = -alphabetaRF( -beta, -alpha, depthleft - 1 )
        board.pop()
        if( score >= beta ):
            return score
        if( score > bestscore ):
            bestscore = score
        if( score > alpha ):
            alpha = score
    return bestscore
def quiesceRF( alpha, beta ):
    stand_pat = evaluate_boardRF()
    if( stand_pat >= beta ):
        return beta
    if( alpha < stand_pat ):
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiesceRF( -beta, -alpha )
            board.pop()

            if( score >= beta ):
                return beta
            if( score > alpha ):
                alpha = score
    return alpha
def selectmoveRF(depth):
    bestMove = chess.Move.null()
    bestValue = -99999
    alpha = -100000
    beta = 100000
    for move in board.legal_moves:
        board.push(move)
        boardValue = -alphabetaRF(-beta, -alpha, depth-1)
        if boardValue > bestValue:
            bestValue = boardValue;
            bestMove = move
        if( boardValue > alpha ):
            alpha = boardValue
        board.pop()
    return bestMove



for i in range(100):
    print(board)
    move = selectmove(2)
    print(move)
    board.push(move)
    print(evaluate_board())
    print(board)
    move = selectmoveRF(2)
    print(move)
    board.push(move)
    print(evaluate_boardRF())
