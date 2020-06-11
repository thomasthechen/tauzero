import numpy as np
import chess
import random
from value_approximator import Net
import torch
from state import State
class MiniMaxAgent:
    '''
    TODO:
    1. implement function that evaluates board state using heuristics
    2. test brute-force search
    3. add alpha-beta pruning
    '''

    def __init__(self):
        '''
        Fields:
        alpha: the maximum lower bound (the largest score guaranteed to the maximizing player)
        beta: the minimum upper bound (the lowest score guaranteed to the minimizing player)
        maxDepth: maximum search depth
        '''
        self.maxDepth = 5
        self.maxBreadth = 5
        self.value_approx = Net()
        self.value_approx.load_state_dict(torch.load('./trained_models/value.pth', map_location=torch.device('cpu')))
        self.value_approx.eval()
    


    # Function takes in board state as a chess.Board object, which you can get the list of valid moves from, append to, etc; Returns evaluation of that board state using Minimax
    def evaluate_max(self, board, currentDepth, alpha=-np.inf, beta=np.inf):
        # first call begins with a depth of 0
        if currentDepth == self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics
            return self.evaluate_board(board)
        branching = 0
        # TODO: we should iterate through these in an order so that the most promising ones are explored first. For this, we need a better move iterator than the default one
        for move in board.legal_moves:
            if branching > self.maxBreadth: break
            branching += 1
            board.push(move)
            # search for best route given negative node optimization
            alpha = max(self.evaluate_min(board, currentDepth + 1, alpha=alpha, beta=beta), alpha)
            board.pop()

            # stop pruning if we know that minimizer is guaranteed
            # a better objective score elsewhere
            if beta < alpha:
                break

        return alpha


    # Function corresponding to above function with the same idea, but maximizing according to opponent's incentives.
    def evaluate_min(self, board, currentDepth, alpha=-np.inf, beta=np.inf):
        # first call begins with a depth of 0
        if currentDepth == self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics
            return self.evaluate_board(board)
        branching = 0
        for move in board.legal_moves:
            if branching > self.maxBreadth: break
            branching += 1
            board.push(move)
            # search for best route given positive node optimization
            beta = min(self.evaluate_max(board, currentDepth + 1, alpha=alpha, beta=beta), beta)
            board.pop()
            
            # stop searching if we know maximizer is guaranteed a better
            # objective score elsewhere
            if  beta < alpha:
                break

        return beta 


    # returns a value from about 0.75 to 1, where central coordinates yield a higher value
    def centerFunction(self, row, col):
        return 1 - ((row - 3.5) * (row - 3.5) + (col - 3.5) + (col - 3.5))/250


    # Function for evaluating board state using heuristics
    def evaluate_board(self, board):
        evaluation = 0
        pieceToValue = {
            'r' : 5,
            'n' : 3,
            'b' : 3.25,
            'q' : 7,
            'k' : 0,
            'p' : 1,
            'R' : -5,
            'N' : -3,
            'B' : -3.25,
            'Q' : -7,
            'K' : 0,
            'P' : -1
        }

        fen = board.fen().split()
        pieces = fen[0].split('/')
        # print(pieces)
        for row in range(len(pieces)):
            col = 0
            for square in pieces[row]:
                if square in pieceToValue:
                    evaluation += pieceToValue[square] * self.centerFunction(row, col)
                else:
                    col += int(square)

        # return -evaluation
        # NN evaluation
        in_tensor = torch.tensor(State(board).serialize()).float()
        in_tensor = in_tensor.reshape(1, 13, 8, 8)
        return self.value_approx(in_tensor).item()

    # Function: takes in board state, returns top 2 moves    
    def minimax(self, board):
        move_evaluations = []
        for move in board.legal_moves:
            board.push(move)
            evaluation = self.evaluate_min(board, 1)
            move_evaluations.append((move, evaluation))
            board.pop()
        move_evaluations.sort(key = lambda x: x[1]) 
        print(move_evaluations)

        return move_evaluations[0:1]


# run the main function
if __name__ == '__main__':
    agent = MiniMaxAgent()
    fen = chess.STARTING_FEN
    board = chess.Board(fen)
    print(agent.evaluate_board(board))

