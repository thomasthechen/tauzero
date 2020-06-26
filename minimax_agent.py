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
        self.maxDepth = 4 # note depth has to be an even number
        self.maxBreadth = 100
        self.value_approx = Net()
        self.value_approx.load_state_dict(torch.load('./trained_models/value_40_6000_4.pth', map_location=torch.device('cpu')))
        self.value_approx.eval()
    


    # Function takes in board state as a chess.Board object, which you can get the list of valid moves from, append to, etc; Returns evaluation of that board state using Minimax
    def evaluate_max(self, board, currentDepth, alpha=-np.inf, beta=np.inf):
        board_eval = self.evaluate_board(board) 
        if board_eval > beta:
            return board_eval
        if currentDepth == self.maxDepth:
            return board_eval
        branching = 0
        for move in board.legal_moves:
            if branching > self.maxBreadth: break
            branching += 1
            board.push(move)
            alpha = max(self.evaluate_min(board, currentDepth + 1, alpha=alpha, beta=beta), alpha)
            board.pop()

            if beta < alpha:
                break

        return alpha


    # Function corresponding to above function with the same idea, but maximizing according to opponent's incentives.
    def evaluate_min(self, board, currentDepth, alpha=-np.inf, beta=np.inf):
        board_eval = self.evaluate_board(board) 
        if (board_eval < alpha):
            return board_eval
        if currentDepth == self.maxDepth:
            return board_eval
        branching = 0
        for move in board.legal_moves:
            if branching > self.maxBreadth: break
            
            branching += 1
            board.push(move)
            beta = min(self.evaluate_max(board, currentDepth + 1, alpha=alpha, beta=beta), beta)
            board.pop()
            
            if  beta < alpha:
                break

        return beta 

    def evaluate_board_heuristic(self, board):
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
                    evaluation += pieceToValue[square]
                else:
                    col += int(square)

        return -0.04 * evaluation

    # Function for evaluating board state using heuristics
    def evaluate_board(self, board):
        in_tensor = torch.tensor(State(board).serialize()).float()
        in_tensor = in_tensor.reshape(1, 13, 8, 8)
        # print(self.evaluate_board_heuristic(board))
        return self.value_approx(in_tensor).item() + self.evaluate_board_heuristic(board)

    # Function: takes in board state, returns top 2 moves    
    def minimax(self, board):
        move_evaluations = []
        for move in board.legal_moves:
            board.push(move)
            evaluation = self.evaluate_max(board, 1)
            move_evaluations.append((move, evaluation))
            board.pop()
        move_evaluations.sort(key = lambda x: x[1]) 
        print(move_evaluations[0:3])

        return move_evaluations[0:1]


# run the main function
if __name__ == '__main__':
    agent = MiniMaxAgent()
    fen = chess.STARTING_FEN
    board = chess.Board(fen)
    print(agent.evaluate_board(board))

    for i in range (10):
        for move in board.legal_moves:
            board.push(move)
            print(board)
            print(agent.evaluate_board(board))
            break

