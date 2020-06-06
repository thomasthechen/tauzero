import numpy as np
import chess
import random

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


    # Function takes in board state as a chess.Board object, which you can get the list of valid moves from, append to, etc; Returns evaluation of that board state using Minimax
    def evaluate_max(self, board, currentDepth, alpha=-np.inf, beta=np.inf):
        # first call begins with a depth of 0
        if currentDepth == self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics
            return self.evaluate_board(board)  
        for move in board.legal_moves:
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
        
        for move in board.legal_moves:
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
        # should really be dependent on who we're evaluating for
        # print(fen[0], evaluation)
        return -evaluation

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

