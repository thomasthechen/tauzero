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
    def evaluate_max(self, board, alpha=-np.inf, beta=np.inf, currentDepth):
        # first call begins with a depth of 0
        if currentDepth == self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics
            return self.evaluate_board(board)  
        for move in board.legal_moves:
            board.push(move)
            # search for best route given negative node optimization
            alpha = max(self.evaluate_min(board, alpha=alpha, beta=beta, currentDepth + 1), alpha)
            board.pop()

            # stop pruning if we know that minimizer is guaranteed
            # a better objective score elsewhere
            if beta < alpha:
                break

        return maxVal

    # Function corresponding to above function with the same idea, but maximizing according to opponent's incentives.
    def evaluate_min(self, board, alpha=-np.inf, beta=np.inf, currentDepth):
        # first call begins with a depth of 0
        if currentDepth == self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics
            return self.evaluate_board(board)
        
        for move in board.legal_moves:
            board.push(move)
            # search for best route given positive node optimization
            beta = min(self.evaluate_max(board, alpha=alpha, beta=beta, currentDepth + 1), beta)
            board.pop()

            # stop searching if we know maximizer is guaranteed a better
            # objective score elsewhere
            if  beta < alpha:
                break

        return beta 
    # Function for evaluating board state using heuristics
    def evaluate_board(self, board):
        # TODO
        pass




