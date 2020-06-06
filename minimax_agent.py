
import chess
import random

class MiniMaxAgent:
    def __init__(self):
        '''
        Fields:
        alpha: the maximum lower bound (the lowest score you are willing to accept)
        beta: the minimum upper bound (the highest score your opponent is willing to accept)
        maxDepth: maximum search depth
        '''
        self.alpha = 0
        self.beta = 10000
        self.maxDepth = 10
    
    # Function takes in board state as a chess.Board object, which you can get the list of valid moves from, append to, etc.
    def evaluate_max(self, board, currentDepth):
        if currentDepth > self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics 
        maxVal = -10000
        for move in board.legal_moves:
            board.push(move)
            maxVal = max(self.evaluate_min(board, currentDepth + 1), maxVal)
            board.pop()
        return maxVal

    def evaluate_min(self, board, currentDepth):
        if currentDepth >= self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics
        minVal = 10000
        for move in board.legal_moves:
            board.push(move)
            minVal = min(self.evaluate_max(board, currentDepth + 1), minVal)
            board.pop()
        return minVal 

    def evaluate_board(self, board):
        # TODO
        pass




