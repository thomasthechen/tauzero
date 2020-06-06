
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
        alpha: the maximum lower bound (the lowest score you are willing to accept)
        beta: the minimum upper bound (the highest score your opponent is willing to accept)
        maxDepth: maximum search depth
        '''
        self.alpha = 0
        self.beta = 10000
        self.maxDepth = 5
    
    # Function takes in board state as a chess.Board object, which you can get the list of valid moves from, append to, etc; Returns evaluation of that board state using Minimax
    def evaluate_max(self, board, currentDepth):
        if currentDepth > self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics
            return self.evaluate_board(board)
        maxVal = -10000
        for move in board.legal_moves:
            board.push(move)
            maxVal = max(self.evaluate_min(board, currentDepth + 1), maxVal)
            board.pop()
        return maxVal
    # Function corresponding to above function with the same idea, but maximizing according to opponent's incentives.
    def evaluate_min(self, board, currentDepth):
        if currentDepth >= self.maxDepth:
            # here we actually need to evaluate the board state using predefined heuristics
            return self.evaluate_board(board)
        minVal = 10000
        for move in board.legal_moves:
            board.push(move)
            minVal = min(self.evaluate_max(board, currentDepth + 1), minVal)
            board.pop()
        return minVal 
    # Function for evaluating board state using heuristics
    def evaluate_board(self, board):
        # TODO
        pass




