
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

    # returns a value from about 0.75 to 1, where central coordinates yield a higher value
    def centerFunction(self, row, col):
        return 1 - ((row - 3.5) * (row - 3.5) + (col - 3.5) + (col - 3.5))/50
    # Function for evaluating board state using heuristics
    def evaluate_board(self, board):
        evaluation = 0
        pieceToValue = {
            'r' : 5,
            'n' : 3,
            'b' : 3.25,
            'q' : 9,
            'k' : 0,
            'p' : 1,
            'R' : -5,
            'N' : -3,
            'B' : -3.25,
            'Q' : -9,
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
        print(fen[0], evaluation)
        return evaluation
        

# run the main function
if __name__ == '__main__':
    agent = MiniMaxAgent()
    fen = chess.STARTING_FEN
    board = chess.Board(fen)
    print(agent.evaluate_board(board))



