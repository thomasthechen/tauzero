'''
This is a file used to test the python-chess library and its functionality. 

Abbreviations:
SAN - standard algebraic notation (Nf3)
UCI - universal chess interface (g1f3)
FEN - Forsyth-Edwards notation (for board state)

board.turn returns True for white and False for black

By default, moves are notated with UCI. 
'''

import chess
import random

def main():

    # initialize board in standard starting position
    # STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    fen = chess.STARTING_FEN
    board = chess.Board(fen)
    
    while not board.is_game_over():
        # display whose turn it is
        if board.turn:
            print('White\'s Turn')
        else:
            print('Black\'s Turn')
        print('\n')
        
        # display board
        print(board)
        print('\n')
        
        # display possible moves
        print('Possible moves: ', end = '')
        for move in board.legal_moves:
            print(move.uci() + ' ', end  = '')
        print('\n')
        
        # read move if human player
        if board.turn:
            input_uci = input('What move would you like to play?\n')
            playermove = chess.Move.from_uci(input_uci)
            if playermove in board.legal_moves:
                board.push(playermove)
        # generate move for ai
        else:
            aimove = random.choice([move for move in board.legal_moves])
            board.push(aimove)

# run the main function
if __name__ == '__main__':
    main()