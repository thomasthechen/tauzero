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
        # display board
        print(board)
        # display possible moves
        print('Possible moves: ', end = '')
        for move in board.legal_moves:
            print(board.san(move) + ' ', end  = '')
        break

# run the main function
if __name__ == '__main__':
    main()