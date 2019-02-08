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
import chess.svg
import random

def play_game(fen):

    # initialize board
    board = chess.Board(fen)
    
    while not board.is_game_over():
        # display whose turn it is
        if board.turn:
            print('White\'s Turn\n')
        else:
            print('Black\'s Turn\n')
        
        # display board
        print(board)
        print('')
        
        # display possible moves
        print('Possible moves:', end = ' ')
        for move in board.legal_moves:
            print(move.uci(), end  = ' ')
        print('\n')
        
        aimove = random.choice([move for move in board.legal_moves])
        board.push(aimove)

    # print final board
    print(board)
    print('')

    # print result
    if board.is_checkmate():
        print(f'Game over. {"Black" if board.turn else "White"} wins.\n')
    elif board.is_stalemate():
        print('Game over. Stalemate.\n')
    elif board.is_insufficient_material():
        print('Game over. Insufficient material.\n')
    elif board.is_fivefold_repetition():
        print('Game over. Fivefold repetition.\n')
    elif board.is_seventyfive_moves():
        print('Game over. Seventy-five moves.\n')

    return board.result()

def main():

    # STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    fen = chess.STARTING_FEN
    whitescore = 0.0
    blackscore = 0.0

    for _ in range(100):
        result = play_game(fen)
        print(result)
        if result == '1-0':
            whitescore += 1.0
        elif result == '0-1':
            blackscore += 1.0
        else:
            whitescore += 0.5
            blackscore += 0.5

    # print final score
    print(f'Score: {whitescore}-{blackscore}\n')

# run the main function
if __name__ == '__main__':
    main()