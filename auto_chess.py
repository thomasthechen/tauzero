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

def play_game(fen):

    # initialize board
    board = chess.Board(fen)
    
    while not board.is_game_over():
        # display whose turn it is
        # if board.turn:
        #     print('White\'s Turn\n')
        # else:
        #     print('Black\'s Turn\n')
        
        # display board
        # print(board)
        # print('')
        
        # display possible moves
        # print('Possible moves:', end = ' ')
        # for move in board.legal_moves:
        #     print(move.uci(), end  = ' ')
        # print('\n')
        
        aimove = random.choice([move for move in board.legal_moves])
        board.push(aimove)

    # print final board
    # print(board)
    # print('')

    # print result
    if board.is_checkmate():
        print(f'{"Black" if board.turn else "White"} wins.')
    elif board.is_stalemate():
        print('Draw. Stalemate.')
    elif board.is_insufficient_material():
        print('Draw. Insufficient material.')
    elif board.is_fivefold_repetition():
        print('Draw. Fivefold repetition.')
    elif board.is_seventyfive_moves():
        print('Draw. Seventy-five moves.')

    return board.result()

def main():

    # STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    fen = chess.STARTING_FEN
    whitescore = 0.0
    blackscore = 0.0

    for i in range(1000):
        print(f'Game {i + 1}: ', end = '')
        result = play_game(fen)
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