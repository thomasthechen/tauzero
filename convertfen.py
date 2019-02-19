import chess
import numpy as np

fen = chess.STARTING_FEN
splitfen = fen.split()

# board states using ascii representations of piece characters
# A1 = 0 ... H8 = 63
board1d = np.zeros([1, 64])
board2d = np.zeros([8, 8])
# white = 0, black = 1
active = -1
# castle availability
# binary representation so that KQkq becomes 1111 base 2
castle = -1
# en passant square
enpassant = -1
# halfmoves counter
halfmoves = -1
# fullmoves counter
fullmoves = -1

row = 7
col = 0

# assign board positions
for c in splitfen[0]:
    if c == ' ': break
    if c == '/': 
        row -= 1
        col = 0
        continue
    if c.isdigit():
        col += int(c)
    else:
        board2d[row, col] = ord(c)
        col += 1

# update 1d baord
board1d = np.reshape(board2d, [1, 64])
    
# check active color
if splitfen[1] == 'w': active = 0
else: active = 1

# check castling
if splitfen[2] == '-':
    castle = 0
else:
    castle = 0
    if 'K' in splitfen[2]: castle += 1
    castle *= 2
    if 'Q' in splitfen[2]: castle += 1
    castle *= 2
    if 'k' in splitfen[2]: castle += 1
    castle *= 2
    if 'q' in splitfen[2]: castle += 1

# check en passant
if splitfen[3] == '-': enpassant = 0
else:
    enpassant = (int(splitfen[3][1]) - 1) * 8 + (ord(splitfen[3][0]) - ord('a'))

# update halfmoves
halfmoves = int(splitfen[4])

# update fullmoves
fullmoves = int(splitfen[5])

# print(board2d)
# print(board1d)
# print(active)
# print(castle)
# print(enpassant)
# print(halfmoves)
# print(fullmoves)