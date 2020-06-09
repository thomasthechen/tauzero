import chess

class State(object):
  def __init__(self, board=None):
    if board is None:
      self.board = chess.Board()
    else:
      self.board = board

  def key(self):
    return (self.board.board_fen(), self.board.turn, self.board.castling_rights, self.board.ep_square)

  def serialize(self):
    import numpy as np
    assert self.board.is_valid()

    bstate = np.zeros((13,8,8), np.uint8)
    for i in range(64):
      pp = self.board.piece_at(i)
      if pp is not None:
        #print(i, pp.symbol())
        k = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, "p": 6, "n":7, "b":8, "r":9, "q":10, "k": 11}[pp.symbol()]
        bstate[k][int(i /8)][i % 8] = 1
    '''
    if self.board.has_queenside_castling_rights(chess.WHITE):
      assert bstate[0] == 4
      bstate[0] = 7
    if self.board.has_kingside_castling_rights(chess.WHITE):
      assert bstate[7] == 4
      bstate[7] = 7
    if self.board.has_queenside_castling_rights(chess.BLACK):
      assert bstate[56] == 8+4
      bstate[56] = 8+7
    if self.board.has_kingside_castling_rights(chess.BLACK):
      assert bstate[63] == 8+4
      bstate[63] = 8+7
    
    if self.board.ep_square is not None:
      assert bstate[self.board.ep_square] == 0
      bstate[self.board.ep_square] = 8
    '''
    # 4th column is who's turn it is
    bstate[12] = (self.board.turn*1.0)

    # 257 bits according to readme
    return bstate

  def edges(self):
    return list(self.board.legal_moves)

if __name__ == "__main__":
  s = State()