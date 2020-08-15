import chess
import torch
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch import optim
from weakref import ref, WeakValueDictionary, WeakSet 
from collections import OrderedDict, namedtuple

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

def bitboard(fen, device='cpu'):
    '''
    One-hot encode our board in the desired format. Note that we have to add an additional 
    dimension to accommodate PyTorch's requirement that there be a batch dimension. 
    We can do this using unsqueeze(). We can also optionally Cythonize this for 
    speed improvements. 
    '''
    board = chess.Board(fen)
    assert board.is_valid()
    bstate = np.zeros((13,8,8), np.uint8)
    for i in range(64):
        pp = board.piece_at(i)
        if pp is not None:
            #print(i, pp.symbol())
            k = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, "p": 6, "n":7, "b":8, "r":9, "q":10, "k": 11}[pp.symbol()]
            bstate[k][int(i /8)][i % 8] = 1

    # 4th column is who's turn it is
    bstate[12] = (board.turn*1.0)

    # 257 bits according to readme, add extra dimension for batch size
    return torch.tensor(bstate, dtype=torch.float64, device=device).unsqueeze(dim=0)

'''
EVERYTHING BENEATH THIS LINE IS OLD, GRACIOUSLY PROVIDED BY AUTHOR GEORGE HOTZ.
IT IS INCLUDED SOLELY FOR REFERENCE, AND IS NOT USED IN THE FINAL PROEJCT. 
'''
class ChessValueDataset(Dataset):
    '''
    From George Hotz's implementation. Solely here for reference
    '''
    def __init__(self):
        dat = np.load("processed/dataset_25M.npz")
        self.X = dat['arr_0']
        self.Y = dat['arr_1']
        print("loaded", self.X.shape, self.Y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def train_george_hotz_model():
    '''
    From George Hotz's implementation. Solely here for reference
    '''
    device = "cpu"

    chess_dataset = ChessValueDataset()
    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=256, shuffle=True)
    model = Net()
    optimizer = optim.Adam(model.parameters())
    floss = nn.MSELoss()

    if device == "cuda":
        model.cuda()

    model.train()

    for epoch in range(10):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(-1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()
            #print(data.shape, target.shape)
            optimizer.zero_grad()
            output = model(data)
            #print(output.shape)

            loss = floss(output, target)
            loss.backward()
            optimizer.step()
            
            all_loss += loss.item()
            num_loss += 1

        print("%3d: %f" % (epoch, all_loss/num_loss))
        torch.save(model.state_dict(), "trained_models/value.pth")
