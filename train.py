'''
@author George Hotz
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim
from value_approximator import Net

class ChessValueDataset(Dataset):
  def __init__(self):
    dat = np.load("processed/dataset_25M.npz")
    self.X = dat['arr_0']
    self.Y = dat['arr_1']
    print("loaded", self.X.shape, self.Y.shape)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return (self.X[idx], self.Y[idx])


if __name__ == "__main__":
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