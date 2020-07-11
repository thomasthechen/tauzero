import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ReLU, LeakyReLU, Linear, Dropout, BatchNorm2d, LayerNorm 
from GameTree import *

TOTAL_PIECES = 12
BOARD_SIZE = 8 ** 2
BOARD_DIM = 8
BOARD_DICT = {'a': 0,
		'b': 1,
		'c': 2,
		'd': 3,
		'e': 4,
		'f': 5,
		'g': 6,
		'h': 7}
INV_BOARD_DICT = {v: k for k, v in BOARD_DICT.items()}

def mask_invalid(board, logits, device='cpu'):
	
	def parse_move(move_str, device='cpu'):
		let_1 = BOARD_DICT.get(move_str[0])
		num_1 = int(move_str[1])

		let_2 = BOARD_DICT.get(move_str[2])
		num_2 = int(move_str[3])

		return (num_1 + let_1 * (BOARD_DIM ** 1)), (num_2 +  let_2 * (BOARD_DIM ** 1))

	# we incur a bit of additional overhead
	logits = logits.view(BOARD_SIZE, BOARD_SIZE)

	move_mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
	
	# number of legal moves	
	k = 0
	for move in board.legal_moves:
		coord = parse_move(str(move))
		move_mask[coord[0]][coord[1]] = 1
		k += 1	
		# print('Move: {} C1: {} C2: {}'.format(move, coord[0], coord[1]))

	move_mask = ~move_mask
	logits[move_mask] = -1 * 10 ** 10 

	logits =  torch.softmax(logits.view(-1), dim=0).view(BOARD_SIZE, BOARD_SIZE)

	probs, coords = torch.topk(logits.reshape(-1), k=k)

	# print(coords)

	moves = []

	for x in coords:
		source_move = x.item() // BOARD_SIZE
		source_letter = INV_BOARD_DICT.get(source_move // BOARD_DIM)
		source_coord = 	source_move % BOARD_DIM

		end_move = x.item() % BOARD_SIZE
		end_letter = INV_BOARD_DICT.get(end_move // BOARD_DIM)
		end_coord = end_move % BOARD_DIM
		moves.append(''.join([source_letter, str(source_coord), end_letter, str(end_coord)]))

	return probs, moves


class DoubleConv(nn.Module):
	def __init__(self, in_ch, out_ch, p=0.2, filter_dim=5, stride=1, padding=True):	
		super(DoubleConv, self).__init__()
		all_layers = []
		extra_px = 0

		if padding:
			extra_px += filter_dim // 2

		all_layers.append(Conv2d(in_ch, out_ch, kernel_size=filter_dim, stride=stride, padding=extra_px))
		all_layers.append(BatchNorm2d(out_ch))
		all_layers.append(ReLU())
		all_layers.append(Dropout(p=p))

		all_layers.append(Conv2d(out_ch, out_ch, kernel_size=filter_dim, stride=stride, padding=extra_px))
		all_layers.append(BatchNorm2d(out_ch))
		all_layers.append(ReLU())
		all_layers.append(Dropout(p=p))

		self.block = nn.Sequential(*all_layers)

	def forward(self, x):
		return self.block(x)


class MoveNet(nn.Module):
	def __init__(self, in_ch=13, num_blocks=3, convs_per_block=5, padding=True, filter_dim=5, res_net=True, loss='relu',
		dropout=0.2, norm='batch', first_ch=64, scale=2):
		# note: convs per block is actually double convs per block. 
		super(MoveNet, self).__init__()
		self.res_net = res_net
		blocks = []

		# first conv
		first_layer = nn.Sequential(DoubleConv(in_ch, first_ch, p=dropout, filter_dim=filter_dim, padding=padding))
		for _ in range(convs_per_block - 1):
			first_layer = nn.Sequential(*first_layer, DoubleConv(first_ch, first_ch, p=dropout, filter_dim=filter_dim, padding=padding))

		for i in range(num_blocks - 1):
			cur_block = []
			in_ch = first_ch * (scale ** i)
			out_ch = first_ch * (scale ** (i + 1))
			for j in range(convs_per_block):
				if j == 0:	
					cur_block.append(DoubleConv(in_ch, out_ch, p=dropout, filter_dim=filter_dim, padding=padding))
				else:
					cur_block.append(DoubleConv(out_ch, out_ch, p=dropout, filter_dim=filter_dim, padding=padding))		

			blocks.extend(cur_block)

		every_move = BOARD_SIZE ** 2

		# some dimensionality reduction before translating to logits
		self.fc1 = Linear(first_ch * 2 ** (num_blocks-1), 128)
		self.value = Linear(BOARD_SIZE * 128, 1)	
		self.prob_logits = Linear(BOARD_SIZE * 128, every_move)
		self.blocks = nn.Sequential(*first_layer, *blocks)
		self.convs_per_block = convs_per_block

	def forward(self, x):
		prev_x = []	
		for idx, layer in enumerate(self.blocks):
			if self.res_net:
				if idx != 0: prev_x.append(x)
			x = layer(x)
			if self.res_net and idx != len(self.blocks) - 1 and idx % self.convs_per_block != 0:
				if idx != 0: x += prev_x[-1]

		# swap channels dimension to the end
		x = x.permute(0, 2, 3, 1)
		# print(x.size())	
		x = F.relu(self.fc1(x))
		x = x.view(-1)	# flatten vector
		val = self.value(x) # scalar valued output 
		logits = self.prob_logits(x) # NOTE: Don't forget to softmax after masking out invalid moves!	

		return val, logits
