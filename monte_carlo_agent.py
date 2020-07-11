import chess
import torch
import numpy as np
import pandas as pd
import random

from weakref import ref, WeakValueDictionary, WeakSet 
from collections import OrderedDict, namedtuple
from utils import bitboard
from GameTree import *
from MoveNet import *


class MonteCarloAgent():
    def __init__(self, board_fen=chess.STARTING_FEN, black = False, **kwargs): 
        '''
        We take the current game state which is parameterized by its FEN string. 
        self.board = the current chess.Board object which we'll use to handle game logic
        self.cur_node = the StateNode that houses the root of the game tree which we wish to explore
        self.black = bool, True if the agent is playing black 
        self.c = float, a hyperparameter controlling the degree of exploration in the UCB1 bandit formula
        '''
        self.policy_net = MoveNet(**kwargs)
        self.tree = GameTree(board_fen)

        # retain the board
        self.board = chess.Board(board_fen)
        self.black = black
        self.c = c
        
        # cur_node will retain the current base state node 
        self.cur_node = StateNode(self.board)
        
        # game_node_dict will retain a map of board_fen --> StateNode
        self.game_node_dict = {}
        self.game_node_dict[board_fen] = self.cur_node
    
    def compute_ucb1(self, node):
        '''
        Computes the UCB1 formula. Returns infinity if nodes haven't been visited.
        This way we'll consider each node at the we consider every node at least once 
        before actually starting to compute true UCB1 values. We can probabilistically weight this 
        according to recommendations by an NN 
        ''' 
       return

    def get_mcts_policy(self, temp=0.7):
        return 
            
    def random_rollout_policy(self, board_fen):
        return

    def nn_rollout_eval(self, board_fen):
        # once it gets to an unexplored node not part of graph, then just use nn to evaluate that position
        # saves a lot of time, will work as nn gets better trained
        return

    def rollout(self, num_iterations = 300):
        '''
        This method performs a rollout on the current game state. It will select from the child nodes 
        a node with the highest UCB1 value, breaking ties randomly. It will then sample states according to some pre-specified policy
        until it reaches the end, at which point it will update the visit count with: +1, 0.5, 0 for a win, tie, or loss, respectively
        num_iterations = number of rollouts to perform
        '''
        return 

    def rollout_get_policy(self, num_iterations=500):
        '''
        This method performs a rollout on the current game state. It will select from the child nodes 
        a node with the highest UCB1 value, breaking ties randomly. It will then sample states according to some pre-specified policy
        until it reaches the end, at which point it will update the visit count with: +1, 0.5, 0 for a win, tie, or loss, respectively
        num_iterations = number of rollouts to perform
        '''

        return


if __name__ == '__main__':
    pass
