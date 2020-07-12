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
    def __init__(self, board_fen=chess.STARTING_FEN, c=2, **kwargs): 
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
        self.c = c

        # game_node_dict will retain a map of board_fen --> StateNode
        # self.game_node_dict = {}
        # self.game_node_dict[board_fen] = self.cur_node
        # self.game_edge_dict = {}

    def compute_ucb1(self, edge):
        '''
        Helper Method:
        Computes the UCB1 formula. Returns infinity if nodes haven't been visited.
        This way we'll consider each node at the we consider every node at least once 
        before actually starting to compute true UCB1 values. We can probabilistically weight this 
        according to recommendations by an NN 
        '''
        if edge.N == 0:
            return np.inf
        else:
            return edge.W/edge.N + self.c * np.sqrt(np.log(self.tree.root.visits)/edge.N)

    def select_node_explore(self, node):
        '''
        Helper Method: selects node in mcts exploration lookahead, based on ucb criterion
        '''
        ties = []
        max_ucb = -np.inf
        for edge in node.out_edges:

            ucb = self.compute_ucb1(edge)
            if ucb > max_ucb:
                ties = [edge]
                max_ucb = ucb
            elif ucb == max_ucb:
                ties.append(edge)
        
        return random.choice(ties)

    def re_root_tree(self, fen, edge):
        '''
        Helper method: re-roots the mcts tree at that board state
        '''
    
        if self.tree.nodes[fen] is not None:
            self.tree.root = self.tree.nodes[fen]
        else:
            # generate new root
            new_root = GraphNode(fen, [edge])
            self.tree.root = new_root
    
    def reset_board_and_tree(self, fen):
        '''
        Helper method: resets tree and board state for a new game
        '''
        self.board = chess.Board(fen)
        self.re_root_tree(fen, None)

    def play_move(self):
        '''
        Main overhead method: runs mcts search to get move and updates tree and internal board state
        returns val, improved policy to be used in main training script as a training example
        We could alternatively train the NN in this method, but we wouldn't be able to compile experiences from multiple MCTS agents
        '''
        val, improved_policy = self.search_and_get_mcts_improved_policy()

        moves = [x[1] for x in improved_policy]
        probs = [x[0] for x in improved_policy]

        aimove = np.random.choice(moves, p=probs)
        print('AI PLAYS', aimove.move)

        self.board.push(aimove.move)
        fen = self.board.fen()

        self.re_root_tree(fen, aimove)
        return aimove, val, improved_policy
        


    def search_and_get_mcts_improved_policy(self, temp=0.7, num_iterations = 300):
        '''
        Main Search Method: uses MCTS search to produce value and improved policy given a root state of tree
        '''
       
        turn = chess.Board(self.tree.root.board).turn
        val = self.search(self.tree.root, turn)

        improved_policy = []
        sum = 0
        for edge in self.tree.root.out_edges:
            # temp is a hyperparameter called temperature that controls the degree of exploration/exploitation. Set arbitrarily to 0.7 for now
            improved_policy.append(edge.N ** temp, edge)
            sum += edge.N ** temp

        improved_policy = [(x[0]/sum, x[1]) for x in improved_policy]

        # state + val and improved policy serve as a training example for the NN, so they're both passed even tho the agent will only use the improved policy
        return val, improved_policy

    def search(self, node, orig_turn):
        '''
        Helper Recursive Search Method: to take in node, give back win update
        orig_turn is the side of the agent--black or white
        
        TODO: update edges generated from gen_edges w/ nn policy and value
        '''
        board = chess.Board(node.board)
        if board.is_game_over:
            if board.turn == orig_turn:
                return 1
            else:
                return -1

        if len(node.out_edges) == 0:
            node.gen_edges(self.tree.edges)
            val, logits = self.policy_net(bitboard(node.board))
            # USE THE NN TO INIT VALUES HERE
            # INCLUDE Q VALUE IN EDGE
            return -val

        best_edge = self.select_node_explore(node)
        best_a = best_edge.action
        new_node = best_edge.dest

        val = self.search(new_node, orig_turn)

        # update values
        best_edge.N += 1
        best_edge.W += val

        return -val
        


if __name__ == '__main__':
    pass
