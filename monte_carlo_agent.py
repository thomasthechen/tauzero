import chess
import torch
import numpy as np
import pandas as pd
import pdb
import random

from weakref import ref, WeakValueDictionary, WeakSet 
from collections import OrderedDict, namedtuple
from utils import bitboard
from GameTree import *
from MoveNet import *

def ucb(edge):
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

def puct(c, p, N_s_b, edge):
    return c * p * np.sqrt(N_s_b) * 1 / (1 + edge.N)

class MonteCarloAgent():
    def __init__(self, board_fen=chess.STARTING_FEN, c=2, **kwargs): 
        '''
        We take the current game state which is parameterized by its FEN string. 
        self.cur_node = the StateNode that houses the root of the game tree which we wish to explore
        self.black = bool, True if the agent is playing black 
        self.c = float, a hyperparameter controlling the degree of exploration in the UCB1 bandit formula
        '''
        self.policy_net = MoveNet(**kwargs)
        self.tree = GameTree(board_fen)

        # retain the board
        self.c = c
        self.tau = 1.0
 
    def select_move(self, num_searches=300):
        for _ in range(num_searches):
            self.tree_search()

        root = ref(self.root)
        board = chess.Board(root.board)
        temp = 1e-5 if board.fullmove_number > 30 else 1.0
        N_s_b_t = 0.0
        for x in root.out_edges.keys():
            N_s_b_t += pow(root.out_edges[k].N, 1/temp)

        key = None
        max_prob = 0.0
        for k in root.out_edges.keys():
            prob = pow(root.out_edges[k].N, temp) / N_s_b_t
            if prob >= max_prob:
                max_prob = prob
                key = k
        print('Agent chooses {}'.format(key))
        next_edge = ref(root.out_edges[key])
        next_edge.gen_nodes(self.tree.nodes)
        board.push(next_edge.a)
        self.board = board.fen()
        self.root = next_edge.dest

    # runs a tree search rollout and update steps
    def tree_search(self):        
        # we use these references to reduce method signature length
        tree = self.tree
        root = self.tree.root
        net = self.policy_net
        root.gen_edges(tree.edges)
        cur_node = root
        is_white = chess.Board(root.board).turn

        # method to run net; lambda to save time
        move_probs = lambda f, n=net, b=bitboard: n(b(f).float())
 
        # in tree phase of the search
        selected_leaf = False

        node_stack = []
        while (not selected_leaf):
            # compute edges
            cur_node.gen_edges(tree.edges)
             
            # global action statistics
            N_s_b = 0 
            for k in cur_node.out_edges.keys():
                N_s_b += cur_node.out_edges[k].N 

            # compute the move probabilities
            val, logits = move_probs(cur_node.board)
            probs, moves = mask_invalid(chess.Board(cur_node.board), logits)            

            # a_t contains the max ucb selection
            a_t = torch.tensor(-float('inf'))
            next_edge = None 
            i_t = 0 
            # iterate over the moves, looking for the max a_t
            for i, move in enumerate(moves):
                # generate the edge key
                edge_key = Edge(cur_node.board, move)
                # pdb.set_trace()
                
                # compute candidate a_t as PUCT + Q
                a_t_i = puct(c, probs[i], N_s_b, cur_node.out_edges[edge_key]) + torch.tensor(cur_node.out_edges[edge_key].Q) # torch.tensor

                # set new maximum if clear winner
                if a_t_i.item() > a_t.item():
                    a_t = a_t_i
                    next_edge = edge_key
                    i_t = 0

                # break ties randomly
                elif a_t_i.item() == a_t.item():
                    if random.choice([0, 1]):
                        a_t = a_t_i
                        next_edge = edge_key

            # set the next edge
            next_edge = cur_node.out_edges[next_edge]
            next_edge.gen_nodes(tree.nodes)
            cur_node = next_edge.dest
 
            node_stack.append(next_edge)
            node_stack.append(cur_node)

            # break out of the enclosing loop
            if cur_node.visits == 0:
                selected_leaf = True

        # backup phase: generate edges, val, logits, probs, and moves
        cur_node.gen_edges()
        val, logits = move_probs(cur_node.board)
        probs, moves = mask_invalid(chess.Board(cur_node.board), logits)

        # expand the node and initialize values of P
        for i, move in enumerate(moves):
            edge_key = Edge(cur_node.board, move)
            cur_node.out_edges[edge_key].P = probs[i]

        # backup phase
        while (len(node_stack) != 0):
            cur_node = node_stack.pop()
            if isinstance(cur_node, GraphNode):
                cur_node.visits += 1
            else:
                cur_node.W += val.item()
                cur_node.N += 1.0
                cur_node.Q = W / N

if __name__ == '__main__':
    x = MonteCarloAgent()
    x.tree_search()
