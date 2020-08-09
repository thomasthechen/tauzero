import chess
import torch
import numpy as np
import pandas as pd
import pdb
import random
import timeit
import sys

from weakref import ref, WeakValueDictionary, WeakSet 
from collections import OrderedDict, namedtuple
from utils import bitboard
from GameTree import *
from MoveNet import *

def puct(c, p, N_s_b, edge):
    return c * p * np.sqrt(N_s_b) * 1 / (1 + edge.N)

class MonteCarloAgent():
    def __init__(self, board_fen=chess.STARTING_FEN, c=0.4, **kwargs): 
        '''
        We take the current game state which is parameterized by its FEN string. 
        self.cur_node = the StateNode that houses the root of the game tree which we wish to explore
        self.black = bool, True if the agent is playing black 
        self.c = float, a hyperparameter controlling the degree of exploration in the UCB1 bandit formula
        '''
        self.policy_net = MoveNet(**kwargs)
        self.policy_net.load_state_dict(torch.load('./mn_value4.pth', map_location=torch.device('cpu')))
        self.tree = GameTree(board_fen)

        # retain the board
        self.c = c
        self.tau = 1.0

    def push_move(self, move):
        move = str(move)
        board = self.tree.root.board
        edge_key = Edge(board, move)
        self.tree.root.gen_edges(self.tree.edges)
        self.tree.root = self.tree.root.out_edges[edge_key]
        self.tree.root.gen_nodes(self.tree.nodes)
        self.tree.root = self.tree.root.dest
 
    def select_move(self, num_searches=300):
        policy = []

        for _ in range(num_searches):
            self.tree_search()

        board = chess.Board(self.tree.root.board)
        temp = 1e-5 if board.fullmove_number > 30 else 1e-1
        N_s_b_t = 0.0
        for k in self.tree.root.out_edges.keys():
            N_s_b_t += pow(self.tree.root.out_edges[k].N, 1/temp)

        key = None
        max_prob = 0.0
        for k in self.tree.root.out_edges.keys():
            prob = pow(self.tree.root.out_edges[k].N, 1/temp) / N_s_b_t
            assert self.tree.root.out_edges[k].action == k.a
            policy.append(((self.tree.root.out_edges[k].state, self.tree.root.out_edges[k].action), prob))
            
            '''
            if prob >= max_prob:
                max_prob = prob
                key = k
            '''
        idx = np.random.choice(len(policy), p=[x[1] for x in policy])
        key = Edge(s=policy[idx][0][0], a=policy[idx][0][1])
        # print('Agent chooses {}'.format(key))

        Q = self.tree.root.out_edges[key].Q

        self.tree.root.out_edges[key].gen_nodes(self.tree.nodes)
        self.tree.root = self.tree.root.out_edges[key].dest
        # return for training
        return key, Q, policy

    # runs a tree search rollout and update steps
    def tree_search(self):
        # we use these references to reduce method signature length
        tree = self.tree
        root = self.tree.root
        net = self.policy_net

        # we will enable trainable gradients later
        net.eval()
        torch.autograd.set_grad_enabled(False)

        root.gen_edges(tree.edges)
        cur_node = root
        is_white = chess.Board(root.board).turn

        # method to run net; lambda to save time
        move_probs = lambda f, n=net, b=bitboard: n(b(f).float())
 
        # in tree phase of the search
        selected_leaf = False

        node_stack = []

        val = 0

        while (not selected_leaf):
            # TODO: Check if state is checkmated or stalemated
            bd = chess.Board(cur_node.board)
            if bd.is_checkmate():
                break
            elif bd.is_stalemate():
                # note val ranges from -1 to 1
                break
            # compute edges
            cur_node.gen_edges(tree.edges)

            # global action statistics
            N_s_b = 0 
            for k in cur_node.out_edges.keys():
                N_s_b += cur_node.out_edges[k].N 

            # compute the move probabilities # use nn prob not collected prob to decide search?
            val, logits = move_probs(cur_node.board)
            probs, moves = mask_invalid(chess.Board(cur_node.board), logits)

            # a_t contains the max ucb selection
            a_t = torch.tensor(-float('inf'))
            next_edge = None 

            # iterate over the moves, looking for the max a_t
            for i, move in enumerate(moves):
                # generate the edge key
                edge_key = Edge(cur_node.board, move)
                # pdb.set_trace()
 
                # compute candidate a_t as PUCT + Q
                # print(puct(self.c, probs[i], N_s_b, cur_node.out_edges[edge_key]).item(), cur_node.out_edges[edge_key].Q)
                a_t_i = puct(self.c, probs[i], N_s_b, cur_node.out_edges[edge_key]) + torch.tensor(cur_node.out_edges[edge_key].Q) # torch.tensor

                # set new maximum if clear winner
                if a_t_i.item() > a_t.item():
                    a_t = a_t_i
                    next_edge = edge_key

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


        bd = chess.Board(cur_node.board)
        if bd.is_checkmate():
            val = torch.tensor(-1)
        elif bd.is_stalemate():
            # note val ranges from -1 to 1
            val = torch.tensor(0)
        else:
            # backup phase: generate edges, val, logits, probs, and moves
            cur_node.gen_edges(tree.edges)
            val, logits = move_probs(cur_node.board)
            probs, moves = mask_invalid(chess.Board(cur_node.board), logits)

            # expand the node and initialize values of P
            for i, move in enumerate(moves):
                edge_key = Edge(cur_node.board, move)
                cur_node.out_edges[edge_key] = GraphEdge(cur_node.board, move) 
                cur_node.out_edges[edge_key].P = probs[i]

        # backup phase
        ## TODO HAVE TO ACCOUNT FOR TURN OF VAL RETURNED AND CURR BOARD 

        end_turn = chess.Board(cur_node.board).turn
        # if checkmated and these two are opposites, then the start state is a winning state, else losing

        while (len(node_stack) != 0):
            cur_node = node_stack.pop()
            if isinstance(cur_node, GraphNode):
                cur_node.visits += 1
            else:
                start_turn = chess.Board(cur_node.state).turn
                value = val.item() * (2 * (start_turn == end_turn) - 1)
        
                cur_node.W += value
                cur_node.N += 1.0
                cur_node.Q = cur_node.W / cur_node.N

if __name__ == '__main__':
    x = MonteCarloAgent()
    num_its = int(sys.argv[1])
    duration = timeit.timeit("x.tree_search()", number=num_its, globals=locals())
    print("%i searches took %.4f s, (%f/s)" % (num_its, duration, num_its/duration))
