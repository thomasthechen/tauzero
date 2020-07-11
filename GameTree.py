import chess
import numpy as np
import pandas as pd
import random
from collections import OrderedDict, namedtuple

# edge is denoted by a state and action 
Edge = namedtuple('Edge', ['s', 'a'])

# lambda to check if iterable
is_iterable = lambda val: hasattr(val,'__iter__') or hasattr(val,'__getitem__')

class GraphEdge():
    def __init__(self, board_fen, action):
        '''
        A state node. 
        board_fen = current state, as characterized by a FEN string
        n = the current visit count 
        w = the number of wins that have been won from this node 
        nextSibling = next sibling state node 
        firstChild = first child state node 
        '''
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0
        self.state = board_fen
        self.action = action

    def gen_node(self, all_node_dict):
        board = chess.Board(self.state)
        node_key = board.push(self.action).fen()
        node = all_node_dict.get(node_key, None)
        if node:
            self.node = node
            self.node.update_indeg(self)
        else:
            self.node = GraphNode(node_key, [self])

    def __str__(self):
        return 'N:{} W:{} Q:{} P:{} source:{} action:{}'.format(self.N, self.W, self.Q, self.P, self.state, self.action)

class GraphNode():
    def __init__(self, board_fen, prev_edges):
        self.edges = []

        assert is_iterable(prev_edges)

        self.prev_edges = prev_edges 
        self.visits = 0
        self.board = board_fen
        self.indeg = len(self.prev_edges)
        self.outdeg = len(self.edges)

    def gen_edges(self, all_edge_dict): 
        '''
        Generate all neighboring edges for a given node, indexed by the Edge Key.
        Params
        :all_edge_dict: mapping from tuple --> edge for all edges in the graph (check to preserve edge statistics) 
        '''
        board = chess.Board(self.board)

        # New edge for each move (Edge is state + action pair)
        for move in board.legal_moves:
            edge_key = Edge(board.push(move).fen(), str(move))
            next_edge = all_edge_dict.get(edge_key, None)
            
            # edge exists
            if next_edge:
                self.edges.append(all_edge_dict[edge_key])

            # edge doesn't exist; create it and append
            else:
                all_edge_dict[edge_key] = GraphEdge(edge_key.s, edge_key.a)
                self.edges.append(all_edge_dict[edge_key])

            board.pop()

        self.outdeg += len(edges)

    def update_indeg(self, edge):
        self.indeg += 1
        self.prev_edges.append(edge)

    def __str__(self):
        return 'Visits: {} Board: {} Last Edge: {}'.format(self.visits, self.board, self.prev_edge)

class GameTree():
    def __init__(self, board_fen):
        self.cur_root = GraphNode(board_fen, [])
        self.edges = OrderedDict()
        self.nodes = OrderedDict()
        self.nodes[board_fen] = self.cur_root

    def free_node(self, node_to_free):
        # recursive routine to identify nodes to remove from the game dictionary 
        pass

