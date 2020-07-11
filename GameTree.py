import chess
import numpy as np
import pandas as pd
import random
from weakref import ref, WeakValueDictionary, WeakSet 
from collections import OrderedDict, namedtuple

# edge is denoted by a state and action 
Edge = namedtuple('Edge', ['s', 'a'])

# lambda to check if iterable
is_iterable = lambda val: hasattr(val,'__iter__') or hasattr(val,'__getitem__')

class GraphEdge():
    def __init__(self, board_fen, action, source):
        '''
        A state node. 
        board_fen = source node state, as characterized by a FEN string
        action = action to be taken from the source board state
        n = the current visit count 
        w = the number of wins that have been won from this node 
        '''
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0
        self.state = board_fen
        self.action = action
        self.source = ref(source)  # weak references to the source to prevent circular references
        self.dest = None

    def gen_nodes(self, all_node_dict):
        '''
        Generate the destination node. If the destination node exists, we retrieve it from the game dict. 
        Otherwise, we create the node, add it to the game dict. The node is then updated to ensure that 
        this edge is in its list of "in_edges", or edges pointing to the node in the directed graph.
        
        Params
        :all_node_dict: dictionary mapping from fen --> nodes for previously considered and 
        retained nodes in the tree.
        '''
        
        # already generated current node        
        if self.node is not None:
            return

        # create the fen string for the destination node    
        board = chess.Board(self.state)
        node_key = board.push(self.action).fen()
        node = all_node_dict.get(node_key, None)
        
        # node was previously considered and retained 
        if node:
            self.dest = node  # update edge destination
            self.dest.update_indeg(self)  # update source edges from the node

        # node is new; we create it and store it in the game dict
        else:
            node = GraphNode(node_key, [self]) 
            all_node_dict[node_key] = node   # add node to the game dict set 
            self.dest = all_node_dict.get(node_key)  # update edge destination

    def __str__(self):
        return 'N:{} W:{} Q:{} P:{} source:{} action:{}'.format(self.N, self.W, self.Q, self.P, self.state, self.action)

    # order objects by concatenation of fen + action
    def __eq__(self, N2):
        return self.state == N2.state and self.action == N2.action

    def __gt__(self, N2):
        return (self.state + self.action) > (N2.state + N2.action)

    def __lt__(self, N2):
        return (self.state + self.action) < (N2.state + N2.action)

class GraphNode():
    def __init__(self, board_fen, prev_edges):
        '''
        GraphNode objects store nodes corresponding to game states and all references pointing to all 
        edges that point at the object. References to edges / nodes higher in the tree are weak by default 
        to prevent circularity in references that impedes garbage collection.

        :in_edges: WeakSet containing weak references to all edges that can produce this state
        :out_edges: list containing strong references to all edges that can be produced from this node
        :board: the board fen string that represents the current state
        :visits: node visit count
        :indeg: in-degree of this node in the game tree
        :outdeg: out-degree of this node in the game tree
        '''        
        assert is_iterable(prev_edges)
        self.in_edges = WeakSet(prev_edges)
        self.out_edges = []
        self.visits = 0
        self.board = board_fen
        self.indeg = len(self.in_edges)
        self.outdeg = len(self.out_edges)

    def gen_edges(self, all_edge_dict): 
        '''
        Generate all neighboring edges for a given node, indexed by the Edge Key.
        
        Params
        :all_edge_dict: mapping from tuple --> edge for all edges in the graph (check to preserve edge statistics) 
        '''
        # edges have already been generated
        if len(self.out_edges) > 0:
            return

        board = chess.Board(self.board)

        # New edge for each move (Edge is state + action pair)
        for move in board.legal_moves:
            board.push(move)
            edge_key = Edge(board.fen(), str(move))
            next_edge = all_edge_dict.get(edge_key, None)
            
            # edge exists
            if next_edge:
                self.out_edges.append(all_edge_dict[edge_key])

            # edge doesn't exist; create it and append
            else:
                edge = GraphEdge(edge_key.s, edge_key.a, self)
                all_edge_dict[edge_key] = edge
                self.out_edges.append(all_edge_dict[edge_key])

            board.pop()

        self.outdeg += len(self.out_edges)

    def update_indeg(self, edge): 
        '''
        This method will add a weak reference to an edge that points to this Node. If the Node already exists
        and we reach it from a different state action pair, we can retain it.
        
        Params
        :edge: the edge to add to the set of in-edges
        ''' 
        if edge not in self.in_edges:
            self.indeg += 1
            self.in_edges.add(edge)

    def __str__(self):
        return 'Visits: {} Board: {} Last Edge: {}'.format(self.visits, self.board, [x for x in self.in_edges])

class GameTree():
    def __init__(self, board_fen):
        self.root = GraphNode(board_fen, [])
        self.edges = WeakValueDictionary()
        self.nodes = WeakValueDictionary()
        self.nodes[board_fen] = self.root
