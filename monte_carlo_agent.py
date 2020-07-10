import chess
import numpy as np
import pandas as pd
import random

class StateNode():
    def __init__(self, board_fen, move=None, nextSibling=None, firstChild=None, n=0, w=0):
        '''
        A state node. 
        board_fen = current state, as characterized by a FEN string
        n = the current visit count 
        w = the number of wins that have been won from this node 
        nextSibling = next sibling state node 
        firstChild = first child state node 
        '''
        self.nextSibling = nextSibling
        self.firstChild = firstChild
        self.n = 0
        self.w = 0
        self.board_fen = board_fen
        self.move = move


class MonteCarloAgent():
    def __init__(self, board_fen=chess.STARTING_FEN, black = False, c = 1.0):
        '''
        We take the current game state which is parameterized by its FEN string. 
        self.board = the current chess.Board object which we'll use to handle game logic
        self.cur_node = the StateNode that houses the root of the game tree which we wish to explore
        self.black = bool, True if the agent is playing black 
        self.c = float, a hyperparameter controlling the degree of exploration in the UCB1 bandit formula
        '''

        # retain the board
        self.board = chess.Board(board_fen)
        self.black = black
        self.c = c
        
        # cur_node will retain the current base state node 
        self.cur_node = StateNode(self.board)
        
        # game_node_dict will retain a map of board_fen --> StateNode
        self.game_node_dict = {}
        self.game_node_dict[board_fen] = self.cur_node
    
    def reset_cur_root(self, board=None):
        self.cur_node.n = 0
        self.cur_node.w = 0
        self.nextSibling = None
        self.firstChild = None
        self.board = board

    def generate_possible_children(self):
        '''
        This method returns nothing.
        Given the current root node, it will update the children such that we know all of the reachable 
        child nodes of the root, along with their visit counts and wins. 
        '''
        first = True
        prev_sibling = None
        for move in self.board.legal_moves:
            # update the board
            self.board.push(move)
            
            # first iteration, the previous sibling is the first child of the current root
            if first:
                self.cur_node.firstChild = StateNode(self.board.fen(), move=move)
                prev_sibling = self.cur_node.firstChild 
                first = False
            
            # on other iterations the "last sibling" is the last one that was set
            else:
                prev_sibling.nextSibling = StateNode(self.board.fen(), move=move)
                prev_sibling = prev_sibling.nextSibling
            
            #TODO: handle duplicate board states in the game state dict

            # pop from the board
            self.board.pop()

    def compute_ucb1(self, node):
        '''
        Computes the UCB1 formula. Returns infinity if nodes haven't been visited.
        This way we'll consider each node at the we consider every node at least once 
        before actually starting to compute true UCB1 values. We can probabilistically weight this 
        according to recommendations by an NN 
        '''
        
        if node.n == 0:
            return np.inf 

        return node.w / node.n + self.c * np.sqrt(np.log(self.cur_node.n) / node.n)

    def select_node(self):
        '''
        Perform node selection. This method should take the children of the root node 
        and compute the maximum UCB1 value of each, storing the values in the list called
        ties. It returns a random node of the nodes with the highest UCB1 value.
        '''

        prev_sibling = self.cur_node.firstChild
        ties = [prev_sibling]
        cur_max = self.compute_ucb1(prev_sibling)

        while prev_sibling.nextSibling is not None:
            prev_sibling = prev_sibling.nextSibling 
            ucb = self.compute_ucb1(prev_sibling)
            if ucb > cur_max:
                ties = [prev_sibling]
                cur_max = ucb
            elif ucb == cur_max:
                ties.append(prev_sibling)
            else:
                continue
        
        return random.choice(ties)

    def get_mcts_policy(self, temp=0.7):
        cur = self.cur_node.firstChild
        policy = [(cur.n, cur.move)]

        sum = 0
        while cur.nextSibling is not None:
            cur = cur.nextSibling
            policy.append((cur.n**temp, cur.move))
            sum += cur.n**temp
        
        
        # policy = [(x[0]/sum, x[1]) for x in policy]
        
        return policy
            

    def random_rollout_policy(self, board_fen):
        local_board = chess.Board(board_fen)

        for _ in range(75):
            local_board.push(random.choice(list(local_board.legal_moves)))
            if local_board.is_checkmate():
                # XOR: if white's turn and black, or if black's turn and white, we lost

                if (local_board.turn and self.black) or (not local_board.turn and not self.black): 
                    return 0
                
                # otherwise we won
                else:
                    return 1

        return 0.5

        
    '''
    TODO: NN ROLLOUT EVAL REPLACEMENT
    '''

    def nn_rollout_eval(self, board_fen):
        # once it gets to an unexplored node not part of graph, then just use nn to evaluate that position
        # saves a lot of time, will work as nn gets better trained
        pass

    def rollout(self, num_iterations = 300):
        '''
        This method performs a rollout on the current game state. It will select from the child nodes 
        a node with the highest UCB1 value, breaking ties randomly. It will then sample states according to some pre-specified policy
        until it reaches the end, at which point it will update the visit count with: +1, 0.5, 0 for a win, tie, or loss, respectively
        num_iterations = number of rollouts to perform
        '''

        for i in range(num_iterations):
            node = self.select_node()
            win_update = self.random_rollout_policy(node.board_fen)

            # update the local number of visits for visited and root node
            node.n += 1
            node.w += win_update
            self.cur_node.w += win_update
            self.cur_node.n += 1
        
        # return a node with the highest UCB1 value
        return self.select_node().move

    def rollout_get_policy(self, num_iterations=500):
        '''
        This method performs a rollout on the current game state. It will select from the child nodes 
        a node with the highest UCB1 value, breaking ties randomly. It will then sample states according to some pre-specified policy
        until it reaches the end, at which point it will update the visit count with: +1, 0.5, 0 for a win, tie, or loss, respectively
        num_iterations = number of rollouts to perform
        '''

        for i in range(num_iterations):
            node = self.select_node()
            win_update = self.random_rollout_policy(node.board_fen)

            # update the local number of visits for visited and root node
            node.n += 1
            node.w += win_update
            self.cur_node.w += win_update
            self.cur_node.n += 1
        
        # return a node with the highest UCB1 value
        return self.get_mcts_policy()

def print_node(node, agent, board=False):
    if board:
        print()
        print('*****BOARD*****')
        print(chess.Board(node.board_fen))
        print('***************')
        print()

    movestr = str(node.move)
    print('visit_count: %i win_count: %i ucb1: %f move: %s' % (node.n, node.w, agent.compute_ucb1(node), movestr))

def print_all_sibs(node, agent, board=False):
    tmp = node
    while tmp is not None:
        print_node(tmp, agent, board=board)
        tmp = tmp.nextSibling


if __name__ == '__main__':
    x = MonteCarloAgent()
    t = x.cur_node
    x.generate_possible_children()

    print_node(x.select_node(), x)

    print(x.rollout())
    
    print_all_sibs(t.firstChild, x)    