import chess
import random
import torch 
import argparse
import numpy as np

from minimax_agent import MiniMaxAgent
from value_approximator import Net
from monte_carlo_agent import MonteCarloAgent
from state import State


class TrainingExample(object):
    def __init__(self, policy, value, board_fen=None):
        self.board_fen = board_fen
        '''
        NOTE: the policy is supposed to encompass all possible 
        moves in the Go implementation. here we only have legal moves.
        Policy is stored as [(move, prob) ..]
        '''
        self.policy = policy 
        self.value = value
    

def main():
    NUM_GAMES = 10
    # append with TrainingExamples
    # TODO setup training framework
    training_examples = []

    for i in range(NUM_GAMES):
        fen = chess.STARTING_FEN
        board = chess.Board(fen)

        while not board.is_game_over():
            print('\n')
            if board.turn:
                print('White\'s Turn')
            else:
                print('Black\'s Turn')
            print('\n')
            
            # display board
            print(board)
            print('\n')
            '''
            # display possible moves
            print('Possible moves: ', end = '')
            for move in board.legal_moves:
                print(move.uci() + ' ', end  = '')
            print('\n')
            '''
            if board.turn:
                aimove = None
                agent = MonteCarloAgent(board_fen=board.fen(), black=False)
                agent.generate_possible_children()
                move_probs = agent.rollout_get_policy()

                moves = [x[1] for x in move_probs]
                probs = [x[0] for x in move_probs]
                probs = probs/np.sum(probs)

                value = agent.cur_node.w/agent.cur_node.n
                aimove = np.random.choice(moves, p=probs)

                print('\nWHITE CHOOSES', aimove)
                board.push(aimove)

                training_examples.append(TrainingExample(move_probs, value, board.fen()))
            else:
                aimove = None
                agent = MonteCarloAgent(board_fen=board.fen(), black=True)
                agent.generate_possible_children()
                move_probs = agent.rollout_get_policy()

                moves = [x[1] for x in move_probs]
                probs = [x[0] for x in move_probs]
                probs = probs/np.sum(probs)

                value = agent.cur_node.w/agent.cur_node.n

                aimove = np.random.choice(moves, p=probs) 
                
                print('\nBLACK CHOOSES', aimove)
                board.push(aimove)

                training_examples.append(TrainingExample(move_probs, value, board.fen()))
                

        print(f'Game over. {"Black" if board.turn else "White"} wins.')
        '''
        TODO: implement training framework here
        '''

# run the main function
if __name__ == '__main__':
    main()