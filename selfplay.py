import chess
import random
import torch 
import argparse
import numpy as np

# from minimax_agent import MiniMaxAgent
# from value_approximator import Net
from monte_carlo_agent import MonteCarloAgent
# from state import State


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
    NUM_GAMES = 1
    # append with TrainingExamples
    # TODO setup training framework

    agent = MonteCarloAgent(board_fen=chess.STARTING_FEN)

    training_examples = []

    for i in range(NUM_GAMES):
        fen = chess.STARTING_FEN
        # agent.reset_board_and_tree(fen)
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

            if board.turn:
               
                aimove, val, improved_policy = agent.select_move()
                print('\nWHITE CHOOSES', aimove)

                assert board.fen() == aimove.s
                board.push(chess.Move.from_uci(aimove.a))

                training_examples.append(TrainingExample(improved_policy, val, aimove.s))
            else:
                aimove, val, improved_policy = agent.select_move()
                print('\nBLACK CHOOSES', aimove)

                assert board.fen() == aimove.s
                board.push(chess.Move.from_uci(aimove.a))

                training_examples.append(TrainingExample(improved_policy, val, aimove.s))

        print(f'Game over. {"Black" if board.turn else "White"} wins.')
        
        '''
        TODO: implement training framework HERE using training_examples; train on value and on improved policy over legal moves
        '''

# run the main function
if __name__ == '__main__':
    main()
