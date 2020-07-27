import chess
import random
import torch 
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# from minimax_agent import MiniMaxAgent
# from value_approximator import Net
from monte_carlo_agent import MonteCarloAgent
from MoveNet import mask_invalid
from utils import bitboard
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
    move_probs = lambda f, n=agent.policy_net, b=bitboard: n(b(f).float())

    optimizer = optim.Adam(agent.policy_net.parameters(), lr=0.0001, amsgrad=True)
    floss = nn.MSELoss()
    dkl = nn.BCELoss()
 
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
               
                aimove, val, improved_policy = agent.select_move(10)
                print('\nWHITE CHOOSES', aimove)

                assert board.fen() == aimove.s
                board.push(chess.Move.from_uci(aimove.a))

                training_examples.append(TrainingExample(improved_policy, val, aimove.s))
            else:
                aimove, val, improved_policy = agent.select_move(10)
                print('\nBLACK CHOOSES', aimove)

                assert board.fen() == aimove.s
                board.push(chess.Move.from_uci(aimove.a))

                training_examples.append(TrainingExample(improved_policy, val, aimove.s))

        print(f'Game over. {"Black" if board.turn else "White"} wins.')
        
        '''
        TODO: implement training framework HERE using training_examples; train on value and on improved policy over legal moves
        '''
        # pick random batch, use to train
        batch_size = 2
        L = len(training_examples)

        idxs = np.random.choice(L, batch_size, replace=False)
        batch_sample = [training_examples[i] for i in idxs]

        for example in batch_sample:
            policy = example.policy
            val = torch.tensor([example.value], requires_grad=True)
            state = example.board_fen

            net_val, logits = move_probs(state)
            probs, moves = mask_invalid(chess.Board(state), logits)
            # net_val.requires_grad=True
            # probs.requires_grad = True

            assert len(policy) == len(probs)
            refined_probs = torch.tensor([x[1] for x in policy], requires_grad=True)
            print(refined_probs, probs)
            print('\n')
            print(net_val, val)
            '''
            loss = floss(net_val, val.detach()) + dkl(probs, refined_probs.detach()) # + nn.CrossEntropyLoss(probs, refined_probs)
            print(loss)
            loss.backward()
            optimizer.step()
            '''







# run the main function
if __name__ == '__main__':
    main()
