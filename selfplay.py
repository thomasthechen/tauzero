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
    NUM_GAMES = 2
    NUM_SEARCHES = 100

    agent = MonteCarloAgent(board_fen=chess.STARTING_FEN)
    move_probs = lambda f, n=agent.policy_net, b=bitboard: n(b(f).float())

    optimizer = optim.Adam(agent.policy_net.parameters(), lr=0.0001, amsgrad=True)
    floss = nn.MSELoss()
    dkl = nn.BCELoss()
 
    training_examples = []

    for i in range(NUM_GAMES):
        fen = chess.STARTING_FEN
        board = chess.Board(fen)
        if i != 0:
            agent.reset_board()
        num_moves = 0
        while not board.is_game_over():
            # display board
            print(board)
            cur_val, pol = move_probs(board.fen())
            # agent.policy_net.eval()
            print("White's" if board.turn else "Black's",'Turn. VALUE:', cur_val.item(), flush=True)

            aimove, val, improved_policy = agent.select_move(NUM_SEARCHES)
    
            if board.turn:
                print('WHITE CHOOSES', aimove.a, '\n')
            else:
                print('BLACK CHOOSES', aimove.a, '\n')
            
            assert board.fen() == aimove.s
            board.push(chess.Move.from_uci(aimove.a))
            num_moves += 1
            training_examples.append(TrainingExample(improved_policy, val, aimove.s))
            
            # if num_moves > 2:
                # break
        print(f'Game over. {"Black" if board.turn else "White"} wins.')
        
        # pick random batch, use to train
        batch_size = 2
        L = len(training_examples)

        idxs = np.random.choice(L, batch_size, replace=False)
        batch_sample = [training_examples[i] for i in idxs]

        torch.set_grad_enabled(True)
        agent.policy_net.train()
        for example in batch_sample:
            policy = example.policy
            refined_val = torch.tensor([example.value], requires_grad=True)
            state = example.board_fen

            net_val, logits = move_probs(state)
            probs, moves = mask_invalid(chess.Board(state), logits)
            policy_move_names = [x[0][1] for x in policy]

            assert len(policy) == len(probs)
            net_probs = torch.tensor([x for _,x in sorted(zip(moves,probs))], requires_grad=True)
            refined_probs = torch.tensor([x for _,x in sorted(zip(policy_move_names,[x[1] for x in policy]))], requires_grad=True)

            loss = floss(net_val, refined_val.detach()) + dkl(net_probs, refined_probs.detach()) # + nn.CrossEntropyLoss(probs, refined_probs)
            print(loss)
            loss.backward()
            optimizer.step()

        torch.save(agent.policy_net.state_dict(), "./trained_models/mc_net_realtime.pth")
            

# run the main function
if __name__ == '__main__':
    main()
