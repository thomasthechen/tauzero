'''
This is a file used to test the python-chess library and its functionality. 

Abbreviations:
SAN - standard algebraic notation (Nf3)
UCI - universal chess interface (g1f3)
FEN - Forsyth-Edwards notation (for board state)

board.turn returns True for white and False for black

By default, moves are notated with UCI. 
'''

import chess
import random
import torch 
import argparse

from minimax_agent import MiniMaxAgent
from value_approximator import Net
from monte_carlo_agent import MonteCarloAgent
from state import State

def parse_arguments():
    parser = argparse.ArgumentParser(description='Provide arguments for which agent you want to play')
    parser.add_argument('--agent', choices=['minimax', 'mcts'], required=True)
    parser.add_argument('--mcts_trials', type=int, default=300)
    return parser.parse_args()

def main():
    args = parse_arguments()

    # STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    fen = chess.STARTING_FEN
    board = chess.Board(fen)

    if args.agent == 'minimax':
        value_approx = Net()
        value_approx.load_state_dict(torch.load('./trained_models/value.pth', map_location=torch.device('cpu')))
        value_approx.eval()
        
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in value_approx.state_dict():
            print(param_tensor, "\t", value_approx.state_dict()[param_tensor].size())
        
        ai = MiniMaxAgent()
        ai.evaluate_board(board)
    
    while not board.is_game_over():
        # display whose turn it is
        print('\n')
        if board.turn:
            print('White\'s Turn')
        else:
            print('Black\'s Turn')
        print('\n')
        
        # display board
        print(board)
        print('\n')
        
        # display possible moves
        print('Possible moves: ', end = '')
        for move in board.legal_moves:
            print(move.uci() + ' ', end  = '')
        print('\n')

        if args.agent == 'minimax':
            # ai.evaluate_board(board)
            # ai.minimax(board)
            in_tensor = torch.tensor(State(board).serialize()).float()
            in_tensor = in_tensor.reshape(1, 13, 8, 8)
            print('AI EVAL:', value_approx(in_tensor))
        # read move if human playerd
        if board.turn:
            input_uci = input('What move would you like to play?\n')
            playermove = chess.Move.from_uci(input_uci)
            if playermove in board.legal_moves:
                board.push(playermove)

        # generate move for ai
        else:
            # add in minimax decision point
            # give minimax an array of legal moves and the current board state
            aimove = None
            if args.agent == 'minimax':
                possible_moves = ai.minimax(board)
                print('\nBEST AI MOVES', possible_moves)
                aimove = random.choice(possible_moves)[0]
            
            else:
                agent = MonteCarloAgent(board_fen=board.fen(), black=True)
                agent.generate_possible_children()
                aimove = agent.rollout(num_iterations=args.mcts_trials)
    
            print('\nAI CHOOSES', aimove)
            board.push(aimove)

    print(f'Game over. {"Black" if board.turn else "White"} wins.')

# run the main function
if __name__ == '__main__':
    main()