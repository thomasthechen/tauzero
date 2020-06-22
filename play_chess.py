'''
This is a file used to test the python-chess library and its functionality. 

Abbreviations:
SAN - standard algebraic notation (Nf3)
UCI - universal chess interface (g1f3)
FEN - Forsyth-Edwards notation (for board state)

board.turn returns True for white and False for black

By default, moves are notated with UCI. 
'''

'''
pruning by heuristic is way faster but leads to dumb king moves; pruning by actual eval is hella slow
'''

import chess
import random
import torch 
import argparse
import os
import traceback
import base64

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
        ''' 
        # display possible moves
        print('Possible moves: ', end = '')
        for move in board.legal_moves:
            print(move.uci() + ' ', end  = '')
        print('\n')
        '''
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
                # print('\nBEST AI MOVES', possible_moves)
                aimove = random.choice(possible_moves)[0]
            
            else:
                agent = MonteCarloAgent(board_fen=board.fen(), black=True)
                agent.generate_possible_children()
                aimove = agent.rollout(num_iterations=args.mcts_trials)
    
            print('\nAI CHOOSES', aimove)
            board.push(aimove)

    print(f'Game over. {"Black" if board.turn else "White"} wins.')



# @author George Hotz
s = State()

value_approx = Net()
value_approx.load_state_dict(torch.load('./trained_models/value_40_6000_4.pth', map_location=torch.device('cpu')))
value_approx.eval()
ai = MiniMaxAgent()


def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request
app = Flask(__name__)

@app.route("/")
def hello():
    ret = open("index.html").read()
    return ret.replace('start', s.board.fen())


def computer_move(s):
    aimove = None
    # if args.agent == 'minimax':
    possible_moves = ai.minimax(s.board)
    # print('\nBEST AI MOVES', possible_moves)
    aimove = random.choice(possible_moves)[0]
    s.board.push(aimove)

@app.route("/selfplay")
def selfplay():
    s = State()

    ret = '<html><head>'
    # self play
    while not s.board.is_game_over():
        computer_move(s)
        ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(s)
        print(s.board.result())

    return ret


# move given in algebraic notation
@app.route("/move")
def move():
    if not s.board.is_game_over():
        move = request.args.get('move',default="")
        if move is not None and move != "":
            print("human moves", move)
            try:
                s.board.push_san(move)
                computer_move(s)
            except Exception:
                traceback.print_exc()
            response = app.response_class(
                response=s.board.fen(),
                status=200
            )
            return response
    else:
        print("GAME IS OVER")
        response = app.response_class(
        response="game over",
        status=200
        )
        return response
    print("hello ran")
    return hello()

# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
    if not s.board.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = True if request.args.get('promotion', default='') == 'true' else False

        move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

        if move is not None and move != "":
            print("human moves", move)
            try:
                s.board.push_san(move)
                computer_move(s)
            except Exception:
                traceback.print_exc()
        response = app.response_class(
        response=s.board.fen(),
        status=200
        )
        return response

    print("GAME IS OVER")
    response = app.response_class(
        response="game over",
        status=200
    )
    return response

@app.route("/newgame")
def newgame():
    s.board.reset()
    response = app.response_class(
        response=s.board.fen(),
        status=200
    )
    return response


if __name__ == "__main__":
    if os.getenv("SELFPLAY") is not None:
        s = State()
        while not s.board.is_game_over():
            computer_move(s)
            print(s.board)
            in_tensor = torch.tensor(State(s.board).serialize()).float()
            in_tensor = in_tensor.reshape(1, 13, 8, 8)
            print('\nAI EVAL: ', value_approx(in_tensor).item())
            print(s.board.result())
    else:
        app.run(debug=True)



'''
# run the main function
if __name__ == '__main__':
    main()
'''