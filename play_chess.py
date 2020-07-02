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
import psycopg2
import traceback
import base64
import numpy as np

from minimax_agent import MiniMaxAgent
from value_approximator import Net
# from monte_carlo_agent import MonteCarloAgent
from state import State

from rq import Queue
from worker import conn

from flask_sqlalchemy import SQLAlchemy

q = Queue(connection=conn)

# DATABASE_URL = os.environ.get('DATABASE_URL')
# conn2 = psycopg2.connect(DATABASE_URL, sslmode='require')


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

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request
app = Flask(__name__)


app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Entry(db.Model):
    __tablename__ = 'boards'

    id = db.Column(db.Integer, primary_key=True)
    board = db.Column(db.String())

    def __init__(self, board):
        self.board = board

    def __repr__(self):
        return '<id {}>'.format(self.id)
    
    def serialize(self):
        return {
            'id': self.id, 
            'board': self.board,
        }
ai = MiniMaxAgent()
# s = State()

@app.route("/")
def hello():
    ret = open("index.html").read()
    fen = Entry.query.first().board
    print('FENFEN', fen)
    return ret.replace('start', fen)


def computer_move():
    aimove = None
    fen = Entry.query.first().board
    s = State()
    s.board = chess.Board(fen)

    possible_moves = ai.minimax(s.board)
    probs = [x[1] for x in possible_moves]
    moves = [x[0] for x in possible_moves] 
    probs = probs/np.sum(probs)
    aimove = np.random.choice(moves, p=probs)
    s.board.push(aimove)

    bk = Entry.query.update(dict(board=s.board.fen()))
    db.session.commit()
    


# move given in algebraic notation
@app.route("/move")
def move():
    
    if not s.board.is_game_over():
        move = request.args.get('move',default="")
        if move is not None and move != "":
            print("human moves", move)
            try:
                s.board.push_san(move)

                q.enqueue(computer_move(s), 'http://heroku.com')
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

    fen = Entry.query.first().board
    s = State()
    s.board = chess.Board(fen)

    if not s.board.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = True if request.args.get('promotion', default='') == 'true' else False

        move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))
        # print(s.board)
        if move is not None and move != "":
            print("human moves", move)
            try:
                s.board.push_san(move)
                bk = Entry.query.update(dict(board=s.board.fen()))
                db.session.commit()
                computer_move()
                # computer_move()
            except Exception:
                traceback.print_exc()
        fen = Entry.query.first().board
        s.board = chess.Board(fen)  
        response = app.response_class(
        response=s.board.fen(),
        status=200
        )
        print(s.board)
        return response

    print("GAME IS OVER")
    response = app.response_class(
        response="game over",
        status=200
    )
    return response

@app.route("/newgame")
def newgame():

    fen = chess.Board.starting_fen
    bk = Entry.query.update(dict(board=fen))
    db.session.commit()
    # entry = Entry(s.board.fen())
    # db.session.add(entry)
    # db.session.commit()

    response = app.response_class(
        response=fen,
        status=200
    )
    return response


if __name__ == "__main__":
    '''
    meta = db.metadata
    for table in reversed(meta.sorted_tables):
        db.session.execute(table.delete())
    db.session.commit()
    
    s = State()
    entry = Entry(s.board.fen())
    db.session.add(entry)
    db.session.commit()
    '''
    '''
    bk = Entry.query.update(dict(board='ok'))
    db.session.commit()
    print('BOARD', Entry.query.all())
    
    print('BOARD', Entry.query.first().board)
    '''
    app.run(debug=True)

