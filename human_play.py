# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
#from policy_value_net_numpy import PolicyValueNetNumpy
from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if location == 3333 :
                return -2
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def get_five_action(self,board):
        
        try:
            move_5_1 = input("Your move in 5th: ")
            
        except Exception as e:
            move_5_1 = [-1]
        if move_5_1[0] == -1 or ((move_5_1[0] not in board.availables) or (move_5_1[1] not in board.availables)):
            print("invalid move in 5 th")
            move_5_1 = self.get_five_action(board)
        return move_5_1

    def move_five_action(self,board,move_5_1):
        move_5_1_0 = board.move_to_location(move_5_1[0])
        move_5_1_1 = board.move_to_location(move_5_1[1])
        print("please delete from:",move_5_1_0,move_5_1_1,"as",move_5_1[0],"as",move_5_1[1])
        try:
            move = input("choose which to delete:")
            print("move",move)
        except Exception as e:
            move = -1
        if (move == -1) or ((move != move_5_1[0]) and (move != move_5_1[1])):
            print("invalid move, please choose from ")
            move = self.move_five_action(board,move_5_1)
        return move


    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = pickle.load(open("{}".format(model_file), 'rb'))
        except:
            policy_param = pickle.load(open("{}".format(model_file), 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNet(width, height, model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player,best_policy, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
