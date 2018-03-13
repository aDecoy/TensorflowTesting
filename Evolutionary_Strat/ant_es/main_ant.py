from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import random
import numpy as np
import argparse
import os
import glob
import gym
import mujoco_py
from Evolutionary_Strat.ant_es.Ant_Agent import *
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


score_requirement = 50
initial_games = 10000

goal_steps = 500

def main(argv):

    agent= Ant_Agent()
    agent.load('wights2.pkl')
    agent.play(5)
    # for i in range(10):
    #     print('Age {}'.format(i))
    #     agent.train(50)
    #     agent.save('wights2.pkl')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

