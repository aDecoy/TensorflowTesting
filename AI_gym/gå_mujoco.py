from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter

import argparse
import tensorflow as tf
import os
import glob
import gym
import mujoco_py
from os.path import dirname



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


env = gym.make('Ant-v2')
env.reset()
# env = gym.make('bipedalwalker')
goal_steps = 500
env._max_episode_steps=goal_steps

def main(argv):
    env.reset()
    env.render()
    for i in range(goal_steps):
        observation, reward, done, debug= env.step([0,0,0,100,0,0,0,0])
        if done:
            break
        env.render()
        print(observation)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

