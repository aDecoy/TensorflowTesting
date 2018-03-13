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
# from tensorflow.contrib.predictor.core_estimator_predictor import CoreEstimatorPredictor
from tensorflow.contrib.predictor.saved_model_predictor import  SavedModelPredictor
from tensorflow.contrib.predictor.predictor_factories import from_saved_model
import glob
from tensorflow.python.ops import math_ops
from AI_gym.create_model import my_model
#import setuptools
import gym
import mujoco_py
from os.path import dirname
print('lalla')
model = mujoco_py.load_model_from_path(dirname(dirname(mujoco_py.__file__))  +"/xmls/claw.xml")
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

sim.step()
print(sim.data.qpos)



env = gym.make('CartPole-v0')
env.reset()




goal_steps = 500
env._max_episode_steps=goal_steps

score_requirement = 50
initial_games = 10000

tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

env = gym.make('CartPole-v0')
# env = gym.make('BipedalWalker-v2')
env = gym.make('Pendulum-v0')
env = gym.make('Acrobot-v1')
# env = gym.make('Ant-v2')
# env = gym.make('bipedalwalker')


def main(argv):
    env.reset()
    for i in range(goal_steps):
        observation, reward, done, debug= env.step(1)
        if done:
            break
        env.render()
        print(observation)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

