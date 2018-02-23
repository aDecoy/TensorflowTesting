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
from AI_gym.tren_modell1 import my_model

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


saveFileTempFolder=''
for filename in glob.iglob('/AppData/Local/Temp/tensorflow/AI_gym/cartpole0_3/*/', recursive=True):
    saveFileTempFolder=(filename.split('\\')[1])

parser.add_argument('--saved_model_dir', type=str,
    default=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                     'tensorflow/AI_gym/cartpole0_3/'+saveFileTempFolder),
    help='Directory to put the trained model.'
)




def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.contrib.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()




def main(argv):
    args = parser.parse_args(argv[1:])
    print("Running model in :{}".format(args.saved_model_dir))


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    tf.logging.set_verbosity(tf.logging.ERROR)

    # load the saved model
    my_feature_columns = [
        tf.feature_column.numeric_column(key='x'),
        tf.feature_column.numeric_column(key='x_dot'),
        tf.feature_column.numeric_column(key='theta'),
        tf.feature_column.numeric_column(key='theta_dot')
    ]
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
        'feature_columns':my_feature_columns,
        'hidden_units':[10, 10],
        'n_classes':2},
        model_dir=args.saved_model_dir )

    classifier_predictor=from_saved_model(args.saved_model_dir)

    # print(output_dict)
    # La modell spille
    modell_games = 5
    for _ in range(modell_games):
        env.reset()
        # env.render()
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        action = random.randrange(0, 2)
        observation, reward, done, info = env.step(action)
        #observation to dict
        observation={'x':[observation[0]],'x_dot':[observation[1]],'theta':[observation[2]],'theta_dot':[observation[3]]}
        for _ in range(goal_steps):
            env.render()
            # choose  action (0 or 1)
            model_input = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'x': tf.train.Feature(
                            float_list=tf.train.FloatList(value=observation['x'])
                        ),
                        'x_dot': tf.train.Feature(
                            float_list=tf.train.FloatList(value=observation['x_dot'])
                        ), 'theta': tf.train.Feature(
                            float_list=tf.train.FloatList(value=observation['theta'])
                        ), 'theta_dot': tf.train.Feature(
                            float_list=tf.train.FloatList(value=observation['theta_dot'])
                        )
                    }
                )
            )
            model_input_bytes = model_input.SerializeToString()
            prediction = classifier_predictor({'inputs': [model_input_bytes]})

            # do it!
            action=np.argmax(prediction['scores'], axis=-1)[0]
            observation, reward, done, info = env.step(action)
            observation = {'x': [observation[0]],
                           'x_dot': [observation[1]],
                           'theta': [observation[2]],
                           'theta_dot': [observation[3]]}
            score+=reward
            # print('score:{}  action:{}  observation:{}'.format(score, action, observation))

            if done:
                break
        print(score)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
