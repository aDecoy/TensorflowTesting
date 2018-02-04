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
env = gym.make('CartPole-v0')
env.reset()


goal_steps = 500
score_requirement = 50
initial_games = 10000

tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--saved_model_dir', type=str,
    default=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                         'tensorflow/AI_gym/cartpole0'),
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

    #load model
    my_feature_columns = [
        tf.feature_column.numeric_column(key='x'),
        tf.feature_column.numeric_column(key='x_dot'),
        tf.feature_column.numeric_column(key='theta'),
        tf.feature_column.numeric_column(key='theta_dot')
    ]
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2,
        model_dir=args.saved_model_dir
        )

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
            # env.render()
            # choose  action (0 or 1)
            prediction = classifier.predict(
                input_fn=lambda: eval_input_fn(observation,
                                               labels=None,
                                               batch_size=1))
            # do it!
            expected = ['Setosa', 'Versicolor', 'Virginica']
            for pred_dict, expec in zip(prediction, expected):
                action=np.argmax(pred_dict['probabilities'], axis=-1)
                # print(action)

            observation, reward, done, info = env.step(action)
            observation = {'x': [observation[0]],
                           'x_dot': [observation[1]],
                           'theta': [observation[2]],
                           'theta_dot': [observation[3]]}

            if done:
                print(reward)
                break

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
