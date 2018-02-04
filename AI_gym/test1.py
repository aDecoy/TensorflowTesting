from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
import pandas as pd
import os
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--saved_model_dir', type=str,
    default=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                         'tensorflow/AI_gym/cartpole0'),
    help='Directory to put the trained model.'
)

def input_fn_train_generic(train_observation,train_action,batch_size):
    #train_observation=dict(train_observation)
    """An input function for training"""
    # Convert the inputs to a Dataset.
    print(train_action.shape)
    class_labels = np.argmax(train_action, axis=-1)
    print(class_labels.shape)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((train_observation, class_labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
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


env = gym.make('CartPole-v0')
env.reset()


goal_steps = 500
score_requirement = 50
initial_games = 10000
batch_size=100
training_steps=1000

def some_random_games_first():
    # Each of these is its own game.
    for episode in range(10):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break


#some_random_games_first()


def initial_population():
    env.reset()
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0, 2)
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                # saving our training data
                training_data.append([data[0], output])
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))
    training_data=np.array(training_data)
    return training_data

def main(argv):
    args = parser.parse_args(argv[1:])

    # lagrinsplass
    if tf.gfile.Exists(args.saved_model_dir):
        tf.gfile.DeleteRecursively(args.saved_model_dir)
    tf.gfile.MakeDirs(args.saved_model_dir)

    #trainingdata is an array with [observation seen,action should be taken]
    training_data = initial_population()
    #make it go up in batch size
    # training_data=training_data[:len(training_data)-len(training_data)%batch_size,:] #make it go up in batch size
    # print("")
    # print(training_data)

    my_feature_columns = [
        tf.feature_column.numeric_column(key='x'),
        tf.feature_column.numeric_column(key='x_dot'),
        tf.feature_column.numeric_column(key='theta'),
        tf.feature_column.numeric_column(key='theta_dot')
    ]
    #tesorflow estimator is used to make a model that is trained with "observation" as x and "action" as label
    #Lag modell

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2,
        model_dir=args.saved_model_dir
        )

    #Treningsdata

    train_observation= training_data[:,0] #creates a numpy list with np arrays containing observations
    #make a 2d numpy
    train_observation= np.array(train_observation.tolist())
    #gir train_observasjoner samme nøkkler som my_feature_columns
    keys=['x','x_dot','theta','theta_dot']
    train_observation = {'x': train_observation[:,0],'x_dot': train_observation[:,1],'theta':train_observation[:,2],'theta_dot':train_observation[:,3]}

    train_action= training_data[:,1]
    train_action= np.array(train_action.tolist())
    #estimator cnn classifier expects one class int, not a one hot vector
    #Tren modellen
    classifier.train(
        input_fn=lambda: input_fn_train_generic(train_observation, train_action,
                                                            args.batch_size),
        # input_fn=lambda:tf.estimator.inputs.numpy_input_fn(train_observation,train_action,
        #                                  args.batch_size,shuffle=True),
        steps=args.train_steps,

    )
    feature_spec  = {"observations": tf.FixedLenFeature([1],tf.float32)}
    feature_spec = {'x': tf.FixedLenFeature([1],tf.float32),'x_dot': tf.FixedLenFeature([1],tf.float32),'theta':tf.FixedLenFeature([1],tf.float32),'theta_dot':tf.FixedLenFeature([1],tf.float32)}
    serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    classifier.export_savedmodel(args.saved_model_dir, serving_input_receiver_fn=serving_fn)

    # La modell spille
    modell_games = 5
    for _ in range(modell_games):
        env.reset()
        env.render()
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        action = random.randrange(0, 2)
        observation, reward, done, info = env.step(action)
        # observation to dict
        observation = {'x': observation[0], 'x_dot': observation[1], 'theta': observation[2],
                       'theta_dot': observation[3]}
        for _ in range(goal_steps):
            env.render()
            # choose  action (0 or 1)
            print(observation)
            action = classifier.predict(
                input_fn=lambda: eval_input_fn(observation,
                                               labels=None,
                                               batch_size=args.batch_size))
            # do it!


            observation, reward, done, info = env.step(action)
            observation = {'x': observation[0],
                           'x_dot': observation[1],
                           'theta': observation[2],
                           'theta_dot': observation[3]}

            if done: break
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
