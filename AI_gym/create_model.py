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
                         'tensorflow/AI_gym/cartpole0_3/'),
    help='Directory to put the trained model.'
)



def my_model(features, labels, mode, params,reinforcement_loss=None):
    """DNN with three hidden layers. This decides what do happen in case you call Estimator.train(_),eval(_) or predict(_)"""
    reinforcement_loss = None
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        export_outputs2=tf.estimator.export.ExportOutput()
        export_outputs2=tf.estimator.export.ClassificationOutput(predictions['probabilities'])
        export_outputs={'name2':export_outputs2}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions,export_outputs=export_outputs )

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    #if training with supervised learning. (labels used instead of loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.5)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    # train_op = optimizer.minimize(loss, global_step=None)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def input_fn_train_generic(train_observation,train_action,batch_size):
    '''
    An input function for training, it converts the inputs to a Dataset
    :param train_observation:  Ovservation dict
    :param train_action:        Labels
    :param batch_size:          Size of training batches
    :return:
    '''

    # print(train_observation)
    # print(train_action)
    class_labels = np.argmax(train_action, axis=-1)
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
env._max_episode_steps=goal_steps

score_requirement = 90
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

def initial_random_population():
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
    training_data = initial_random_population()
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

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
        'feature_columns':my_feature_columns,
        'hidden_units':[10, 10],
        'n_classes':2},
        model_dir=args.saved_model_dir,
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

    feature_spec = {'x': tf.FixedLenFeature([1],tf.float32),'x_dot': tf.FixedLenFeature([1],tf.float32),'theta':tf.FixedLenFeature([1],tf.float32),'theta_dot':tf.FixedLenFeature([1],tf.float32)}
    serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    classifier.export_savedmodel(args.saved_model_dir, serving_input_receiver_fn=serving_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
