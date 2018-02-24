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
from tensorflow.contrib.predictor.predictor_factories import from_saved_model
import glob
from AI_gym.create_model import my_model,input_fn_train_generic,initial_random_population

env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500  #how long to run shall last before you get max score. (Evry step is one score)
env._max_episode_steps=goal_steps

new_game_runs=100                 #how many times to play the game with the neural network. Best resutls are used as new trainingdata
score_requirement = 130         #Only best runs are kept. this is the requierment to add the run's [observation, action] to trainingdata
number_of_training_loops=10      #Number of times to play game with model, get trainingdata and then train model on the resutling trainingdata
# initial_games = 10000
noise_value=20 #less is more noise     Adds randomness. Help to generalize and make sure evry run is not the same.
# tf.reset_default_graph()    #maybe unessesary

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
saveFileTempFolder=''
for filename in glob.iglob('/AppData/Local/Temp/tensorflow/AI_gym/cartpole0_3/*/', recursive=True):
    saveFileTempFolder=(filename.split('\\')[1])   #tensorflow saves model in a subfolder with random name (or maybe timestamp idk). To load we need to know what temp folder to use. That is why we do this

parser.add_argument('--saved_estimator_model_dir', type=str,
    default=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                     'tensorflow/AI_gym/cartpole0_3/'+saveFileTempFolder),
    help='Directory to put the trained model.'
)
parser.add_argument('--predictor_save_model_dir', type=str,
    default=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                         'tensorflow/AI_gym/cartpole0_predictor/'),
    help='Directory to export the Estimator model to after training.'
)
parser.add_argument('--predictor_load_model_dir', type=str,
    default=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                         'tensorflow/AI_gym/cartpole0_predictor/'),
    help='Directory to load an Estimator model to be used by a Predictor class.'
)
#load location will be modified because every new export operation gives a new folder name




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


def get_new_training_data(classifier_predictor, score_requirement, new_game_runs, goal_steps, top_training_data):
    '''
    Play games using classifier_predictor to chose input action. Best play sessions are used to make the new training data.
    :param classifier_predictor:  Predictor class used to make rapid prediction of best action based on game state
    :param score_requirement:       How good score we require a game session/run to have before we keep its observarton to action mapping.
    :param new_game_runs:           How many games to run
    :param goal_steps:              How long the simulation shoud run before we force an end
    :param top_training_data:       List of the top 90% runs
    :return  training_data, next_score_requierment:                        TrainingData is numpy array with eac runh row holding [observation, appropiate action]. (appropiateAction is a one hot array)  , next_score_requirement is how strict we should be next training
    '''
    next_score_requierment= score_requirement
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []


    # La modell spille  / Let the model/Predictor play
    for _ in range(new_game_runs):
        env.reset()
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        prev_observation_list= []

        #first move is random
        action = random.randrange(0, 2)
        observation, reward, done, info = env.step(action)
        # observation to dict. x is cart. theta is pole. _dot is momentum. Not my names, the AI gym use these names
        observation = {'x': [observation[0]], 'x_dot': [observation[1]], 'theta': [observation[2]],
                       'theta_dot': [observation[3]]}
        #Play one game
        for _ in range(goal_steps):
            # env.render()
            #Noise intorduce new move combinations and helps the network to generalize
            if random.randrange(0,noise_value) ==1 :
                # Randomly choose action (0 or 1)
                action=random.randrange(0,2)
            else:
                #predictor makes choise
                #The Predictor wants a very spesific format on its input. You need to change the observation into this
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
                #Now you can use this input to make prediction
                prediction = classifier_predictor({'inputs': [model_input_bytes]})
                action = np.argmax(prediction['scores'], axis=-1)[0]  # One hot array to a single int. (0 or 1)
            # Do it!
            observation_list, reward, done, info = env.step(action)
            #"It is not the jedi way"
            observation = {'x': [observation_list[0]],
                           'x_dot': [observation_list[1]],
                           'theta': [observation_list[2]],
                           'theta_dot': [observation_list[3]]}
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])   #used to map seen observation to what action should be taken
            prev_observation = observation_list
            score += reward
            # print('score:{}  action:{}  observation:{}'.format(score, action, observation))
            if done:
                break

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
                #if top 90%, Join the best training data set
                if score> goal_steps*0.90:
                    top_training_data.append([data[0],output])


        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # in case you wanted to reference later:
    # training_data_save = np.array(training_data)
    # np.save('saved.npy', training_data_save)

    #debugging stuff:
    # print(accepted_scores)
    # some stats here, to further illustrate the neural network magic!
    accepted_scores=sorted(accepted_scores)
    print('Average score: {}'.format(mean(scores)))
    # print('Average accepted score:', mean(accepted_scores))
    # print('Median score for accepted scores:', median(accepted_scores))
    print('Number of each scores {}'.format(Counter(accepted_scores)))
    print('Number of of accepted scores {}'.format(len(accepted_scores)))
    # print('accepted scores: {}'.format(accepted_scores))

    if accepted_scores:
        next_score_requierment=max(accepted_scores[int(len(accepted_scores)*0.8)],mean(scores)) # only accept better than the 8/10 of accepted_scores, on the next training round. Or average score if that is higher
    next_score_requierment=min(int(goal_steps*0.9),next_score_requierment)  #strictest we are to trainingdata is 90% of max possible steps
    training_data= training_data+top_training_data                          #No reason to not make use of the best data so far
    training_data = np.array(training_data)  # Estimator.train function demand numpy array

    #sometimes you get stuck in local optimum or you play a lot of games to only have less than 5 of them actually be helpfull. In that case try to start anew with random data
    if len(accepted_scores)<4 or random.randrange(1,noise_value*3)==1:
        #new random training data seem to be needed
        training_data=initial_random_population()
        next_score_requierment=score_requirement*0.5
        print('average score {}'.format(mean(scores)))
        print('random trainingdata used')

    return training_data, next_score_requierment


def train_mode_with_training_data(classifier,training_data,args):
    '''
    Transform trainingdata to correct format and use estimator.train function

    :param classifier: Estimator object to update/train
    :param training_data: list of [observation list,action] arrays
    :param args: just for file paths
    :return:
    '''
    train_observation = training_data[:, 0]  # creates a numpy list with np arrays containing observations
    train_observation = np.array(train_observation.tolist())
    # gir train_observasjoner samme nøkkler som my_feature_columns
    #train funciton want this format
    train_observation = {'x': train_observation[:, 0], 'x_dot': train_observation[:, 1],
                         'theta': train_observation[:, 2], 'theta_dot': train_observation[:, 3]}
    # keys = ['x', 'x_dot', 'theta', 'theta_dot']
    train_action = training_data[:, 1]
    train_action = np.array(train_action.tolist())
    # print(train_action)
    # Tren modellen / Train the Estimator
    classifier.train(
        input_fn=lambda: input_fn_train_generic(train_observation, train_action,args.batch_size),
        steps=args.train_steps,
    )

def export_estimator_to_file(classifier,args):
    '''Saves the estimator in the file system. Predictor only worked when it loaded from filesystem. Because of this we save and load every time we train. Slows down system a little, but beside running the games hundrets of time it is not that important'''

    #delete previus model in predictor space
    if tf.gfile.Exists(args.predictor_save_model_dir):
        tf.gfile.DeleteRecursively(args.predictor_save_model_dir)
    tf.gfile.MakeDirs(args.predictor_save_model_dir)

    #save it
    feature_spec = {'x': tf.FixedLenFeature([1], tf.float32), 'x_dot': tf.FixedLenFeature([1], tf.float32),
                    'theta': tf.FixedLenFeature([1], tf.float32), 'theta_dot': tf.FixedLenFeature([1], tf.float32)}
    serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    classifier.export_savedmodel(args.predictor_save_model_dir, serving_input_receiver_fn=serving_fn)

    #update where it was saved since it creates a new random foldername
    predictor_model_TempFolder_name = ''
    for filename in glob.iglob(args.predictor_save_model_dir+'/*/', recursive=True):
        predictor_model_TempFolder_name = (filename.split('\\')[1])
    args.predictor_load_model_dir=args.predictor_save_model_dir+predictor_model_TempFolder_name
    # print('now using predictor in file path: {}'.format(args.predictor_load_model_dir))


def main(argv):
    args = parser.parse_args(argv[1:])
    print("Running model to be trained in :{}".format(args.saved_estimator_model_dir))
    print("Running predictors in :{}".format(args.predictor_save_model_dir))

    #hides some warnings that spams the console
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    tf.logging.set_verbosity(tf.logging.ERROR)

    # lagrinsplass for predictor    / Prepare work folders for the Predictor
    if tf.gfile.Exists(args.predictor_save_model_dir):
        tf.gfile.DeleteRecursively(args.predictor_save_model_dir)
    tf.gfile.MakeDirs(args.predictor_save_model_dir)

    # load the saved model. Assumes that a model was allready made using create new model
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
        model_dir=args.saved_estimator_model_dir )

    print('now using predictor in file path: {}'.format(args.predictor_load_model_dir))
    # print('estimators dense2 kernel: {}'.format(classifier.get_variable_value('dense_2/kernel')))  # watch changes in the weights. Only works after training have been done. therefore disabled


    new_score_requirement=score_requirement
    classifier_predictor=from_saved_model(args.saved_estimator_model_dir)

    best_training_data=[]
    training_data=[]
    for i in range(number_of_training_loops):
        print('Run number {}'.format(i))
        training_data, new_score_requirement=get_new_training_data(classifier_predictor,new_score_requirement,new_game_runs,goal_steps,best_training_data)
        # print('best training data len {}'.format(len(best_training_data)))
        train_mode_with_training_data(classifier,training_data,args)
        export_estimator_to_file(classifier,args)
        classifier_predictor=from_saved_model(args.predictor_load_model_dir)
        print('/////////////7////////////////7 \n score_requierment: {}'.format(new_score_requirement))
        # print('estimators dense2 kernel: {}'.format(classifier.get_variable_value('dense_2/kernel')))   #watch changes in the weights
        # classifier_predictor=from_saved_model(args.saved_estimator_model_dir)


    print('len best data {}'.format(len(best_training_data)))
    best_training_data=np.array(best_training_data)

    # train one last time on only the best training data
    train_mode_with_training_data(classifier, best_training_data, args)
    export_estimator_to_file(classifier, args)
    classifier_predictor = from_saved_model(args.predictor_load_model_dir)
    # classifier_predictor = from_saved_model(args.saved_estimator_model_dir)

    #this is just to see the results.
    best_training_data2=[]
    training_data, new_score_requirement = get_new_training_data(classifier_predictor, new_score_requirement,
                                                                 new_game_runs, goal_steps, best_training_data2)
    # print('estimators dense2 kernel: {}'.format(classifier.get_variable_value('dense_2/kernel')))  # watch changes in the weights

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
