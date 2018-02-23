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
from AI_gym.tren_modell1 import my_model,input_fn_train_generic,initial_random_population

env = gym.make('CartPole-v0')

env.reset()
goal_steps = 500
env._max_episode_steps=goal_steps
new_game_runs=150

score_requirement = 130
number_of_training_loops=3
initial_games = 10000
noise_value=20 #less is more noise
tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


saveFileTempFolder=''
for filename in glob.iglob('/AppData/Local/Temp/tensorflow/AI_gym/cartpole0_3/*/', recursive=True):
    saveFileTempFolder=(filename.split('\\')[1])

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
    # print(output_dict)
    next_score_requierment= score_requirement
    # La modell spille
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []


    for _ in range(new_game_runs):
        env.reset()
        # env.render()
        score = 0
        # moves specifically from this environment:
        game_memory = []
        prev_observation = []
        prev_observation_list= []
        # previous observation that we saw
        action = random.randrange(0, 2)
        observation, reward, done, info = env.step(action)

        # observation to dict
        observation = {'x': [observation[0]], 'x_dot': [observation[1]], 'theta': [observation[2]],
                       'theta_dot': [observation[3]]}
        for _ in range(goal_steps):
            # env.render()
            # choose  action (0 or 1)
            #Noise intorduce new move combinations and helps network to generalize
            if random.randrange(0,noise_value) ==1 :
                action=random.randrange(0,2)
            else:
                #predictor makes choise
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
                action = np.argmax(prediction['scores'], axis=-1)[0]
            # do it!
            observation_list, reward, done, info = env.step(action)
            observation = {'x': [observation_list[0]],
                           'x_dot': [observation_list[1]],
                           'theta': [observation_list[2]],
                           'theta_dot': [observation_list[3]]}
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation_list
            score += reward
            # print('score:{}  action:{}  observation:{}'.format(score, action, observation))
            if done:
                # print(score)
                break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement or score> goal_steps*0.9:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                # saving our training data
                training_data.append([data[0], output])
                if score> goal_steps*0.90:
                    top_training_data.append([data[0],output])


        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # print(accepted_scores)
    accepted_scores=sorted(accepted_scores)



    print('average score {}'.format(mean(scores)))
    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    # some stats here, to further illustrate the neural network magic!
    # print('Average accepted score:', mean(accepted_scores))
    # print('Median score for accepted scores:', median(accepted_scores))
    print('Number of each scores {}'.format(Counter(accepted_scores)))
    print('Number of of accepted scores {}'.format(len(accepted_scores)))
    # print('accepted scores: {}'.format(accepted_scores))

    # next_score_requierment=max(scores[int(len(scores)*0.8)],mean(scores)) # only accept better than the 9/10 of accepted_scores, on the next training round
    if accepted_scores:
        next_score_requierment=max(accepted_scores[int(len(accepted_scores)*0.8)],mean(scores)) # only accept better than the 9/10 of accepted_scores, on the next training round
    next_score_requierment=min(int(goal_steps*0.9),next_score_requierment)  #strictest we are to trainingdata is 90% of max possible steps
    training_data= training_data+top_training_data
    training_data = np.array(training_data)


    if len(accepted_scores)<4 or random.randrange(1,noise_value*3)==1:
        #new random training data seem to be needed
        training_data=initial_random_population()
        next_score_requierment=score_requirement*0.5
        print('average score {}'.format(mean(scores)))
        print('random trainingdata used')

    return training_data, next_score_requierment


def train_mode_with_training_data(classifier,training_data,args):
    '''
    :param classifier: classifier to update/train
    :param training_data: list of [observation list,action]
    :param args: just for file path
    :return:
    '''
    train_observation = training_data[:, 0]  # creates a numpy list with np arrays containing observations
    train_observation = np.array(train_observation.tolist())
    # gir train_observasjoner samme nøkkler som my_feature_columns
    train_observation = {'x': train_observation[:, 0], 'x_dot': train_observation[:, 1],
                         'theta': train_observation[:, 2], 'theta_dot': train_observation[:, 3]}
    # train_observation_2={}
    # train_observation_2['x']= [observation['x'] for observation in train_observation][0]
    # print('train_observation_2 {}'.format(train_observation_2))
    # keys = ['x', 'x_dot', 'theta', 'theta_dot']
    train_action = training_data[:, 1]
    train_action = np.array(train_action.tolist())
    # print(train_action)
    # Tren modellen
    classifier.train(
        input_fn=lambda: input_fn_train_generic(train_observation, train_action,
                                                args.batch_size),
        # input_fn=lambda:tf.estimator.inputs.numpy_input_fn(train_observation,train_action,
        #                                  args.batch_size,shuffle=True),
        steps=args.train_steps,

    )


def export_estimator_to_file(classifier,args):
    feature_spec = {'x': tf.FixedLenFeature([1], tf.float32), 'x_dot': tf.FixedLenFeature([1], tf.float32),
                    'theta': tf.FixedLenFeature([1], tf.float32), 'theta_dot': tf.FixedLenFeature([1], tf.float32)}
    serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    #delete previus model in predictor space
    if tf.gfile.Exists(args.predictor_save_model_dir):
        tf.gfile.DeleteRecursively(args.predictor_save_model_dir)
    tf.gfile.MakeDirs(args.predictor_save_model_dir)

    #save it
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


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    tf.logging.set_verbosity(tf.logging.ERROR)



    # lagrinsplass for predictor
    if tf.gfile.Exists(args.predictor_save_model_dir):
        tf.gfile.DeleteRecursively(args.predictor_save_model_dir)
    tf.gfile.MakeDirs(args.predictor_save_model_dir)

    #TODO  Hvis modell ikke eksisterer, lag en med random outputs


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
        model_dir=args.saved_estimator_model_dir )

    print('now using predictor in file path: {}'.format(args.predictor_load_model_dir))
    print('estimators dense2 kernel: {}'.format(classifier.get_variable_value('dense_2/kernel')))  # watch changes in the weights


    new_score_requirement=score_requirement
    classifier_predictor=from_saved_model(args.saved_estimator_model_dir)

    best_training_data=[]
    training_data=[]
    for i in range(number_of_training_loops):
        training_data, new_score_requirement=get_new_training_data(classifier_predictor,new_score_requirement,new_game_runs,goal_steps,best_training_data)
        # print('best training data len {}'.format(len(best_training_data)))
        print('/////////////7////////////////7 \n score_requierment: {}'.format(new_score_requirement))
        if len(training_data)==0:
            print("found no more accepted scores this run")
            break
        train_mode_with_training_data(classifier,training_data,args)
        # print('estimators dense2 kernel, {}'.format(classifier.get_variable_value('dense_2/kernel')))
        export_estimator_to_file(classifier,args)
        classifier_predictor=from_saved_model(args.predictor_load_model_dir)
        # print('estimators dense2 kernel: {}'.format(classifier.get_variable_value('dense_2/kernel')))   #watch changes in the weights
        # classifier_predictor=from_saved_model(args.saved_estimator_model_dir)

        print('Run number {}'.format(i))



    print('len best data {}'.format(len(best_training_data)))
    best_training_data=np.array(best_training_data)

    # train one last time on the best trainingset
    train_mode_with_training_data(classifier, best_training_data, args)
    export_estimator_to_file(classifier, args)
    classifier_predictor = from_saved_model(args.predictor_load_model_dir)
    #this is just to see the results.
    best_training_data2=[]
    training_data, new_score_requirement = get_new_training_data(classifier_predictor, new_score_requirement,
                                                                 new_game_runs, goal_steps, best_training_data2)

    print('estimators dense2 kernel: {}'.format(classifier.get_variable_value('dense_2/kernel')))  # watch changes in the weights

    print('now using predictor in file path: {}'.format(args.predictor_load_model_dir))



if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
