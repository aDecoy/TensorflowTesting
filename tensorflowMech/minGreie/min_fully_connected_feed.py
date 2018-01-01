"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflowMech.minGreie import min_mnist

# Basic model parameters as external flags.
FLAGS = None #lager basic default model

def do_evaluation(sess,
                  eval_correct,
                  image_placeholder,
                  labels_placeholder,
                  data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    true_count=0 # Antall riktige svar
    steps_per_epoch= data_set.num_examples //FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size #hvorfor ikke bare data.set.num_examples? bruker sikkert bare så mange at det går opp perfekt i batch size
    for step in range(steps_per_epoch):

        image_feed,label_feed = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
        feed_dict = {
            image_placeholder : image_feed,
            labels_placeholder : label_feed
        }
        true_count+=  sess.run(eval_correct,feed_dict=feed_dict)
    precision= float(true_count)/num_examples
    print(' Antall eksempler: %d   Antall riktige: %d   Presision @ 1: %0.04f' %
          (num_examples,true_count,precision))


def run_training():
    data_sets=input_data.read_data_sets(FLAGS.input_data_dir,FLAGS.fake_data)
    with tf.Graph().as_default():

            #på tide å initiere variablene
        with tf.Session() as sess:
            # placeholders
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, mnist.IMAGE_PIXELS)) #array med bilder. 2d
            LABELS_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))   #array med riktig svar. 1d


            #MNIST spesifike funksjoner vi vil kalle. Bygger instruksjoner.

            # Build a Graph that computes predictions from the inference model.
            #bruker grafen med input, hidden etc til å gi oss prediction/logits. Gis til loss
            logits = min_mnist.inference(images_placeholder,
                                     FLAGS.hidden1,
                                     FLAGS.hidden2)
            #finn feilen på logits/predictions/anslag/forutsigelse  . Loss brukes for hver trening sammen med training_op
            loss= min_mnist.loss(logits=logits,labels=LABELS_placeholder)

            #tren bakover
            train_op= min_mnist.training(loss=loss,learningrate=FLAGS.learningrate)

            # Add the Op to compare the logits to the labels during evaluation.
            eval_correct = min_mnist.evaluation(logits, LABELS_placeholder)

            # Build the summary Tensor based on the TF collection of Summaries.
            summary = tf.summary.merge_all()


            init = tf.global_variables_initializer()

            #thank god for checkpoints. praise the sun!
            saver = tf.train.Saver()

            #summary writer skriver ut sammendrag og graph
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph)

            sess.run(init)
        #på tide å kjøre treningsdelen av session
            for step in range(FLAGS.max_steps):
                #men først litt debugging stuff
                start_time = time.time()

                #så feed_dict
                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                # Create the feed_dict for the placeholders filled with the next
                # `batch size` examples.
                images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size,
                                                               FLAGS.fake_data)
                feed_dict = {
                    images_placeholder: images_feed,
                    LABELS_placeholder: labels_feed,
                }
                    #kjør en batch. gi den loss og hvordan mnist modellen skal trenes. også gi den feed dict som er de nye labels og data
                _, loss_value = sess.run([train_op,loss],feed_dict=feed_dict)

                duration= time.time()-start_time

                #print statusrapport fra tid til annen.
                if step %100==0:
                    #print status to stdout
                    print('Step %d: loss = %.2f (%.3f sec)' % (step,loss_value,duration) )
                    #oppdater event fila
                    summary_str = sess.run(summary,feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str,step)
                    summary_writer.flush()# sikkert mye i cashen?

                    #bonfire!!                 #Lagre og evaluer modellen, fra tid til annen

                if (step+1) % 1000 == 0 or (step+1) == FLAGS.max_steps: #få med siste
                        checkpoint_file = os.path.join(FLAGS.log_dir,'model.ckpt')
                        saver.save(sess,checkpoint_file,global_step=step)

                        print('Training Data Eval:')
                        do_evaluation(sess,
                                eval_correct,
                                images_placeholder,
                                LABELS_placeholder,
                                data_sets.train)
                        #validation set
                        print('Validation Data Evaluation:')
                        do_evaluation(sess,eval_correct,images_placeholder,LABELS_placeholder,data_sets.validation)
                        #evaluate agianst test set
                        print('Test Data Eval:')
                        do_evaluation(sess,eval_correct,images_placeholder,LABELS_placeholder,data_sets.test)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--learningrate',
                        type=float,
                        default=0.01,
                        help='Initial learning rate.'
                        )
    parser.add_argument('--max_steps',
                        type=int,
                        default=2000,
                        help='max itertions')
    parser.add_argument('--hidden1',
                        type=int,
                        default=128,
                        help='Antall nevroner i hidden 1 laget.')
    parser.add_argument('--hidden2',
                            type=int,
                            default=32,
                            help='Antall nevroner i hidden 2 laget.')
    parser.add_argument('--batch_size',
                            type=int,
                            default=100,
                            help='Antall exempler i hver treningsbatch. (Må gå opp i totalt antall treningsett)')
    parser.add_argument('--input_data_dir',
                        type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                           'tensorflow/mnist/input_data'),
                        help='Hvor vi putter input data ')
    parser.add_argument('--log_dir',
                        type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/AppData/Local/Temp/'),
                           'tensorflow/mnist/logs/min_fully_connected_feed'),
                        help='Hvor vi putter log data ')
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+ unparsed)








