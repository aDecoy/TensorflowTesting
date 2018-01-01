from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import math

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images,hidden1_units,hidden2_units):
    '''bygger grafen
    :param images:
    :param hidden1_units: antall nevroner i hidden1 nevroner
    :param hidden2_units: antall nevroner i hidden 2
    :return:

    Returns:
    softmax_linear: Output
    tensor
    with the computed logits.'''

    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            #starter med random vekter
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],#antall from nodes, og to nodes
                stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)#kalkulerer node verdier med
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            #starter med random vekter
            tf.truncated_normal([hidden1_units, hidden2_units],#antall from nodes, og to nodes
                stddev=1.0/math.sqrt(float(hidden1_units))),  #random verdier rundt stddev
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
        hidden2= tf.nn.relu(tf.matmul(hidden1,weights)+biases) #kalkulerer node verdier med relu
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            #starter med random vekter
            tf.truncated_normal([hidden2_units, NUM_CLASSES],#antall from nodes, og to nodes
                stddev=1.0/math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
        logits= tf.matmul(hidden2,weights)+biases #kalkulerer node verdier uten relu på siste.
    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss,learningrate):
    tf.summary.scalar('loss',loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)

    global_step = tf.Variable(0,name='global_step',trainable=False)#number of steps taken
    train_op = optimizer.minimize(loss,global_step=global_step) #makes it do one round of training
    return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  #husk at logits er en array med hvor sannsynleig det er at den er av hver av "cases'ene"
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))




