'''
Author: hiocde
Email: hiocde@gmail.com
Date: 1.17.17
Original/New: 
Domain: 
'''

import tensorflow as tf
from . import input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")

# Global constants describing the Paris data set.
NUM_CLASSES = 12
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 6412
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 12*50

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    """
    Brief:
          Build the AlexNet model.
    Args:
      images: Images, 4D Tensor from input
    Returns:
      Logits, 2D tensor
    """
    # conv1
    with tf.name_scope('conv1') as scope:
        kernels = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernels, [1, 4, 4, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)

    # lrn1
    # local response normalization.
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # pool1
    pool1 = tf.nn.max_pool(norm1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernels = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernels, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
    print_activations(conv2)

    # lrn2
    # local response normalization.
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # pool2
    pool2 = tf.nn.max_pool(norm2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernels = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernels, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernels = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernels, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernels = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernels, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)

    # fc1
    with tf.variable_scope('fc1') as scope:
        # Move everything into depth so we can perform a single matrix
        # multiply.
        plane = tf.reshape(pool5, [FLAGS.batch_size, -1])
        columns = plane.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights', shape=[columns, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [4096], tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(plane, weights) + biases, name=scope)

    # fc2
    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [4096], tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope)

    # softmax-linear, not normalize at once
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[4096, NUM_CLASSES], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [4096], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope)

    return softmax_linear


def regular_loss(logits, labels):
    """
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Regularized loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(
        cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the
    # weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='regular_loss')


def step_train(total_loss, global_step):
    """
    One step train AlexNet model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from regular_loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for step training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def _add_loss_summaries(total_loss):
    """Add summaries for losses in AlexNet model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from regular_loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the
        # loss as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_decay')
        tf.add_to_collection('losses', weight_decay)
    return var
