'''
Author: hiocde
Email: hiocde@gmail.com
Date: 1.17.17
Original/New: 
Domain: 
'''

from datetime import datetime
import time
import os.path
from six.moves import xrange
import numpy as np
import tensorflow as tf
#* Not use relative import like from . import input, something werid.
import input
import alexnet

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_data', 'G:/machine_learning/dataset/Alexnet_tf/paris',
                           """Training data directory.""")
tf.app.flags.DEFINE_string('log_dir', 'G:/machine_learning/models/Alexnet_tf/log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# Constants describing the training process.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 6412
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 12*50
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-5      # Initial learning rate, my experience to set it more litter if occurs NAN(Gradient Exploding).

trainset = input.ImageSet(FLAGS.train_data)


def train():
    """Train AlexNet for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images, labels = trainset.next_batch(FLAGS.batch_size)	# Dont need like alexnet.FLAGS.batch_size

        #*** you dont need feed every step, I built the input pipeline(subgraph) for inference
        logits = alexnet.inference(images)
        loss = regular_loss(logits, labels)
        train_op = step_train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Init all variables on the Graph.
        sess = tf.Session()
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, _loss, model_loss = sess.run([train_op, loss, tf.get_default_graph().get_tensor_by_name("cross_entropy_mean:0")])
            duration = time.time() - start_time

            print(step)
            print(_loss)
            print(model_loss)
            print()
            #print(_labels)
            assert not np.isnan(_loss), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, _loss,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_prefix = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_prefix, global_step=step)

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


def main(argv=None):  # pylint: disable=unused-argument    
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    tf.app.run()	# <=> FLAGS(sys.argv); main()
