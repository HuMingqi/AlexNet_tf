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
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', 'G:/machine_learning/models/Alexnet_tf/log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
trainset = input.ImageSet(FLAGS.train_data)


def train():
    """Train AlexNet for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images, labels = trainset.next_batch(alexnet.FLAGS.batch_size)

        #*** you dont need feed every step, I built the input pipeline(subgraph) for inference
        logits = alexnet.inference(images)
        loss = alexnet.regular_loss(logits, labels)
        train_op = alexnet.step_train(loss, global_step)

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
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            
            print(step)
            print(loss_value)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_prefix = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_prefix, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument    
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
