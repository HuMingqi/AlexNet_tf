

import tensorflow as tf
import input
import alexnet

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', 'G:/machine_learning/dataset/Alexnet_tf/paris',
                            """Data input directory when using a product level model(trained and tested).""")

imageset = input.ImageSet(FLAGS.input, False)

