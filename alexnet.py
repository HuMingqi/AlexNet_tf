'''
Author: hiocde
Email: hiocde@gmail.com
Date: 1.17.17
Original/New:
Domain:
'''

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# model cmd parameter, input standard.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
# model output standard.
NUM_CLASSES = 12


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
        #** assign to use scope not scope.name , because it's not tf.variable_scope
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
        # Move everything into depth so we can perform a single matrix multiply.
        # plane = tf.reshape(pool5, [FLAGS.batch_size, -1]) #it's not good to depend on global variable!
        plane = tf.reshape(pool5, [images.get_shape()[0].value, -1])
        columns = plane.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights', shape=[columns, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [4096], tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(plane, weights) + biases, name=scope.name)

    # fc2
    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [4096], tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

    # softmax-linear, not normalize at once
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[4096, NUM_CLASSES], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return softmax_linear


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

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
