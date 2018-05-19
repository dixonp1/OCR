import tensorflow as tf


def initialzie_weights(shape):
    w = tf.truncated_normal(shape, stddev=0.01)
    b = tf.zeros([shape[-1]])
    return tf.Variable(w), tf.Variable(b)


def conv(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_classifier(input, settings, training=False):
    ""

    '''   Convolution Layer 1   '''
    w, h, c, nf = settings['conv1']
    W1, b1 = initialzie_weights([w, h, c, nf])
    with tf.name_scope('conv1'):
        conv1 = conv(input, W1)
        relu1 = tf.nn.relu(conv1 + b1)
        pool1 = pool(relu1)

    '''   Convolution Layer 2   '''
    w, h, c, nf = settings['conv2']
    W2, b2 = initialzie_weights([w, h, c, nf])
    with tf.name_scope('conv2'):
        conv2 = conv(pool1, W2)
        relu2 = tf.nn.relu(conv2 + b2)
        pool2 = pool(relu2)

    '''   Flatten   '''
    n_out1 = settings['full1']
    flatten = tf.reshape(pool2, [-1, w*h*nf])

    '''   Fully Connected Layer 1   '''
    W3, b3 = initialzie_weights([w*h*nf, n_out1])
    with tf.name_scope('full1'):
        full1 = tf.matmul(flatten, W3)
        relu3 = tf.nn.relu(full1 + b3)

    '''   Fully Connected Layer 2   '''
    n_out2 = settings['full2']
    W4, b4 = initialzie_weights([n_out1, n_out2])
    with tf.name_scope('full2'):
        full2 = tf.matmul(relu3, W4)
        z = full2 + b4

    '''   Softmax   '''
    if training:
        prediction = z
    else:
        prediction = tf.nn.softmax(z)

    return prediction


def load_model(sess, save_dir):
    saver = tf.train.Saver
    saver.restore(sess, save_dir)