import tensorflow as tf


def initialzie_weights(shape):
    if len(shape) == 4:
        std = 2 / (shape[0]*shape[1]*shape[2])
    else:
        std = 2 / (shape[0] * shape[1])
    w = tf.truncated_normal(shape, stddev=std)
    b = tf.zeros([shape[-1]])
    return tf.Variable(w), tf.Variable(b)


def conv(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_classifier(input, settings, keep_prob=1.0, training=False):
    ""
    # normalize
    mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=True)
    norm_input = (input - mean) / tf.sqrt(variance)

    '''   Convolution Layer 1   '''
    w, h, c, nf = settings['conv1']
    W1, b1 = initialzie_weights([w, h, c, nf])
    with tf.name_scope('conv1'):
        conv1 = conv(norm_input, W1)
        relu1 = tf.nn.leaky_relu(conv1)
        bn1 = tf.layers.batch_normalization(relu1, axis=3, training=training)
        pool1 = pool(relu1)

    '''   Convolution Layer 2   '''
    w, h, c, nf = settings['conv2']
    W2, b2 = initialzie_weights([w, h, c, nf])
    with tf.name_scope('conv2'):
        conv2 = conv(pool1, W2)
        relu2 = tf.nn.leaky_relu(conv2)
        bn2 = tf.layers.batch_normalization(relu2, axis=3, training=training)
        pool2 = pool(relu2)

    '''   Convolution Layer 3   '''
    w, h, c, nf = settings['conv3']
    W3, b3 = initialzie_weights([w, h, c, nf])
    with tf.name_scope('conv3'):
        conv3 = conv(pool2, W3)
        relu3 = tf.nn.leaky_relu(conv3)
        bn3 = tf.layers.batch_normalization(relu3, axis=3, training=training)

    '''   Flatten   '''
    w, h, nf = settings['flat']
    flatten = tf.reshape(relu3, [-1, w*h*nf])

    '''   Fully Connected Layer 1   '''
    n_out1 = settings['full1'][0]
    W3, b3 = initialzie_weights([w*h*nf, n_out1])
    with tf.name_scope('full1'):
        full1 = tf.matmul(flatten, W3)
        relu_full = tf.nn.leaky_relu(full1 + b3)

    '''   Dropout   '''
    if training != False:
        dropout = tf.nn.dropout(relu_full, keep_prob=keep_prob)
    else:
        dropout = relu_full

    '''   Fully Connected Layer 2   '''
    n_out2 = settings['full2'][0]
    W4, b4 = initialzie_weights([n_out1, n_out2])
    with tf.name_scope('full2'):
        full2 = tf.matmul(dropout, W4)
        z = full2 + b4

    '''   Softmax   '''
    #prediction = tf.nn.softmax(z)
    prediction = z

    return prediction