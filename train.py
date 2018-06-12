import tensorflow as tf
import numpy as np
import os
import random
import model

from scipy import io
np.set_printoptions(linewidth=150, threshold=np.nan)


CLASSES = 26

def extract_data(data_file):
    data = io.loadmat(data_file)['dataset']

    train_data = data[0][0][0][0][0][0]
    train_data = train_data.astype(np.float32)
    train_labels = data[0][0][0][0][0][1]

    test_data = data[0][0][1][0][0][0]
    test_data = test_data.astype(np.float32)
    test_labels = data[0][0][1][0][0][1]


    val_set_size = test_data.shape[0]
    val_index = train_data.shape[0] - val_set_size
    val_data = train_data[val_index:]
    val_labels = train_labels[val_index:]

    train_data = train_data[:val_index]
    train_labels = train_labels[:val_index]

    train_labels = convert_to_one_hot(train_labels, CLASSES)
    val_labels = convert_to_one_hot(val_labels, CLASSES)
    test_labels = convert_to_one_hot(test_labels, CLASSES)

    train_data = train_data.reshape(-1, 28, 28, 1, order='F') / 255
    test_data = test_data.reshape(-1, 28, 28, 1, order='F') / 255
    val_data = val_data.reshape(-1, 28, 28, 1, order='F') / 255

    dataset = {'train': (train_data, train_labels),
               'val': (val_data, val_labels),
               'test': (test_data, test_labels)}

    return dataset


def convert_to_one_hot(labels, num_classes):
    labels -= 1
    num_labels = labels.shape[0]
    offset = np.arange(num_labels) * num_classes
    new_labels = np.zeros((num_labels, num_classes))
    new_labels.flat[offset + labels.ravel()] = 1
    return new_labels


def shuffle_data(imgs, labels):
    shuff = np.arange(imgs.shape[0])
    np.random.shuffle(shuff)
    shuff_imgs = imgs[shuff]
    shuff_labels = labels[shuff]
    return shuff_imgs, shuff_labels


def shift_img(img):
    rx = random.randint(-4,4)
    ry = random.randint(-4,4)
    if rx < 0 and ry < 0:
        new_img = img[:, :ry, :rx, :]
        new_img = np.pad(new_img, ((0, 0), (-ry, 0), (-rx, 0), (0, 0)), 'constant')
    elif rx >= 0 and ry >= 0:
        new_img = img[:, ry:, rx:, :]
        new_img = np.pad(new_img, ((0, 0), (0, ry), (0, rx), (0, 0)), 'constant')
    elif rx < 0 and ry >= 0:
        new_img = img[:, ry:, :rx, :]
        new_img = np.pad(new_img, ((0, 0), (0, ry), (-rx, 0), (0, 0)), 'constant')
    else:
        new_img = img[:, :ry, rx:, :]
        new_img = np.pad(new_img, ((0, 0), (-ry, 0), (0, rx), (0, 0)), 'constant')
    return new_img


def build_conf_matrix(y_, labels, conf_matrix):
    y_index = np.argmax(y_, axis=1)
    l_index = np.argmax(labels, axis=1)
    np.add.at(conf_matrix, [l_index, y_index], 1)
    return conf_matrix


def calc_metrics(conf_matrix, epsilon=1e-12):
    tp_idx = np.arange(CLASSES)

    conf_copy = np.copy(conf_matrix)
    conf_copy[tp_idx, tp_idx] = 0

    t_p = conf_matrix[tp_idx, tp_idx]
    f_p = np.sum(conf_copy, axis=1)
    f_n = np.sum(conf_copy, axis=0)

    precision = t_p / (t_p + f_p + epsilon)
    recall = t_p / (t_p + f_n + epsilon)

    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    acc = np.sum(t_p) / np.sum(conf_matrix)

    return precision, recall, f1, acc


data_dir = os.path.join('data', 'emnist-letters')

dataset = extract_data(data_dir)

settings = {'conv1': [3, 3, 1, 8],     # output 14x14x16
            'conv2': [3, 3, 8, 16],    # output 7x7x32
            'conv3': [3, 3, 16, 32],    # output 7x7x128
            'flat':  [7, 7, 32],
            'full1': [512],
            'full2': [CLASSES]}

batch_size = 32
epochs = 30
learning_rate = 0.001
steps = [5, 10, 15, 30]
scale = [0.1, 0.5, 0.1, 0.5]
momentum = 0.9
dropout_prob = 0.5


input_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
train_placeholder = tf.placeholder(tf.bool)
keep_prob_placeholder = tf.placeholder(tf.float32)
preds = model.create_classifier(input_placeholder, settings,
                                keep_prob=keep_prob_placeholder,
                                training=train_placeholder)

label_placeholder = tf.placeholder(tf.float32)
cross_entrophy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_placeholder,
                                                                           logits=preds))

lr_placeholder = tf.placeholder(tf.float32)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entrophy)

softmax = tf.nn.softmax(preds)
correct = tf.equal(tf.argmax(label_placeholder, axis=1), tf.argmax(softmax, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
num_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_imgs = dataset['train'][0]
    train_labels = dataset['train'][1]

    val_imgs = dataset['val'][0]
    val_labels = dataset['val'][1]

    test_imgs = dataset['test'][0]
    test_labels = dataset['test'][1]

    num_train_batches = int(train_imgs.shape[0] / batch_size)
    val_batch_size = int(val_imgs.shape[0] / 5)
    num_val_batches = int(val_imgs.shape[0] / val_batch_size)
    val_imgs, val_labels = shuffle_data(val_imgs, val_labels)
    c_val_batch = 0
    i = 0
    for e in range(epochs):
        print('\nepoch: %d/%d' % (e, epochs))
        if e % 10 == 0:
            if c_val_batch == 5:
                c_val_batch = 0
                val_imgs, val_labels = shuffle_data(val_imgs, val_labels)
            start = c_val_batch * val_batch_size
            end = start + val_batch_size
            v_batch = val_imgs[start:end]
            v_batch = shift_img(v_batch)
            v_labels = val_labels[start:end]
            v_accuracy, loss = sess.run([accuracy, cross_entrophy], feed_dict={input_placeholder: v_batch,
                                                                               train_placeholder: False,
                                                                               keep_prob_placeholder: 1.0,
                                                                               label_placeholder: v_labels})
            print('validation accuracy: %g\tloss: %g' % (v_accuracy, loss))
            c_val_batch += 1

        train_imgs, train_labels = shuffle_data(train_imgs, train_labels)

        if e in steps:
            learning_rate *= scale[i]
            i += 1
        for b in range(num_train_batches):
            start = b * batch_size
            end = start + batch_size
            t_batch = train_imgs[start:end]
            t_batch = shift_img(t_batch)
            t_labels = train_labels[start:end]
            train_step.run(feed_dict={input_placeholder: t_batch,
                                      train_placeholder: True,
                                      keep_prob_placeholder: dropout_prob,
                                      label_placeholder: t_labels,
                                      lr_placeholder: learning_rate})
            print('\rtraining batch: %d/%d' % (b, num_train_batches), end='')

    print('\nevaluating against test set..')
    num_test_iters = int(test_imgs.shape[0] / 10)
    test_accuracy = 0
    conf_matrix = np.zeros((CLASSES, CLASSES), dtype=np.int32)
    for t in range(num_test_iters):
        start = t * num_test_iters
        end = start + num_test_iters
        t_input = test_imgs[start:end]
        t_labels = test_labels[start:end]
        t_correct, y_ = sess.run([num_correct, preds],feed_dict={input_placeholder: t_input,
                                                                   train_placeholder: False,
                                                                   keep_prob_placeholder: 1.0,
                                                                   label_placeholder: t_labels})

        conf_matrix = build_conf_matrix(y_, t_labels, conf_matrix)

    print(conf_matrix)

    precision, recall, f1, acc = calc_metrics(conf_matrix)
    print('accuracy: %g' % acc)

    print('precision:')
    print('mean: %g\t min: %g[%d]\t max: %g[%d]'
          % (np.mean(precision), np.amin(precision),
             np.argmin(precision), np.amax(precision), np.argmax(precision)))
    print(precision)
    print('recall:')
    print('mean: %g\t min: %g[%d]\t max: %g[%d]'
          % (np.mean(recall), np.amin(recall),
             np.argmin(recall), np.amax(recall), np.argmax(recall)))
    print(recall)
    print('f1:')
    print('mean: %g\t min: %g[%d]\t max: %g[%d]'
          % (np.mean(f1), np.amin(f1),
             np.argmin(f1), np.amax(f1), np.argmax(f1)))
    print(f1)


    print('saving model..')
    save_dir = os.path.join('saved_model', 'c_%g' % acc)
    saver.save(sess, save_dir)