import tensorflow as tf
import numpy as np
import os
import random
import model
import cv2

from glob import glob

def prep_data(data_dir, test_perc=10, val_perc=10):
    dataset = {'training': [], 'test': [], 'val': []}
    labels = {}

    folders = glob(data_dir)
    classes = [os.path.split(f)[1] for f in folders]
    classes.sort()

    for folder in folders:
        _, c = os.path.split(folder)
        files = glob(folder)
        for f in files:
            l = np.zeros(len(classes))
            l[classes.index(c)] = 1
            labels[f] = l

        random.shuffle(files)
        test_part = int(len(files) * test_perc)
        val_part = int(len(files) * val_perc) + test_part

        dataset['test'] += (files[:test_part])
        dataset['val'] += (files[test_part:val_part])
        dataset['training'] += (files[val_part:])

    return dataset, labels, len(classes)

def load_data(data, labels):
    batch = []
    gtruth = []
    for f in data:
        batch.append(cv2.imread(f))
        gtruth.append(labels[f])
    return batch, gtruth

data_dir = 'data'

dataset, labels, num_classes = prep_data(data_dir)

settings = {'conv1': [],
            'conv2': [],
            'full1': [],
            'full2': [1024, num_classes]}
learning_rate = 0.001
batch_size = 64
epochs = 300


input_placeholder = tf.placeholder(tf.float32)
preds = model.create_classifier(input_placeholder, settings, training=True)

label_placeholder = tf.placeholder(tf.float32)

cross_entrophy = tf.reduce_mean(
                 tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_placeholder, logits=preds))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entrophy)

correct = tf.equal(tf.argmax(label_placeholder, axis=1), tf.argmax(preds, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_batches = int(len(dataset['training']) / batch_size)
    for e in range(epochs):
        print('\repoch: %d' % e, end='')
        if e % int(epochs/10) == 0:
            v_batch, v_labels = load_data(dataset['val'], labels)
            v_accuracy = accuracy.eval(feed_dict={input_placeholder: v_batch,
                                                  label_placeholder: v_labels})
            print('\nepoch: %d validation accuracy: %g' % (e, v_accuracy))
        dataset['training'].sort()
        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            t_batch, t_labels = load_data(dataset['training'][start:end], labels)
            train_step.run(feed_dict={input_placeholder: t_batch,
                                      label_placeholder: t_labels})

    test_input, test_labels = load_data(dataset['test'], labels)
    test_accuracy = accuracy.eval(feed_dict={input_placeholder: test_input,
                                             label_placeholder: test_labels})
    print('\ntest accuracy: %g' % test_accuracy)
    print('saving model..')
    save_dir = os.path.join('saved_model', 'c_%g' % test_accuracy)
    saver.save(sess, save_dir)
