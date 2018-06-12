import tensorflow as tf
import numpy as np
import preprocess as p
import model
import cv2
import os

np.set_printoptions(linewidth=150, threshold=np.nan)

model_file = os.path.join('saved_model', 'c_0.932981')


def resize(img):
    h, w = img.shape
    n_w, n_h = 28, 28

    ratio = min(w, h) / max(w, h)
    if w < h:
        n_w = int(28 * ratio)
    elif h < w:
        n_h = int(28 * ratio)
    img_resized = cv2.resize(img, (n_w, n_h), interpolation=cv2.INTER_AREA)
    #cv2.imshow('resized', img_resized)
    #cv2.waitKey(0)
    return img_resized.reshape(n_h, n_w, 1)


def segment(img):
    img = cv2.resize(img, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    kernel = np.ones((3, 70), np.uint8)
    img_dil = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('dilated', img_dil)
    cv2.waitKey(0)

    im2, ctrs, heir = cv2.findContours(img_dil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sort_ctrs = sorted(ctrs, key=lambda ctr: (cv2.boundingRect(ctr)[0], cv2.boundingRect(ctr)[1]))

    words = []
    chars = []
    for i, ctr in enumerate(sort_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        if h < 20:
            continue

        roi = img[y:y + h, x:x + w]
        words.append(roi)
        chars.append(seg_char(roi))

        #cv2.imshow('word' + str(i), roi)
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)
        #cv2.waitKey(0)

    #cv2.imshow('marked', img)
    #cv2.waitKey(0)
    return words, chars

def seg_char(word):
    #word2 = cv2.resize(word, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    word_dil = cv2.dilate(thresh, kernel, iterations=1)
    #cv2.imshow('dilated word', word_dil)
    #cv2.waitKey(0)

    im2, ctrs, heir = cv2.findContours(word_dil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sort_ctrs = sorted(ctrs, key=lambda ctr: (cv2.boundingRect(ctr)[0], cv2.boundingRect(ctr)[1]))

    chars = []
    for i, ctr in enumerate(sort_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        if y+h < word.shape[0]/2:
            continue
        roi = word_dil[y:y + h, x:x + w]
        chars.append(roi)
        #cv2.imshow('letter' + str(i), roi)
        #cv2.waitKey(0)
    return chars



l = 'abcdefghijklmnopqrstuvwxyz'

settings = {'conv1': [3, 3, 1, 8],     # output 14x14x16
            'conv2': [3, 3, 8, 16],    # output 7x7x32
            'conv3': [3, 3, 16, 32],    # output 7x7x128
            'flat':  [7, 7, 32],
            'full1': [512],
            'full2': [26]}

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph(model_file + '.meta')
saver.restore(sess, model_file)

input_placeholder = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
train_placeholder = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
keep_prob_placeholder = tf.get_default_graph().get_tensor_by_name('Placeholder_2:0')
preds = tf.get_default_graph().get_tensor_by_name('full2/add:0')


#input_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
#train_placeholder = tf.placeholder(tf.bool)
#keep_prob_placeholder = tf.placeholder(tf.float32)
#preds = model.create_classifier(input_placeholder, settings,
#                                keep_prob=keep_prob_placeholder,
#                                training=train_placeholder)

#sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

#saver = tf.train.Saver()
#saver.restore(sess, model_file)

softmax = tf.nn.softmax(preds)

filename = 'image.jpg'
img = cv2.imread(filename)

sentence = ''
words, chars = segment(img)
for word, w in zip(words, chars):
    cv2.imshow('word', word)
    cv2.waitKey(0)
    for c in w:
        #cv2.imshow('letter', c)
        #cv2.waitKey(0)
        n_c = resize(c)
        n_c = tf.image.resize_image_with_crop_or_pad(n_c, 28, 28).eval()
        #gray_c = cv2.cvtColor(n_c, cv2.COLOR_BGR2GRAY)
        #print(n_c.reshape(28,28))
        gray_c = n_c.reshape(1, 28,28,1)
        letter_pred = sess.run(softmax, feed_dict={input_placeholder: gray_c,
                                                   train_placeholder: False,
                                                   keep_prob_placeholder: 1.0})
        n = np.argmax(letter_pred)
        letter = l[n]
        sentence += letter
        #print(letter, end='')
    #print(" ", end=)
    sentence += ' '

print(sentence)
sess.close()