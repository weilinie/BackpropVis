import numpy as np
import tensorflow as tf


class FC(object):

    def __init__(self, sess=None):

        self.layers_dic = {}
        self.parameters = []

        self.n_labels = 1000
        self.n_input = 64

        # zero-mean input
        with tf.name_scope('input') as scope:
            self.images = tf.placeholder(tf.float32, [None, self.n_input, self.n_input, 3])
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.imgs = self.images - mean
            self.layers_dic['imgs'] = self.imgs

        with tf.name_scope('output') as scope:
            self.labels = tf.placeholder(tf.float32, [None, self.n_labels])

        self.fc_layers()

        self.logits = self.fc2l

        self.probs = tf.nn.softmax(self.logits)
        self.cost = tf.reduce_sum((self.probs - self.labels) ** 2)
        self.maxlogit = tf.reduce_max(self.logits, axis=1)

        if sess is not None:
            self.init(sess)
        else:
            print("fc initialization failed ... ")

    def fc_layers(self):

        # fc_len = 4096
        # fc_len = 1024
        fc_len = 5000

        shape = int(np.prod(self.imgs.get_shape()[1:]))
        self.pool5_flat = tf.reshape(self.imgs, [-1, shape])

        # fc1
        with tf.name_scope('fc1') as scope:

            fc1w = tf.Variable(tf.truncated_normal([shape, fc_len],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[fc_len], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc1l = tf.nn.bias_add(tf.matmul(self.pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]
            self.layers_dic['fc1'] = self.fc1

        # fc2
        with tf.name_scope('fc3') as scope:
            fc2w = tf.Variable(tf.truncated_normal([fc_len, self.n_labels],
                                                         dtype=tf.float32,
                                                         stddev=1e1), name='weights')
            fc2b = tf.Variable(tf.constant(0.0, shape=[self.n_labels], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)

            self.parameters += [fc2w, fc2b]
            self.layers_dic['fc2'] = self.fc2l

    def init(self, sess):
        sess.run(tf.global_variables_initializer())