import numpy as np
import tensorflow as tf


class Convnet(object):
    def __init__(self, sess=None, max_pool=False):

        self.layers_dic = {}
        self.parameters = []

        self.n_labels = 1000
        self.n_input = 224

        # zero-mean input
        with tf.name_scope('input') as scope:
            self.images = tf.placeholder(tf.float32, [None, self.n_input, self.n_input, 3])
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.imgs = self.images - mean
            self.layers_dic['imgs'] = self.imgs

        with tf.name_scope('output') as scope:
            self.labels = tf.placeholder(tf.float32, [None, self.n_labels])

        self.convlayers()
        # ADD Max-pooling
        if max_pool:
            self.max_pooling()
        self.fc_layers()

        self.logits = self.fc2l

        self.probs = tf.nn.softmax(self.logits)
        self.cost = tf.reduce_sum((self.probs - self.labels) ** 2)
        self.maxlogit = tf.reduce_max(self.logits, axis=1)

        if sess is not None:
            self.init(sess)
        else:
            print("convnet initialization failed ... ")

    def convlayers(self):

        conv_ch1 = 16  # adjust this value to control the visual quality of visualizations

        # conv1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([7, 7, 3, conv_ch1], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv1_1'] = self.conv1

        self.convnet_out = self.conv1

    def max_pooling(self):
        with tf.name_scope('max_pool') as scope:
            self.max_pool = tf.nn.max_pool(self.convnet_out, ksize=[1, 9,9, 1], strides=[1,1,1,1],padding='SAME')
            self.layers_dic["max_pool"] = self.max_pool
        self.convnet_out = self.max_pool


    def fc_layers(self):

        shape = int(np.prod(self.convnet_out.get_shape()[1:]))
        self.conv_flat = tf.reshape(self.convnet_out, [-1, shape])

        flat_len = self.conv_flat.get_shape().as_list()[1]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([flat_len, self.n_labels],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(0.0, shape=[self.n_labels], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc2l = tf.nn.bias_add(tf.matmul(self.conv_flat, fc2w), fc2b)

            self.parameters += [fc2w, fc2b]
            self.layers_dic['fc2'] = self.fc2l

    def init(self, sess):
        sess.run(tf.global_variables_initializer())
