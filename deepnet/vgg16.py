import numpy as np
import tensorflow as tf


class Vgg16(object):
    def __init__(self, weights=None, plain_init=None, sess=None, act_type='relu', pool_type='maxpool'):

        self.act_type = act_type
        self.pool_type = pool_type

        self.layers_dic = {}
        self.parameters = []
        self.layers_W_dic = {}

        # zero-mean input
        with tf.name_scope('input') as scope:
            self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.imgs = self.images - mean
            self.layers_dic['imgs'] = self.imgs

        with tf.name_scope('output') as scope:
            self.labels = tf.placeholder(tf.float32, [None, 1000])

        self.convlayers()
        self.fc_layers()

        self.logits = self.fc3l
        self.probs = tf.nn.softmax(self.logits)
        self.cost = tf.reduce_sum((self.probs - self.labels) ** 2)
        self.maxlogit = tf.reduce_max(self.logits, axis=1)

        if plain_init and sess is not None:
            self.init(sess)

        elif weights is not None and sess is not None:
            self.load_weights(weights, sess)

        else:
            print("vgg16 initialization failed ... ")

    def act(self, tensor, name):

        if self.act_type == 'relu':
            return tf.nn.relu(tensor, name=name)

        if self.act_type == 'softplus':
            return tf.nn.softplus(tensor, name=name)

    def pool(self, tensor, name):

        if self.pool_type == 'maxpool':
            return tf.nn.max_pool(tensor,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name=name)

        if self.pool_type == 'avgpool':
            return tf.nn.avg_pool(tensor,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name=name)

    def convlayers(self):

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv1_1'] = self.conv1_1
            self.layers_W_dic['conv1_1'] = kernel

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv1_2'] = self.conv1_2
            self.layers_W_dic['conv1_2'] = kernel

        # pool1
        self.pool1 = self.pool(tensor=self.conv1_2, name='pool1')
        self.layers_dic['pool1'] = self.pool1

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv2_1'] = self.conv2_1
            self.layers_W_dic['conv2_1'] = kernel

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv2_2'] = self.conv2_2
            self.layers_W_dic['conv2_2'] = kernel

        # pool2
        self.pool2 = self.pool(tensor=self.conv2_2, name='pool2')
        self.layers_dic['pool2'] = self.pool2

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv3_1'] = self.conv3_1
            self.layers_W_dic['conv3_1'] = kernel

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv3_2'] = self.conv3_2
            self.layers_W_dic['conv3_2'] = kernel

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv3_3'] = self.conv3_3
            self.layers_W_dic['conv3_3'] = kernel

        # pool3
        self.pool3 = self.pool(tensor=self.conv3_3, name='pool3')
        self.layers_dic['pool3'] = self.pool3

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv4_1'] = self.conv4_1
            self.layers_W_dic['conv4_1'] = kernel

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv4_2'] = self.conv4_2
            self.layers_W_dic['conv4_2'] = kernel

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv4_3'] = self.conv4_3
            self.layers_W_dic['conv4_3'] = kernel

        # pool4
        self.pool4 = self.pool(tensor=self.conv4_3, name='pool4')
        self.layers_dic['pool4'] = self.pool4

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv5_1'] = self.conv5_1
            self.layers_W_dic['conv5_1'] = kernel

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv5_2'] = self.conv5_2
            self.layers_W_dic['conv5_2'] = kernel

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = self.act(tensor=out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv5_3'] = self.conv5_3
            self.layers_W_dic['conv5_3'] = kernel

        # pool5
        self.pool5 = self.pool(tensor=self.conv5_3, name='pool5')
        self.layers_dic['pool5'] = self.pool5

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = self.act(tensor=fc1l, name=scope)

            self.parameters += [fc1w, fc1b]
            self.layers_dic['fc1'] = self.fc1
            self.layers_W_dic['fc1'] = fc1w

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = self.act(tensor=fc2l, name=scope)

            self.parameters += [fc2w, fc2b]
            self.layers_dic['fc2'] = self.fc2
            self.layers_W_dic['fc2'] = fc2w

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

            self.parameters += [fc3w, fc3b]
            self.layers_dic['fc3'] = self.fc3l
            self.layers_W_dic['fc3'] = fc3w

    def load_weights_part(self, n, weight_file, sess):

        # fill the first "idx" layers with the trained weights
        # n = idx * 2 + 1
        # randomly initialize the rest
        self.init(sess)
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i <= n:
                sess.run(self.parameters[i].assign(weights[k]))

    def load_weights_reverse(self, n, weight_file, sess):

        # do not fill the first "idx" layers with the trained weights
        # n = idx * 2 + 1
        # randomly initialize them
        self.init(sess)
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i > n:
                sess.run(self.parameters[i].assign(weights[k]))

    def load_weights_only(self, n, weight_file, sess):

        # do not load a specific layer ("idx") with the trained weights
        # n = idx * 2 + 1
        # randomly initialize it
        self.init(sess)
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i != n and i != n - 1:
                sess.run(self.parameters[i].assign(weights[k]))

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))

    def init(self, sess):
        sess.run(tf.global_variables_initializer())
