from scipy.misc import imread, imresize
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import glob

from convnet import Convnet
from utils import print_prob, visualize

image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243, 'panda': 388}


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

#ADD DECONV
@ops.RegisterGradient("DeconvRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))


def visualize_convnet():
    sal_map_type = "Deconv_maxlogit"  # change it to get different visualizations
    image_name = 'tabby'  # or using a list to deal with multiple images
    max_pool = True  #indicate whether to add a max-pooling layer

    data_dir = "../data"
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    n_labels = 1000
    n_input = 224

    fns = []
    image_list = []
    label_list = []

    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}.png'.format(image_name))):
        fns.append(os.path.basename(image_path).split('.')[0])
        image = imread(image_path, mode='RGB')
        image = imresize(image, (n_input, n_input)).astype(np.float32)
        image_list.append(image)
        onehot_label = np.array([1 if i == image_dict[image_name] else 0 for i in range(n_labels)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)

    batch_size = batch_img.shape[0]

    # tf session
    sess = tf.Session()

    # construct the graph based on the gradient type we want
    # plain relu vs guidedrelu
    if sal_map_type.split('_')[0] == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            conv_model = Convnet(sess,max_pool)
    # ADD DECONV
    elif sal_map_type.split('_')[0] == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            conv_model = Convnet(sess,max_pool)
    elif sal_map_type.split('_')[0] == 'PlainSaliency':
        conv_model = Convnet(sess,max_pool)
    else:
        raise Exception("Unknown saliency type")

    # saliency gradient to input layer
    if sal_map_type.split('_')[1] == "cost":
        sal_map = tf.gradients(conv_model.cost, conv_model.imgs)[0]
    elif sal_map_type.split('_')[1] == 'maxlogit':
        sal_map = tf.gradients(conv_model.maxlogit, conv_model.imgs)[0]
    elif sal_map_type.split('_')[1] == 'randlogit':
        sal_map = tf.gradients(conv_model.logits[random.randint(0, 999)], conv_model.imgs)[0]
    else:
        raise Exception("Unknown logit gradient type")

    # predict
    probs = sess.run(conv_model.probs, feed_dict={conv_model.images: batch_img})

    # sal_map
    sal_map_val = sess.run(sal_map, feed_dict={conv_model.images: batch_img, conv_model.labels: batch_label})

    for idx in range(batch_size):
        print_prob(probs[idx])
        visualize(sal_map_val[idx], sal_map_type, save_dir, fns[idx])


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    visualize_convnet()
