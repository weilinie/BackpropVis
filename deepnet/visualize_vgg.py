from scipy.misc import imread, imresize
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import glob

from vgg16 import Vgg16
from utils import print_prob, visualize

image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243, 'panda': 388}  # randomly chosen images to visualize

layers = [
    'conv1_1',
    'conv1_2',
    'conv2_1',
    'conv2_2',
    'conv3_1',
    'conv3_2',
    'conv3_3',
    'conv4_1',
    'conv4_2',
    'conv4_3',
    'conv5_1',
    'conv5_2',
    'conv5_3',
    'fc1',
    'fc2',
    'fc3']


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))


@ops.RegisterGradient("DeconvRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))


def prepare_vgg(sal_type, sess):
    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            vgg = Vgg16(sess=sess)

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            vgg = Vgg16(sess=sess)

    elif sal_type == 'PlainSaliency':
        vgg = Vgg16(sess=sess)

    else:
        raise Exception("Unknown saliency type")

    return vgg


def get_saliency(vgg, logit_type):
    # saliency gradient to input layer
    if logit_type == "cost":
        sal_map = tf.gradients(vgg.cost, vgg.imgs)[0]
    elif logit_type == 'maxlogit':
        sal_map = tf.gradients(vgg.maxlogit, vgg.imgs)[0]
    elif logit_type == 'randlogit':
        sal_map = tf.gradients(vgg.logits[random.randint(0, 999)], vgg.imgs)[0]
    else:
        raise Exception("Unknown logit gradient type")
    return sal_map


def visualize_vgg():
    sal_map_type = "GuidedBackprop_maxlogit"  # change it to get different visualizations
    load_weights = 'random'  # how to load the weights of vgg16
    image_name = 'tabby'  # or using a list to deal with multiple images

    data_dir = "../data"
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sal_type = sal_map_type.split('_')[0]
    logit_type = sal_map_type.split('_')[1]

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
    # tf.reset_default_graph()
    vgg = prepare_vgg(sal_type, sess)

    # saliency gradient to input layer
    sal_map = get_saliency(vgg, logit_type)

    if load_weights in ['part', 'reverse', 'only']:
        for layer_idx, layer in enumerate(layers):
            if load_weights == 'part':
                # fill the first "idx" layers with the trained weights
                # randomly initialize the rest
                vgg.load_weights_part(layer_idx * 2 + 1, 'vgg16_weights.npz', sess)

            elif load_weights == 'reverse':
                # do not fill the first "idx" layers with the trained weights
                # randomly initialize them
                vgg.load_weights_reverse(layer_idx * 2 + 1, 'vgg16_weights.npz', sess)

            elif load_weights == 'only':
                # do not load a specific layer ("idx") with the trained weights
                # randomly initialize it
                vgg.load_weights_only(layer_idx * 2 + 1, 'vgg16_weights.npz', sess)

            # sal_map
            sal_map_val = sess.run(sal_map, feed_dict={vgg.images: batch_img, vgg.labels: batch_label})

            # predict
            probs = sess.run(vgg.probs, feed_dict={vgg.images: batch_img})

            for idx in range(batch_size):
                print_prob(probs[idx])
                visualize(sal_map_val[idx], sal_map_type, load_weights, save_dir, fns[idx], layer_idx)

    elif load_weights in ['random', 'trained']:
        # different options for loading weights
        if load_weights == 'trained':
            vgg.load_weights('vgg16_weights.npz', sess)

        elif load_weights == 'random':
            vgg.init(sess)

        # sal_map
        sal_map_val = sess.run(sal_map, feed_dict={vgg.images: batch_img, vgg.labels: batch_label})

        # predict
        probs = sess.run(vgg.probs, feed_dict={vgg.images: batch_img})

        for idx in range(batch_size):
            print_prob(probs[idx])
            visualize(sal_map_val[idx], sal_map_type, load_weights, save_dir, fns[idx])

    else:
        raise Exception("Unknown load_weights type")


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    visualize_vgg()
