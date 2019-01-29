from scipy.misc import imread, imresize
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import glob

from fc import FC
from utils import print_prob, visualize

image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243, 'panda': 388}


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

#ADD DECONV NET
@ops.RegisterGradient("DeconvRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))


def visualize_fc():
    sal_map_type = "GuidedBackprop_maxlogit"  # change it to get different visualizations
    image_name = 'tabby' # or using a list to deal with multiple images

    data_dir = "../data"
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    n_labels = 1000
    dim_input = 64

    fns = []
    image_list = []
    label_list = []

    # load in the original image
    for image_path in glob.glob(os.path.join(data_dir, '{}.png'.format(image_name))):
        fns.append(os.path.basename(image_path).split('.')[0])
        image = imread(image_path, mode='RGB')
        image = imresize(image, (dim_input, dim_input)).astype(np.float32)
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
            fc_models = FC(sess)

    elif sal_map_type.split('_')[0] == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            fc_models = FC(sess)

    elif sal_map_type.split('_')[0] == 'PlainSaliency':
        fc_models = FC(sess)

    else:
        raise Exception("Unknown saliency type")

    # saliency gradient to input layer
    if sal_map_type.split('_')[1] == "cost":
        sal_map = tf.gradients(fc_models.cost, fc_models.imgs)[0]
    elif sal_map_type.split('_')[1] == 'maxlogit':
        sal_map = tf.gradients(fc_models.maxlogit, fc_models.imgs)[0]
    elif sal_map_type.split('_')[1] == 'randlogit':
        sal_map = tf.gradients(fc_models.logits[random.randint(0, 999)], fc_models.imgs)[0]
    else:
        raise Exception("Unknown logit gradient type")

    # predict
    probs = sess.run(fc_models.probs, feed_dict={fc_models.images: batch_img})

    # sal_map and conv_grad
    sal_map_val = sess.run(sal_map, feed_dict={fc_models.images: batch_img, fc_models.labels: batch_label})

    for idx in range(batch_size):
        print_prob(probs[idx])
        visualize(sal_map_val[idx], sal_map_type, save_dir, fns[idx])

if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    visualize_fc()
