import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from imagenet_classes import class_names


# returns the top1 string
def print_prob(prob):
    pred = (np.argsort(prob)[::-1])
    # Gegt top1 label
    top1 = [(pred[0], class_names[pred[0]], prob[pred[0]])]  # pick the most likely class
    print("Top1: ", top1)

    # Get top5 label
    top5 = [(pred[i], class_names[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)


def visualize(sal_map, sal_map_type, load_weights, save_dir, fn, layer_idx=0):
    # normalizations
    sal_map -= np.min(sal_map)
    sal_map /= sal_map.max()

    # plot guided backpropagation separately
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(sal_map)
    ax2.axis('off')
    if layer_idx == 0:
        print("Saving {}_{}_{}.png".format(sal_map_type, fn, load_weights))
        plt.savefig(os.path.join(save_dir, "{}_{}_{}.png".format(sal_map_type, fn, load_weights)))
    else:
        print("Saving {}_{}_{}_{}.png".format(sal_map_type, fn, load_weights, layer_idx))
        plt.savefig(os.path.join(save_dir, "{}_{}_{}_{}.png".format(sal_map_type, fn, load_weights, layer_idx)))
