Some simple codes for the paper https://arxiv.org/abs/1805.07039

## Requirement. 
We suggest you run the platform under Python 3.6+ with following libs:
* **TensorFlow >= 1.4.0**
* Numpy 1.12.1
* Scipy 0.19.0
* NLTK 3.2.3

## Get Started

1. Visualize the fully-connected networks (FC): 
```bash
cd fc
# default setting: Guided Backpropagation (GBP) with the max logit
python3 visualize_fc.py
```
2. Visualize the three-layer CNN: 
```bash
cd convnet
# default setting: Guided Backpropagation (GBP) with the max logit
python3 visualize_convnet.py
```
3. Visualize the VGG-16 net: 
```bash
cd deepnet
# default setting: Guided Backpropagation (GBP) with the max logit
python3 visualize_vgg.py
```

Note: In the visualize_[fc, convnet, vgg].py, you could change the variable 
**"sal_type" from 
['GuidedBackprop', 'Deconv', 'PlainSaliency']**
to get different visualizations, and change the variable 
**"logit_type" 
from ['maxlogit', 'randlogit', 'cost']**
to get different ways to compute visualizations. 
Particularly in the visualize_vgg.py, you can change the variable **"load_weights" from 
['random', 'trained', 'part', 'reverse', 'only']** to decide different weight loading methods.

Finally, the pre-trained VGG-16 net is based on https://www.cs.toronto.edu/~frossard/post/vgg16/. 
You first need to download the trained weights [vgg16_weights.npz](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)
into the *deepnet* directory.


## Reference
```bash
@InProceedings{pmlr-v80-nie18a,
  title = 	 {A Theoretical Explanation for Perplexing Behaviors of Backpropagation-based Visualizations},
  author = 	 {Nie, Weili and Zhang, Yang and Patel, Ankit},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  year = 	 {2018},
}
```