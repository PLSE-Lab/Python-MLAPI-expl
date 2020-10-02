#!/usr/bin/env python
# coding: utf-8

# ## Restricted Boltzmann machines and pretraining
# 
# ### About
# 
# Restricted Boltzmann machines are a simple but non-trivial type of neural network. RBMs are interesting because they can be used with a larger neural network to perform **pre-training**: finding "good default" weights for a neural network without having to do more expensive backpropogation work, by lifting weights from a simpler network that we train beforehand that we teach to approximately reconstruct the input data. Pre-training has historically been important in the development of neural networks because it allows training large neural networks faster and more reliably than would otherwise be possible. We will see how they do so in this notebook.
# 
# Note that pre-training has fallen somewhat out of favor on the bleeding edge because it has been somewhat obviated by [better optimization algorithms](https://www.kaggle.com/residentmario/keras-optimizers). Additionally RBMs have been somewhat displaced by more complicated but more robust autoencoders (topic for the future).
# 
# RBMs are a shallow network with just two layers: an input layer and a hidden layer. The nodes on the input layer are fully connected to the nodes on the hidden layer...and the nodes on the hidden layers are in turn fully connected to the nodes on the input layer. Signal thus bounces between the nodes on the nodes on the input layer and the hidden layer and back again.
# 
# The RBM recycles weights. The weight for the forward connection between an input node and a hidden node is the same as the weight of the *backwards* connection between the hidden node and the input node.
# 
# The RBM learns on each pass. On the forward (projection) pass, the reconstruction is treated as the target, and gradient descent is used to adjust the weights on the input layer. On the backwards pass, the original dataset is treated as the target, and gradient descent is used to adjust the weights on the hidden layer. Since both backwards and forwards weights are the same, each pass simultaneously adjusts both forward and backwards weights.
# 
# We are effectively managing two distributions. One distribution is a function of the input data, $f(X) = X_p$. The other distribution is a function of that output, $g(X_p) = X_q$. By making the weights that control $f(X)$ and $g(X)$ the same, and by performing backpropogration on each pass through the network, we are effectively making $f(X_p)$ and $g(X_p)$ converge to roughly the same distribution. In other words, $f(X) \approx g^{-1}(X)$ and vice versa: the forward and backwards functions become roughly inverses of one another.
# 
# To measure the distance between its estimated probability distribution and the ground-truth distribution of the input, RBMs use Kullback Leibler Divergence.  KL-Divergence measures the diverging areas under the distribution curves formed by $X_p$ and $X_q$. Low divergence means that the distributions are similar, which in turn means that they are closer approximations of one another.
# 
# Doing all of this earns you weight and bias terms on your hidden nodes that create an effective reconstruction of your input layer when you run them back. If you have $n$ layers in the neural network you are actually interested in building, and you repeat this procedure for each layer in that other network step by step, you will end up with nice weights that you can set on that network that mangle the input data as little as possible when it is passed through the network. This "no-op default" is a much better starting point for then performing your actual training, relative to starting with random weights, and will greatly speed up your training.
# 
# Put another way, the hidden layer in an RBM learns a sparse representation of the data.
# 
# RBMs do not usually use the same optimization algorithms used by other neural network types (e.g. SGD, ADAM, etc.). Instead RBMs use a different algorithm, still within a gradient descent framework, known as "contrastive divergence". This algorithm is the result of a long search to find an algorithm that is both sufficiently fast and sufficiently accurate to make RBMs useful in practice; Geoff Hinton alleges that it took him 17 years to finally stumble across it.
# 
# The classical definition of an RBM is categorical. Target layer values must be either binary trails (0 or 1) or probabilities (in the range 0 to 1).

# ## Demo
# 
# We will demonstrate how to create and interpret an RBM using the classic MNIST dataset.
# 
# RBMs are not implemented in `keras`, because as mentioned in the lead they have fallen out of favor. You may implement them yourself using `caffe` or `tensorflow` or a similar low level fraomework (or copy one of a number of code snippets floating around online that do just that), or you can use the `scikit-learn` implementation.

# In[ ]:


import pandas as pd
import numpy as np

X_train = pd.read_csv('../input/train.csv').values[:,1:]  # exclude the target
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # rescale to (0, 1)


# Notice that the actual `sklearn` class for the RBM is very specifically called the `BernoulliRBM`.
# 
# Recall that the **Bernoulli distribution** is a simple distribution where a value may be 1 or 0. In the MNIST dataset each pixel is in the range (0, 255), but we'll simplify the data to be just 0 or 1. In that case we have $n \times n$ Bernoulli distributed data points to work with, where $n$ is the dimensionality of the image (and $n \times n$ is its overall pixel count).
# 
# There no RBM other than the `BernoulliRBM`. So the `sklearn` RBM can only used for binary trials or for probabilities, as mentioned in the lead.

# In[ ]:


from sklearn.neural_network import BernoulliRBM
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=42, verbose=True)
rbm.fit(X_train)


# To reconstruct data using the RBM, use the `gibbs` method.

# In[ ]:


def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)

xx = X_train[:40].copy()
for _ in range(1000):
    for n in range(40):
        xx[n] = rbm.gibbs(xx[n])
        
import matplotlib.pyplot as plt
plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(xx), cmap='gray')


# To see the list of components extracted by the RBm see `components_`.

# In[ ]:


plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.RdBu,
               interpolation='nearest', vmin=-2.5, vmax=2.5)
    plt.axis('off')


# This notebook copies code from [this one](https://www.kaggle.com/nicw102168/restricted-boltzmann-machine-rbm-on-mnist) by nic.
# 
# The work that the RBM does in finding a sparse representation of the original dataset actually falls into a general class of techniques known as dimensionality reduction techniques. The most famous reduction technique is PCA, which I demonstrate in the notebook ["Dimensionality Reduction and PCA for Fashion-MIST](https://www.kaggle.com/residentmario/dimensionality-reduction-and-pca-for-fashion-mnist/). RBMs are not very useful for dimensionality reduction because they are too slow.
