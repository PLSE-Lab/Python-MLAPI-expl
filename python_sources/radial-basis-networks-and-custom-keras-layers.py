#!/usr/bin/env python
# coding: utf-8

# # Radial Basis Networks and Custom Keras Layers
# 
# This notebook explores two incidentally related topics.
# 
# The first is **radial basis networks**. Radial basis networks are an uncommon subtype of neural network that uses **radial basis functions** as the activation function (instead of the logistic function or ReLU or other linear or nonlinear peers). RBNs can be used for functional approximation.
# 
# The second is custom `keras` networks. Incidentally, radial basis networks do not have their own `keras.layer` definition. So if you'd like to use an RBN you'll have to define the necessary layer yourself as a custom layer. Since RBF is a pretty simple layer architecture doing so will be a good crash course on doing that yourself.

# ## RBNs
# 
# Radial basis networks are fully-conneced feedforward neural networks that use a radial basis function as their activation on their hidden layers.
# 
# A radial basis function is any function which is defined as a function of distance from a certain central point (a radius). This is the property $\phi(x) = \phi(||x||)$. There are many function signatures which satisfy this property, but the most common is the Gaussian: $\phi(r) = e^{-(\epsilon r)^2}$.
# 
# RBFs can be used to approximate any other function by solving for the form $y(x) = \sum_{i=1}^N w_i \phi ( || x - x_i || )$. Since RBFs are defined as function based on a certain central point, this can be thought of as functional approximation through the summation of a sequence of functions which are accurate in some local neighborhood, inaccurate  elsewhere, but which add up to a close approximation of the root function. This is somewhat similar to [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation).
# 
# RBFs are attractive for functional approximation because they can be solved using a generalization of least squares regression. RBFs are the most powerful kernel available for [support vector machines](https://www.kaggle.com/residentmario/support-vector-machines-and-stoch-gradient-descent).
# 
# The neural network equivalent is RBNs.

# ## Custom trainable keras layers
# 
# To build your own keras layer you need toimplement a class that is a subclass of `keras.layers.Layer` which implements three functions (plus `__init__`): `build`, `call`, and .`compare_output_shape`.
# 
# ###  `build(self, input_shape)`
# 
# This is where you will define your weights. This method should end with calling `super([Layer], self).build()`.
# 
# To define weights use the `self.add_weight` method. `name`, `initializer`, and `trainable` control the name, initialization settings (e.g. random), and trainability (probably `True`) of the weight. The key parameter is `shape`, which defines the number and dimensionality of the weights matrix for this layer. It should probably be parameterized with the number of neural units to be included in the layer (e.g. 10), which is passed as `self.units`, and the shape of the input from the previous layer, which is passed as `input_shape`. Note that the zeroeth index of `input_shape` will be the batch size.
# 
# ### `call(x)`
# The logic for executing the layer should be `call(x)`. This accepts the input tensor and returns the output tensor. `Tensor` objects are how `keras` handles layer input and output under the hood. They have a certain number of defined dimensions (the hard input size) and a certain number of undefined dimensions (the batch size and any other flexible dimensionality). Tensors (and `Variable` objects) are a sort of asychronous promise that is not partially realized until compile time and not fully realized until run time.
# 
# The keras backend (`keras.backend`) contains arithmatic operations which operate on tensors, and the convention is to `from keras import backend as K` and use the methods from there. These ops are actually thin wrappers of the corresponding operations defined by the `keras` backend, e.g. `tensorflow` or `pytorch`.
# 
# Given that $x$ is a single input tensor (a vector, as we are only defining RBF for a flat record); $\mu$ is a learned weight vector (e.g. one weight per $x_i$ in $x$); and $\gamma$ is a user-specified tuning parameter, the formula for a single RBF neuron activation is:
# 
# $$\exp{(-\gamma \sum(x - \mu)^2)}$$
# 
# The `call` code demo'd below implements this formula.
# 
# ### `compute_output_shape(input_shape)`
# This method should specify the output shape for a given input shape. It is used during compile time to validate that the shapes of the tensors passed between layers is correct. This method is optional if your layer doesn't modify input shape (but almost all useful neural layers do).
# 
# ----
# 
# The code in the next section comes courtesy of [this StackOverflow answer](https://stackoverflow.com/questions/53855941/how-to-implement-rbf-activation-function-in-keras/53867101#53867101), and it shows the implementation in action:

# In[ ]:


from keras.layers import Layer
from keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# ## Custom non-trainable keras layers
# 
# It's worth noting, as an aside, that if you want to define a layer which doesn't contain any trainable weights, there is a simpler way to do that with `keras`: the `lambda` layer. See the [docs](https://keras.io/layers/core/).

# ## Quick demo
# 
# Here's a quick but pretty meaningless demo establishing that our layer works.

# In[ ]:


import pandas as pd
import numpy as np

X = np.load('../input/k49-train-imgs.npz')['arr_0']
y = np.load('../input/k49-train-labels.npz')['arr_0']
y = (y <= 25).astype(int)

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.losses import binary_crossentropy

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(RBFLayer(10, 0.5))
model.add(Dense(1, activation='sigmoid', name='foo'))

model.compile(optimizer='rmsprop', loss=binary_crossentropy)


# In[ ]:


model.fit(X, y, batch_size=256, epochs=3)


# ## Tuning RBNs
# 
# In the code sample above we chose to initialize the RBN with uniform random initialized weights (`initializer='uniform'`). This works because, given a sufficiently large number of nodes, the Gaussian RBFs we are using can approximate any decision boundary (as the sum of some large set of overlapping bell curves, since they are Gaussian). 
# 
# However recall that every RBF can be stated as a function of distance from a precise location in space, what is known as the "prototype". We can reduce the number of nodes necessary (and hence reduce computational time and increase model explainability) by choosing our RBF prototypes more wisely. Specifically, a good strategy is to initialize the network using node locations corresponding with [k-means](https://en.wikipedia.org/wiki/K-means_clustering) cluster centers. For example:
# 
# ![](https://i.imgur.com/3EhWItU.png)
# 
# This has the added benefit that we can pick a good $\gamma$ value more easily: set that parameter to be the average distance between points in a cluster.
# 
# ## Further reading
# 
# To browse the list of functions available for custom vector operations and layer definitions in `keras` see ["Keras backends"](https://keras.io/backend/) in the official docs.
# 
# To learn more about RBNs I highly recommend reading ["Radial Basis Function Network (RBFN) Tutorial"](http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/), which goes much further than this post in terms of explaining how RBNs work from first principles.
