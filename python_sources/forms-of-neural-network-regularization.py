#!/usr/bin/env python
# coding: utf-8

# # Forms of neural network regularization
# 
# Neural networks are intrinsically highly complex, and so have many forms of regularization, which are often used in conjunction.
# 
# ## Dropout
# Dropout is the removal of a random subset of nodes and their connections in a layer, with the nodes that get removed changing from batch to batch. Dropout induces regularization in the network by forcing the nodes of the network to spread responsibility for important features out across multiple nodes, creating featurization redundancies. This reduces the expertise of the model, but increases its robustness.
# 
# The mechanism by which dropout works and its effect on the network is clear and obvious, part of the reason why it's such a popular choice.
# 
# As a rule of thumb, a dropout rate of 0.2 is often used for input layers, and a dropout rate of 0.4 or 0.5 is often used for intermediate layers.
# 
# As a rule of thumb, dropout is a percentage reduction in network expertise. For example, a 200-node layer with 50% dropout is comparable to a 100-node layer with no dropout.
# 
# ## Weight decay
# Weight decay is a reformulation of the L2 penalty for neural network training tasks. L2 penalty is a well-known and studied form of regularization popular in classical regression contexts which underlies e.g. [ridge regression](https://www.kaggle.com/residentmario/ridge-regression-with-video-game-sales-prediction). Adding an L2 penalty to the cost function for a model has the effect of shrinking weights, particularly those for non-informative features, towards zero.
# 
# The L1 penalty is also often used in regression (this is lasso regression), and has the advantage of inducing true sparsity (setting weights all the way to zero), but it's non-differenciable, so not a fast option for nueral networks. To induce sparse weights in your model, use the ReLU activation function instead.
# 
# Another practical way of thinking about the effect of weight decay on model training: before each training run, every weight in the model is multiplied by a number slightly less than 1 ([link](https://bbabenko.github.io/weight-decay/)). This is "friction" in the training process, and the model must constantly fight the desire of the regularization term to decrease weight values as part of the training process.
# 
# ## Batch normalization
# Batch normalization is a pre-processing step applied to the data input to a nueral network layer that performs regularization. A layer with batch normalization will independently scale each component of the input batch vectors to zero mean and unit variance. For example, if input is in the form `[<1, 2, 3>, <4, 5, 6>, <7, 8, 9>]`, the three vectors will be sliced component-wise to `[<1, 4, 7>, <2, 5, 8>, <3, 6, 9>]`, and each of these components will be normalized independently.
# 
# Neural networks incorporating batch normalization in their architecture have been found in practice to train as much as twice as fast as neural networks without them, using a larger learning rate that would cause a non-batch-normalized network to diverge.
# 
# Although the effect of batch normalization has been clear in practice, the theoretical reason why it works is still inconcrete. It was originally posited that batch normalization reduced a property known as "covariate shift". Further research has shown that this is neither true nor important in practice. Instead, batch normalization seems to work because it results in a smoother cost surface.
# 
# Intuitively, small changes in weights early in a deep network can cause dramatic shifts in the data inputted to later network layers. Batch normalization brings these vector inputs back to well-known and numerically stable distribution properties, which smooths out the cost surface when the learning rate (and hence the resulting weight changes) is large.
# 
# ## Data augmentation
# Data augmentation is the process of generating and including in training transformed versions of the input training data. Examples of data augmentation include matrix transforms (flipped, sheared, clipped, whitened, etecetera) for image data, or differently-sampled data for featurized data.
# 
# Data augmentation can be used to increase the size of the training set, allowing for the training of a deeper, better-generalizing model. It also makes the model more robust to the forms of transformations included in the training set.
# 
# However, since the image inputs to the model become (to a certain extent) merely transforms of one another, data augmentation also has the secondary effect of inducing bias in the network, resulting in regularization. The input data should be augmented so as to allow for some regularization, but not so much that the model begins to fail to generalize.
# 
# ## Weight sharing
# Yet another historically popular form of regularization is weight sharing. Weight sharing occurs whenever two different weights are constrained to have the same value, such that an update to one weight results in a simultaneous update to the other weight.
# 
# Weight sharing reduces the number of weights that need to be updated. This reduces model expertise, but also model training time. An example application of weight sharing comes from the classical LeCunnNet, which used shared weights on the convolutional layer of the model. For example, a given 3x3 filter would have the same weights in every pixel of input data. This creates strong location invariance regularization, but has the principal effect and purpose) of *exponentially* reducing the number of weights which need to be trained.
# 
# ## Rectified linear activation
# Rectified linear activation is the primary methodology for inducing sparsity in a neural network model. Sparsity is the property that unimportant features in the model are given a weight of zero, effectively "turning off" that part of the model. ReLU activation achieves sparsity by trapping any weights which are updated below that value zero to zero. Once a node goes to or below zero, it is effectively "turned off" and no longer contributes to the model.
# 
# So long as optimal nodes kept on during the early and intermediate parts of the training process, ReLU activation will result in a sparse, computationally efficient, easier-to-train model. Along with its separate effect of making exploding or vanishing gradients less likely, ReLU allows for the training of deeper models more quickly than would be otherwise possible. As a result, ReLU is the dominant activation function in model neural network architectures.
# 
# ## Conclusion
# Batch normalization should always be used. Dropout should always be used, but the precise amount to use is subject to tuning. ReLU should always be used, but you may try a different activation function later in the hyperparameter optimization game. Data augmentation should always be used, but the right transformations to use requires a lot of tuning and knowledge about dataset.
# 
# If all other forms of regularization have already been applied to the model, but you suspect more is needed, weight decay can be added to the model.
# 
# Weight sharing is an uncommon form of regularization in modern practice, it is too heavy-handed and introduces too much bias.
