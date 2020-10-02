#!/usr/bin/env python
# coding: utf-8

# # Notes on model weights initializers
# 
# The **weight initializer** is the component that sets the initial weights for a neural network. If you are using a pretrained network, the initial weights will be set to the values used in that network; otherwise, you will have to initialize those weights yourself somehow. Since a better choice of weights translates into less training time and (perhaps) a better model, the choice of initializer is a consequential one for model training.
# 
# We will use the model weights initializers available in Keras for this notebook. These are documented [here](https://keras.io/initializers/)
# 
# ## Zeros, Ones, Constant, Identity
# These do exactly what it sounds like they will do.
# 
# Zeros is the appropriate initializer for bias weights. Otherwise, these weights are for demonstratory purposes only, and should not be used in practical applications.
# 
# 
# ## RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling
# `TruncatedNormal` is `RandomNormal` with values more than two standard deviations (99%) from the mean discarded and redrawn. `TruncatedNormal` is the default "unfancy" initializer for neural networks whose optimal weights you have no insight into. `VarianceScaling` is a fancier version of `TruncatedNormal` that does some additional stuff.
# 
# Random normal initialization was historically the weights initializer of choice for most applications, up until the ResNet paper in 2015 (see He initialization below). It is still an appropriate choice of weight initialization for sigmoid layers.
# 
# ## Orthogonal
# `Orthogonal` matrices are those which have the property that their eigenvalue (the sum of their eigenvector) is 1. Orthogonal matrices have various mathematical properties, the important of which (in a neural network training context) is the fact that they are stable when taken to a power. As a result, gradient updates that are performed on an orthogonal (or near-orthogonal) matrix of weights will tend to be more robust to exploding or vanishing gradients. There's a terrific blog post on the subject that outlines the mathematics and visualizes the result here: ["Explaining and illustrating orthogonal initialization for recurrent neural networks"](https://smerity.com/articles/2016/orthogonal_init.html).
# 
# Orthogonal initializers are a useful alternative to LeCun initialization. It is occassionally but rarely used in practice.
# 
# ## lecun_uniform, lecun_normal
# The LeCun uniform initializer draws from a uniform distribution within `[-limit, limit]` where limit is `sqrt(3 / fan_in)` where `fan_in` is the number of input units in the weight tensor. This distribution was found to work well emperically "in practice", and it is the rule of thumb that was applied to the 1998 LeCun-Net.
# 
# Note that both normal and uniform samplers are provided. All of the remaining initializers have both of these options available. There is no consensus on which paradigm is better (see [this SO post](https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are) for a reference on the discussion on this matter).
# 
# This initialization is rarely used in practice today.
# 
# ## glorot_normal, glorot_uniform
# 
# The `glorot_*` initializers are the state-of-the-art initializer choice for neural layers using logistic activation or similar. They are designed to keep the weights stable under multiplication, just as with orthogonal activation, but are tuned to do so using these activations. It is the subject of a 2011 paper.
# 
# A readable summary of the mathematical mechanisms by which Glorot and He initialize achieves these goals is given in this blog post: ["Weight Initialization in Neural Networks: A Journey From the Basics to Kaiming"](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79). Some additional mathematical justification in the CS231n (Stanford) course notes [here](http://cs231n.github.io/neural-networks-2/#init).
# 
# ## he_normal, he_uniform
# The `he_*` initializers are the state-of-the-art initializer choce for neural layers using ReLU activation.
# 
# ResNet introduced ReLU activation, which, by its sparsity-inducing nature, immediately and permanently turns off half of the neural network units in a layer initialized with random normal (all those for which the normal value falls below 2). He initialization (Kaiming initialization in PyTorch corrects for this essentially by multiplying the Glorot factor by two).
# 
# ## Current state of the art
# The current state of the art is to use Glorot (non-ReLU) or He (ReLU) initialization alongside batch normalization.
