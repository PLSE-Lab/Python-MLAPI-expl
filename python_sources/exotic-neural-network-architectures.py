#!/usr/bin/env python
# coding: utf-8

# # Exotic neural network architectures
# 
# Some neural network architectures are so common so as to be ubiquitious, and likely to be familiar to any machine learning practitioner with some deep learning experience. Feedforward neural networks. Convolutionanal neural networks. Recurrant neural networks. Just some examples.
# 
# Other neural network architectures or ideas are less common, or specialized to specific problem domains. Sequence-to-sequence models. Transformers.
# 
# Still other neural network architectures are historically important, or still novelties. Autoencoders. Deep belief networks.
# 
# A great post that provides an outline of the ecosystem of neural networks that I find myself coming back to again and again is the following Towards Data Science article: ["The mostly complete chart of neural networks explained"](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464). That article contains the following outline image, which is almost worth framing and putting on your wall:
# 
# ![](https://i.imgur.com/ygLqyOI.png)
# 
# Over the course of my explorations and note-taking on neural networks on Kaggle, I have covered many of these architectures in detail in specific dedicated notebooks:
# 
# * [Perceptrons](https://www.kaggle.com/residentmario/implementing-a-perceptron/)
# * [Feedforward neural networks](https://www.kaggle.com/residentmario/implementing-a-feedforward-neural-network)
# * [Restricted boltzmann machines](https://www.kaggle.com/residentmario/restricted-boltzmann-machines-and-pretraining)
# * [Autoencoders](https://www.kaggle.com/residentmario/autoencoders)
# * [Radial basis networks](https://www.kaggle.com/residentmario/radial-basis-networks-and-custom-keras-layers/)
# * [Convolutional neural networks](https://www.kaggle.com/residentmario/notes-on-convolutional-neural-networks)
# * [Recurrant neural networks](https://www.kaggle.com/residentmario/notes-on-recurrent-neural-networks)
# * [Generative adverserial networks](https://www.kaggle.com/residentmario/notes-on-gans)
# * [Deep belief networks](https://www.kaggle.com/residentmario/notes-on-deep-belief-networks)
# * [Variational autoencoders](https://www.kaggle.com/residentmario/variational-autoencoders)
# 
# In this notebook, I will survey the remainder of the more "exotic" neural network architectures from this blog post not already covered in my specific notes above.

# ## Markov chains
# 
# Markov chains, it turns out, can be implemented as neural networks. Fully connected neural network nodes that are not layer-oriented but instead cyclic. Cool.
# 
# ## Hopfield network
# 
# A Markov chain trained via backpropogation (with depth). This architecture was proposed in the 1970s and 1980s as a computational model for human memory, but is not directly applicable to machine learning problems.
# 
# ## Boltzmann machines
# 
# Hopfield networks which are trained using Gibbs sampling instead of backpropogation. Boltzmann machines are restricted Boltzmann machines, which have historical applications in pre-training regimens, but arranged in a fully connected graph instead of in a two-layer feedforward network. Boltzmann machines have not proved useful for machine learning.
# 
# ## Deep convolutional inverse graphics network (DCIGN)
# 
# DCIGN is a very interesting domain-specific application of neural networks. A DCIGN takes an image input, compresses it to an information vector, and then deconvolves that vector into an image output. What makes a DCIGN special is that it uses a well-specified graphics code as its encoding format: a vector containing components for shape, lighting, pose, etecetera that comes from the graphics processing world. The use of this pre-specified format results in information vectors that are highly **disentangled**, e.g. varying an attribute of interest of the image varies just one or a few components of the image at a time.
# 
# A DCIGN thus has the capacity to ingest and input image and output a modified version of that image as its output. For example, one might vary the location of lightning on a face, or the direction that a chair is facing, or the pose of a human in a photograph.
# 
# ## Liquid state machine
# 
# A liquid state machine is a modification of classic neural network architecture which has been put forth as an explanation for how the brain works.
# 
# First of all, in a liquid state machine, all nodes except for the output layer nodes are what are known as **spiking neural nodes**. Spiking neural nodes differ from non-spiking nodes in that they incorporate time delay: a spiking node has a limit on how often it can fire. This mimics the way that neurons work in the brain.
# 
# The hidden layer in a liquid state machine consists of what is called a "soup" of spiking neural nodes which connected to one another (and through the soup, to the output layer) in a random recurrent manner. Given a large enough number of connections, a liquid state machine can theoretically approximate the necessary mathematical function for performing any tasks of interest.
# 
# ## Echo state network
# 
# An echo state network is very similar to a liquid state machine: it differs only in that it has a soup of sparsely randomly connected recurrant neural network nodes, instead of spiking neural network nodes.
# 
# Like the liquid state machine, it has no practical applications at this time.
# 
# ## Extreme learning machine
# 
# Extreme learning machines are a feedforward network architecture with randomly sparsified feedfoward connections. This is more computationally efficient than having a densely connected feedforward layer, but the actual efficacy of the learning seems highly task dependent.
# 
# ## Kohenon network
# 
# A Kohenon network is a self-organizing map. Subject of a future notebook.
# 
# ## Neural turing machine
# 
# A feedforward network architecture which features memory cells, connected to the last hidden layer of the network, which the network may read from and write to. This architecture was invented in 2014, and drew some interest, but the original authors never published their source code, and it wasn't until 2018 that someone finally wrote a stable implementation (previous attempts at an implementation suffered from numerical stability problems in the gradient).
