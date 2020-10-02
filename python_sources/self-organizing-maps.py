#!/usr/bin/env python
# coding: utf-8

# # Self-organizing maps
# 
# These notes based on the [Wikipedia article](https://en.wikipedia.org/wiki/Self-organizing_map) on self-organizing maps as well as [this slide deck](http://www.cs.bham.ac.uk/~jxb/NN/l16.pdf) explaining their mathematics in good detail.
# 
# **Self-organizing maps** are a simple technique for performing unsupervised dimensionality reduction using neural network nodes. They can thought of as a non-parametric competitor to classical dimensionality reduction techniques like [principal component analysis](https://www.kaggle.com/residentmario/dimensionality-reduction-and-pca-for-fashion-mnist/) or [linear discriminant analysis](https://www.kaggle.com/residentmario/linear-discriminant-analysis-with-pokemon-stats).
# 
# Self-organizing maps are in use in practical applications, but due tend to find less use than the classical parametric methods (which are much faster), or newer methods like autoencoders (which have themselves largely fallen out of favor) or t-SNE (which is what most people use nowadays). But they're still an interesting algorithm that's worth knowing about.
# 
# ## How they work
# 
# Self-organizing maps have been around since the 1980s, and are very neurobiologcally inspired. In a self-organizing map, the input is connected to every node, and the nodes form a multidimensional lattice over a space (typically a two-dimensional lattice). Nodes are competitive: input is fed one at a time, and the node that has the strongest reaction to the input is declared the "winning node" and gets to assign its weights to the input and return that as output. This is known as a **topographical map**.
# 
# Node winningness is determined using a discriminant function. The discriminant function of choice is simple Euclidian distance between the node weights and the input vector:
# 
# $$d_j(x) = \sum_{i=1}^D (x_i - w_ji)^2$$
# 
# Training is performed using topological ordering. A training sample is presented to the network, which then picks the node whose weights are closest to that input. The node shifts its weights towards the input value by a certain learning rate. The neighborhood of nodes closest to that node *also* shift their location, less strongly than the winning node does, with the strength of the movement dependent on that node's distance to the winning node, subject to a learning rate and to exponential decay. The learning rate is annealed over time.
# 
# Each update can be thought of as a shift in the topology of a neighborhood local to the output point. With sufficiently many iterations through this process, the nodes will form the dimension-reduced feature space over the original input which best preserves local topology:
# 
# ![](https://i.imgur.com/0B6ySvk.png)
# 
# ## Implementations
# 
# There is no `scikit-learn` implementation of self-organizing maps. It has been contributed to `scikit-learn` multiple times, but every discussion has ended with the `scikit-learn` team not feeling like they were sufficiently heavily in use to be worth including in the mainline package.
# 
# The package of choice for using self-organizing maps in Python is `minisom`: https://github.com/JustGlowing/minisom.
