#!/usr/bin/env python
# coding: utf-8

# ## Notes on convolutional neural networks
# 
# ### Background
# Convolutional neural networks are one of the major neural network archetypes. They're a network architecture that is designed to work well for image datasets. CNNs are inspired by the biological principles behind how the human retina works. In the retina, each neural bundle is responsible for a single overlapping patch of the overall image.
# 
# Image processing tasks have two important properties, which greatly inform how best to go about solving them. The first is that images are, from an input perspective, huge; even a 100x100 image would feed 10,000 individual weights to each node in a fully connected network. The first is that locality matters. For any given feature, pixels that are near that feature's central position in an image are important, and pixels which are far away are very unimportant. Thus a fully connected network is not an appropriate architecture for image tasks, because a fully connected network will be both computational intractable and high wasteful.
# 
# CNNs address this problem by decreasing the number of weights that need to be learned for each node in a smart, locality-aware way. CNNs have the important fundamental proprety that they are location invariant; a feature showing up in one location in an image can be detected no better and no worse than that same feature showing up anywhere else in the image.
# 
# ### How they work
# The principal action of a CNN is **convolution**. Convolutions (really cross-correlations, mathematically) first define a feature a few pixels in size. Then, they try to find the segment of the image which mostly closely matches that feature. A rolling window is applied to the image, and a summary statistic of "fitness" is generated for each selection in that window. The winning window is the one which is the closest match. The result of the application of a convolution for a particular feature to an image is known as a **feature map**, and it looks like a matrix of fit statistics.
# 
# Another important action is **pooling**. Pooling divides the fit statistics in the feature map and computes a summary statistic on them (max is usually the default choice, but average and other fancier things are also possible). A good pooling layer will reduce the complexity .of the feature map while compressing it significantly, making it significantly more computationally tractable.
# 
# Finally, CNNs use **ReLU** layers to induce sparsity. I covered ReLU in [neural network activation functions](https://www.kaggle.com/residentmario/neural-network-activation-functions); it's an activation function which "deadens" nodes which return sum-zero or sum-negative outputs by scaling the value they emit to the next layer to zero. A ReLU layer is used to turn the non-matches in the feature layer into no-ops for the next layer down the line.
# 
# CNNs work by stacking layers in a Convolution > ReLU > Pool layering. The first few stacks of layers learn feature maps for very small chunks of the images, typically just a few pixels in size. The relevant feature maps are preserved by a ReLU layer, and the irrelevant maps are disabled. Then a pool upvets the feature maps to larger sizes (e.g. a dozen pixels), and those feature maps get parsed by a ReLU, repooled, and so on down the line.
# 
# We thus generate a vertical stack of progressively more complex feature maps, and the patterns that appear at various sizes in the image corpus (e.g. bright spots at the lowest size, versus noses at the moderate size, versus whole faces at the highest size) are, in a fully-trained network, voted upon by nodes at the most similarly-sized computational layer of the network.
# 
# Finally, at the end of the image there is a fully connected layer that pools votes from each of the high-level detectors on whatever the network is trying to discern. Fully connected layers can be stacked to allow the network to vote on deeper feature combinations, which may further improve decision-making.
# 
# ### Tuning
# CNN architecture is highly adjustable. Questions include:
# 
# * For each convolution layer, how many features?
# * For each pooling layer, what window size?
# * For each fully connected layer, how many neurons?
# 
# In the general case there are "well-known" networks that can be recommended for particular combinations of problem design and training resource constraints.
# 
# ### Non-image applications
# CNNs can be applied to any dataset which can be _meaningfully_ reordered into an "image-like" format. A columnar dataset where the columns are mostly independent of one another would perform poorly, whilst a columnar dataset whose columns are dependent on one another in an interesting way might work. However, in practice other architectures (like RNNs) work better for these other types of datasets.
# 
# ### Further reading
# https://brohrer.github.io/how_convolutional_neural_networks_work.html
