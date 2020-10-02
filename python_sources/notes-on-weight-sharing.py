#!/usr/bin/env python
# coding: utf-8

# # Notes on weight sharing
# 
# **Weight sharing** is an old-school technique for reducing the number of weights in a network that must be trained; it was leveraged by LeCunn-Net circa 1998. It is exactly what it sounds like: the reuse of weights on nodes that are close to one another in some way.
# 
# ## Weight sharing in CNNs
# A typical application of weight sharing is in convolutional neural networks. CNNs work by passing a filter over the image input. For the trivial example of a 4x4 image and a 2x2 filter with a stride size of 2, this would mean that the filter (which has four weights, one per pixel) is applied four times, making for 16 weights total. A typical application of weight sharing is to share the same weights across all four filters. 
# 
# In this context weight sharing has the following effects:
# * It reduces the number of weights that must be learned (from 16 to 4, in this case), which reduces model training time and cost.
# * It makes feature search insensitive to feature location in the image.
# 
# So we reduce training cost at the cost of model flexibility. Weight sharing is for all intents and purposes a form of regularization. And as with other forms of regularization, it can actually *increase* the performance of the model, in certain datasets with high feature location variance, by decreasing variance more than they increase bias (see ["Bias-variance tradeoff"](https://www.kaggle.com/residentmario/bias-variance-tradeoff)).
