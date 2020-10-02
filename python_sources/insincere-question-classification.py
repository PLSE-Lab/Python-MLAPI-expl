#!/usr/bin/env python
# coding: utf-8

# Ref: https://www.youtube.com/watch?v=mmUttpXu7oE

# # Theory
# data imbalancing handling - 
# * generating fake data with GANS
# * weighted sampling
# 
# We using model-roBERTa
# takes text->word vector(768 D)(represent meaning and context)
# 
# How RoBERTa works-
# Transformers - Self attention: 
# My dog is lazy and it crossed the street(sentence) 
# How "it" refers to "dog"? -- attention
# def: takes a bunch of words, understand context behind each words and makes sense about surrounding, massive deep neural network, with bunch of blocks in it, blocks are built around self attention, blocks are ENCODERS
# 
# BERT - a transformer
# most language model looks from left to right or right to left
# In BERT-Bidirectional Encoder Representation of Transformers
# "My name is Manami" 
# looks at all these words at the same time 
# * Huge corpus of text
# * took 15% of the words and masked them
# * based on other words, predicted the masked word - MLM
# * NSP - next sentence prediction
#     * gave sentences
#     * given- pair of 2 senences-second sentence directly follows the first(50% of the time)
#     * model predicts 0/1 (linked/notlinked)
#     * My name is Manami. I have a dog. His name is Blue.(2nd and 3rd sentences are linked)
# * BERT - trained on NSP, MLM
# * Fine tuning
# 
# RoBERTa-
# * direclty based on BERT
# * NSP was removed
# * only trained on MLM
# * trained on much more data than BERT
#     

# # Modeling pipeline
# Question text -> RoBERTa *o/p vector(meaning of the question)* ->Fully connected Dense Layer(1 neuron) ->sigmoid layer(prob 0-1)

# **Set up PyTorch-XLA**
# Pytorch but on TPU
# * These few lines of code sets up PyTorch XLA for us.
# * We need PyTorch XLA to help us train PyTorch models on TPU.

# Set the pytorch XLA TPU

# In[ ]:





# In[ ]:




