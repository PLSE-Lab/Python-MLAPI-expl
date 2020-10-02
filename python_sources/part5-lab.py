#!/usr/bin/env python
# coding: utf-8

# ## TODO : make a submission on kaggle
# 
# 
# Hints to improve the score
# 
# - We were using only 50% of the dataset to train our NN
# - decrease the batch size (we were using 2048 for prototyping)
# - The NN architecture can be improved
# - We didnt use any regularization (L1/L2, dropout, SpatialDropout)
# - improve the text preprocessing
# - try other embeddings
# - try other finetuning approaches :
#    1. Setting a different LR per layer, https://erikbrorson.github.io/2018/04/30/Adam-with-learning-rate-multipliers/
#    2. ReduceOnPlateau schedule
# - increase the size of bagging
# - ensembe different architectures
# - During text preprocessing, some information is lost. Make additional numerical features as inputs for the NN. For example : number of uppercase characters, punctuations, bad words, etc
# - We have removed rare words from the vocabularym maybe performance will improve if we keep them
# 
# - read kernels from https://www.kaggle.com/c/quora-insincere-questions-classification/kernels and try other ideas

# ## To keep in mind
# - Learning rates may be quit different depending on the way you initialize your embeddings (random vs transfer learning)
# 
# - Always use transfer learning (cant't be worse than random initialization)
# 
# - when you find an optimal batch size, you will probably not need to tune it anymore
# 
# - Always do bagging to check for improvements (not only for NN)
# 
# - Sometimes simple things works better (Attention mechanism didnt work for me on this dataset)
# 
# - Dont be misleaded by the sequential approach I used for this lecture. Looking for the best performance is an iterative approach
# 
# 
# 
# 
# 

# In[ ]:




