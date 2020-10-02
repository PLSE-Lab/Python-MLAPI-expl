#!/usr/bin/env python
# coding: utf-8

# The Federalist papers were published by 3 people under the pseudoname PUBLIUS.
# There has been much work using plagarism detection techinques to determine their authorship.
# In this kernel we strip the supposed authorship of each one, build a model to predict the authorship
# of a paper and use the original annotations as an accuracy measure.
# 
# previous works:
# * https://github.com/mkmcc/FederalistPapers
# * https://github.com/matthewzhou/FederalistPapers
# * https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python
# * http://www.aicbt.com/authorship-attribution/

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


file = open('../input/The_federalist_papers.txt')


# In[ ]:


text = file.read()


# In[ ]:


text[:1000]


# In[ ]:


# TODO strip the author of each paper
# TODO cluster the papers to see if 3 clusters do show up
# TODO use DeepLearning to build generative model, https://gist.github.com/karpathy/d4dee566867f8291f086

