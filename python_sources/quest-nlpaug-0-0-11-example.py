#!/usr/bin/env python
# coding: utf-8

# # Short Introduction
# Hi everybody, in the past few days, I played around with this great text augmentation repo : https://github.com/makcedward/nlpaug .
# I think that it has a potential, so I would love to share how to use it here. Please give the repo a star, and also please consider upvote the corresponding Kaggle dataset : https://www.kaggle.com/ratthachat/nlpaug0011
# 
# Note that we can use wordnet-based or glove-based word augmentation offline. But I still cound't find how to use bert-family-based offline yet. So I turn on the internet for this kernel.
# Note also that this is only an early development version, so we shall see more and more very nice features soon!
# 
# I am sorry I have not much time to write a good kernel. I will have to go for a family trip soon! Hope you all a happy long new year holliday!!!

# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')

import os
import sys
import glob
import torch

sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers
import numpy as np
import pandas as pd
import math


# In[ ]:


get_ipython().system('ls ../input/nlpaug0011/nlpaug-master')


# In[ ]:


get_ipython().system('pip install ../input/nlpaug0011/nlpaug-master #> /dev/null')


# In[ ]:


print(transformers.__version__)
print(torch.__version__)


# In[ ]:


import os
import re
import gc
import pickle  
import random
import tensorflow.keras as keras

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata, entropy
from os.path import join as path_join
from numpy.random import seed
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import MultiTaskElasticNet

from tqdm import tqdm_notebook


SEED = 42

seed(SEED)
tf.random.set_seed(SEED )
random.seed(SEED )


# # Test NLPAUG

# In[ ]:



import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

from nlpaug.util import Action


# In[ ]:


text = 'The quick brown fox jumps over the lazy dog .'
# !ls -a 


# In[ ]:


aug_syn = naw.SynonymAug(aug_src='wordnet')
# augmented_text = aug_syn.augment(text)
print("Original:")
print(text)
print("Augmented Synonym Text:")
for ii in range(5):
    augmented_text = aug_syn.augment(text)
    print(augmented_text)

aug = naw.AntonymAug()
_text = 'Good boy is very good'

print("Original:")
print(_text)
print("Augmented Antonym Text:")
for ii in range(5):
    augmented_text = aug.augment(_text)
    print(augmented_text)


# In[ ]:


get_ipython().system('ls ../input/nlpword2vecembeddingspretrained -l')


# In[ ]:


# model_type: word2vec, glove or fasttext
aug_w2v = naw.WordEmbsAug(
#     model_type='word2vec', model_path='../input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin',
    model_type='glove', model_path='../input/nlpword2vecembeddingspretrained/glove.6B.300d.txt',
    action="substitute")
print("Original:")
print(text)


# In[ ]:


aug_w2v.aug_p=0.1


# In[ ]:


print("Augmented Text:")
for ii in range(5):
    augmented_text = aug_w2v.augment(text)
    print(augmented_text)


# In[ ]:


#BERT Augmentator
TOPK=20 #default=100
ACT = 'insert' #"substitute"

aug_bert = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased', 
    #device='cuda',
    action=ACT, top_k=TOPK)
print("Original:")
print(text)
print("Augmented Text:")
for ii in range(5):
    augmented_text = aug_bert.augment(text)
    print(augmented_text)


# In[ ]:


aug = nas.ContextualWordEmbsForSentenceAug(
#     model_path='gpt2'
    model_path='xlnet-base-cased',
#     model_path='distilgpt2', 
    top_k=TOPK
)

print("Original:")
print(text)
print("Augmented Text:")
for ii in range(5):
    augmented_text = aug.augment(text)
    print(augmented_text)


# In[ ]:


# try Compose & Sometimes
# make offline augmentation + pseudo label from my best ensemble
# re-train

text = "I have a question about programming language. Which is the best between python and R?"
text = "What is your recommended book on Bayesian Statistics?"
# text = "How do you make a binary image in Photoshop?"
# text = "Can an affidavit be used in Beit Din?"

aug = naf.Sequential([
    aug_bert,aug_w2v
])

aug.augment(text, n=10)


# In[ ]:


aug2 = naf.Sometimes([
    aug_bert,aug_w2v
],aug_p=0.5, pipeline_p=0.5)

aug2.augment(text, n=10) # seems Sometimes has a bug, it still EveryTime, but results look better than sequential
# However, in the manual aug_p pipeline_p are not clearly defined (have to look at the source)


# # Try augmentation for training data

# In[ ]:


get_ipython().run_cell_magic('time', '', '# around 3-4/5-7 mins for Distil/BertBase [300-350words to 512subwords] respectively\ntrain = pd.read_csv("../input/google-quest-challenge/train.csv").fillna("none")\ntest = pd.read_csv("../input/google-quest-challenge/test.csv").fillna("none")\n\nsample = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")\ntarget_cols = list(sample.drop("qa_id", axis=1).columns)\ntargets = target_cols\ninput_columns = [\'question_title\', \'question_body\', \'answer\']')


# In[ ]:


texts = train['question_title'].values

for ii in tqdm_notebook(range(-7,-1)):
    print(texts[ii])
    print(aug.augment(texts[ii],n=1),'\n')


# In[ ]:





# In[ ]:




