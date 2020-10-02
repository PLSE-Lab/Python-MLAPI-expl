#!/usr/bin/env python
# coding: utf-8

# # **How to train your <del>dragon</del> custom word embeddings**
# 
# In the [baseline-keras-lstm-is-all-you-need](https://www.kaggle.com/huikang/baseline-keras-lstm-is-all-you-need) notebook shared by Hui Kang (thanks again!), it was demonstrated that a LSTM model using generic global vector (GLOVE) achieved a pretty solid benchmark results.
# 
# After playing around with GLOVE, you will quickly find that certain words in your training data are not present in its vocab. These are typically replaced with same-shape zero vector, which essentially means you are 'sacrificing' the word as your input feature, which can potentially be important for correct prediction. Another way to deal with this is to train your own word embeddings, using your training data, so that the semantic relationship of your own training corpus can be better represented.
# 
# In this notebook, I will demonstrate how to train your custom word2vec using Gensim.
# 
# For those who are new to word embeddings and would like to find out more, you can check out the following articles:
# 1. [Introduction to Word Embedding and Word2Vec](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)
# 2. [A Beginner's Guide to Word2Vec and Neural Word Embeddings](https://skymind.ai/wiki/word2vec)

# In[ ]:


import numpy as np
import pandas as pd
import os
import re
import time

from gensim.models import Word2Vec
from tqdm import tqdm

tqdm.pandas()


# In[ ]:


def preprocessing(titles_array):
    
    """
    Take in an array of titles, and return the processed titles.
    
    (e.g. input: 'i am a boy', output - 'am boy')  -> since I remove those words with length 1
    
    Feel free to change the preprocessing steps and see how it affects the modelling results!
    """
    
    processed_array = []
    
    for title in tqdm(titles_array):
        
        # remove other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces).
        processed = re.sub('[^a-zA-Z ]', '', title)
        
        words = processed.split()
        
        # keep words that have length of more than 1 (e.g. gb, bb), remove those with length 1.
        processed_array.append(' '.join([word for word in words if len(word) > 1]))
    
    return processed_array


# ## **Something to take note**
# Word2vec is a **self-supervised** method (well, sort of unsupervised but not unsupervised, since it provides its own labels. check out this [Quora](https://www.quora.com/Is-Word2vec-a-supervised-unsupervised-learning-algorithm) thread for a more detailed explanation), so we can make full use of the entire dataset (including test data) to obtain a more wholesome word embedding representation.

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train['processed'] = preprocessing(df_train['title'])
df_test['processed'] = preprocessing(df_test['title'])

sentences = pd.concat([df_train['processed'], df_test['processed']],axis=0)
train_sentences = list(sentences.progress_apply(str.split).values)


# In[ ]:


# Parameters reference : https://www.quora.com/How-do-I-determine-Word2Vec-parameters
# Feel free to customise your own embedding

start_time = time.time()

model = Word2Vec(sentences=train_sentences, 
                 sg=1, 
                 size=100,  
                 workers=4)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')


# ## **Pretty fast isn't it.**
# 
# Let's check out some of the features of the customised word vector.

# In[ ]:


# Total number of vocab in our custom word embedding

len(model.wv.vocab.keys())


# In[ ]:


# Check out the dimension of each word (we set it to 100 in the above training step)

model.wv.vector_size


# In[ ]:


# Check out how 'iphone' is represented (an array of 100 numbers)

model.wv.get_vector('iphone')


# ## Now, why are word embeddings powerful? 
# 
# This is because they capture the semantics relationships between words. In other words, words with similar meanings should appear near each other in the vector space of our custom embeddings.
# 
# Lets check out an example:

# In[ ]:


# Find words with similar meaning to 'iphone'

model.wv.most_similar('iphone')


# Well, you will see words similar to 'iphone', sorted based on euclidean distance.
# Of cause, there are also not so intuitive and relevant ones (e.g. jetblack, cpo, ten). If you would like to tackle this, you can do a more thorough pre-processing/ try other embedding dimensions
# 

# ## **The most important part!**
# Last but not least, save your word embeddings, so that you can used it for modelling. You can load the text file next time using Gensim KeyedVector function.

# In[ ]:


model.wv.save_word2vec_format('custom_glove_100d.txt')


# How to load:
# w2v = KeyedVectors.load_word2vec_format('custom_glove_100d.txt')

# How to get vector using loaded model
# w2v.get_vector('iphone')


# In[ ]:




