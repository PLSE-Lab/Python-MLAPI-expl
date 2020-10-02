#!/usr/bin/env python
# coding: utf-8

# This is a basic kernal for understanding of how pretrained Glove model is used
# in this we will find the similar words near to the input word .

# In[ ]:


import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

word2vec={}
embedding=[]
idx2word=[]
with open('/kaggle/input/glove6b50d/glove.6B.50d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
        


# there are 400000 words in this model and vector space of 50 dimension try len(embedding[0]).
# 

# also use word2vec.keys() to see words in glove. 

# V is Vocab size  and D is Dimension where our words are embedded.

# In[ ]:


D = 50
V = 400000


# Function for finding similarities

# In[ ]:




def similaraties(w ,n =5):

    if w not in word2vec:
        print('word not in our vocab')
        return
    
    wordvec = word2vec[w]
    distances = pairwise_distances(wordvec.reshape(1,D),embedding).reshape(V)
    idxes = distances.argsort()[1:n+1]
    
    for idx in idxes:
        print(idx2word[idx])
    
      


# In[ ]:


similaraties('ronaldo',n = 10)


# In[ ]:


similaraties('mom')


# In[ ]:


similaraties('1')


# In[ ]:


similaraties('january')


# **You can try on your own with this function and see the results 
# Pls upvote if this is helpful to you.**
# 
