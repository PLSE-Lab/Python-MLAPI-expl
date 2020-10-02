#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading and Displaying the dataset

# In[ ]:


df = pd.read_csv("../input/train.csv")
df.head()


# In[ ]:


df.shape


# Creating vocabulary using Bag of Words techniques

# In[ ]:


def read_questions(row,column_name):
    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))
    
documents = []
for index, row in df.iterrows():
    documents.append(read_questions(row,"question1"))
    if row["is_duplicate"] == 0:
        documents.append(read_questions(row,"question2"))


# In[ ]:


print("List of lists. Let's confirm: ", type(documents), " of ", type(documents[0]))


# In[ ]:


model = gensim.models.Word2Vec(size=150, window=10, min_count=10, sg=1, workers=10)
model.build_vocab(documents)  # prepare the model vocabulary


# size: The size means the dimensionality of word vectors. It defines the number of tokens used to represent each word. For example, rake a look at the picture above. The size would be equal to 4 in this example. Each input word would be represented by 4 tokens: King, Queen, Women, Princess. Rule-of-thumb: If a dataset is small, then size should be small too. If a dataset is large, then size should be greater too. It's the question of tuning.
# 
# window: The maximum distance between the target word and its neighboring word. For example, let's take the phrase "agama is a reptile " with 4 words (suppose that we do not exclude the stop words). If window size is 2, then the vector of word "agama" is directly affected by the word "is" and "a". Rule-of-thumb: a smaller window should provide terms that are more related (of course, the exclusion of stop words should be considered).
# 
# min_count: Ignores all words with total frequency lower than this. For example, if the word frequency is extremally low, then this word might be considered as unimportant.
# 
# sg: Selects training algorithm: 1 for Skip-Gram; 0 for CBOW (Continuous Bag of Words).
# 
# workers: The number of worker threads used to train the model.

# In[ ]:


model.train(sentences=documents, total_examples=len(documents), epochs=model.iter)


# In[ ]:


w1 = "phone"
model.wv.most_similar(positive=w1, topn=5)


# In[ ]:


w1 = ["women","rights"]
model.wv.most_similar (positive=w1,topn=2)


# In[ ]:


model.wv.doesnt_match(["government","corruption","peace"])

