#!/usr/bin/env python
# coding: utf-8

# ## **Import a few things**
# #### numpy is a numerical python library for doing fast matrix math in C instead of in python
# #### gensim is a NLP library with lots of useful stuff including a simple api for word2vec
# #### annoy is a library for super fast approximate nearest neighbors calculations on a memory mapped index

# In[ ]:


import numpy as np
from gensim.models import Word2Vec
from annoy import AnnoyIndex
from functools import reduce


# In[ ]:


import os


# In[ ]:


os.listdir('/kaggle/input')


# ## **Load in the user sessions**

# In[ ]:


with open('/kaggle/input/sessions', 'r') as f:
    sessions = [line[:-1].split(' ') for line in f]


# ## **Train the word2vec model on the session data**

# In[ ]:


model = Word2Vec(sessions, min_count=5, window=5, workers=8, size=100, iter=20)


# In[ ]:


# create a quick mapping from product to how many sessions it appeared in
# proxy for popularity 
i_to_num_sessions = {}
for session in sessions:
    for i in session:
        i_to_num_sessions[i] = i_to_num_sessions.get(i, 0) + 1


# ## **Load in product titles**

# In[ ]:


with open('/kaggle/input/id_to_title', 'r') as f:
    id_to_title = [line[:-1] for line in f]


# In[ ]:


# split a title by space and make lowercase
def tokenize(title):
    return title.lower().split(" ")


# ## **Let's create an inverted index so that we can find products quickly by their titles**
# ### Try searching for products by title by performing a scan of `id_to_title` and see how long it takes... I dare you

# In[ ]:


available_product_ids = set(model.wv.index2word)
inverted_index = {}

for i, title in enumerate(id_to_title):
    if i % 100000 == 0: print(i)
    if str(i) in available_product_ids:
        tokens = tokenize(title)
        for token in tokens:
            inverted_index[token] = inverted_index.get(token, set())
            inverted_index[token].add(i)


# ## **Define our methods for finding products by title using our inverted index and finding by the product vector**
# #### very sorry for the long lambda

# In[ ]:


findTitlesLike = lambda words: sorted(list(map(lambda i:(i, id_to_title[i]), list(reduce(lambda acc, word: acc.intersection(inverted_index.get(word, set())), words, inverted_index[words[0]])))), key=lambda x: -1 * i_to_num_sessions[str(x[0])])[:10] 

getSimilarProducts = lambda i: list(map(lambda p: (p, id_to_title[int(p[0])]), model.wv.similar_by_word(str(i))))


# In[ ]:


findTitlesLike(['macbook', 'pro', 'retina', '256gb'])[:5]


# In[ ]:


getSimilarProducts(497370)[:5]


# In[ ]:


findTitlesLike(['nintendo', 'switch', 'console'])[:5]


# In[ ]:


getSimilarProducts(813190)[:5]


# In[ ]:


findTitlesLike(['instant', 'pot'])


# In[ ]:


getSimilarProducts(925724)[:5]


# In[ ]:


findTitlesLike(['harry', 'potter', 'deathly', 'hallows'])[:5]


# In[ ]:


getSimilarProducts(102115)[:5]


# In[ ]:


findTitlesLike(['airpods'])[:5]


# In[ ]:


getSimilarProducts(322055)[:5] # airpods headphones


# In[ ]:


getSimilarProducts(152574)[:5] # airpods case


# ## **How fast can we perform this nearest neighbors calculation?**

# In[ ]:


import time


# In[ ]:


start = time.time()

for i in range(100):
    pid = model.wv.index2word[i]
    model.wv.similar_by_word(pid)
end = time.time()

print(end - start)


# ## **So being able to find similar products like this is cool, but relatively slow, especially as you scale to production sized data** Let's build our annoy index to see if we can speed that up

# In[ ]:


ai = AnnoyIndex(100)


# In[ ]:


for i, v in enumerate(model.wv.vectors):
    try:
        ai.add_item(int(model.wv.index2word[i]), v)
    except:
        print(model.wv.index2word[i])


# In[ ]:





# In[ ]:


ai.build(4)


# In[ ]:


findTitlesLike(['nintendo', 'switch', 'console'])[:5]


# In[ ]:


[(i, id_to_title[i]) for i in ai.get_nns_by_item(813190, 5)]


# In[ ]:


start = time.time()

for i in range(10, 45000):
    pid = model.wv.index2word[i]
    ai.get_nns_by_item(int(pid), 5)
end = time.time()

print(end - start)


# ## **100 vs 45,000 so much faster!!**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




