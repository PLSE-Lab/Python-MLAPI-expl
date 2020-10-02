#!/usr/bin/env python
# coding: utf-8

# # COVID-19 papers Word2Vec model
# 
# Model built from the [CORD-19 research challenge](kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) using [this](https://www.kaggle.com/elsonidoq/train-a-word2vec) notebook.
# 
# You can get it on [here](https://drive.google.com/file/d/1SAsWsA2RgLFgJuwZ0kBLUYE5Xh1t7eF3/view?usp=sharing)

# In[ ]:


from gensim.models import Word2Vec

model = Word2Vec.load('/kaggle/input/covid19-challenge-trained-w2v-model/covid.w2v')


# In[ ]:


model.wv.most_similar('coronavirus', topn=20)


# In[ ]:


model.wv.most_similar('transmission', topn=20)


# In[ ]:




