#!/usr/bin/env python
# coding: utf-8

# I trained some word embeddings on kaggle forum posts. It's a pretty fun tool for searching up related techniques people are using. You can use it to pick what to work on or explore next. 

# In[ ]:


from gensim.models import KeyedVectors


# In[ ]:


w2v = KeyedVectors.load_word2vec_format("../input/kaggle-tuned-word2vec/kaggleword2vec.bin", binary = True)


# In[ ]:


w2v.most_similar("unet")


# In[ ]:


w2v.most_similar("augmentation")


# In[ ]:


w2v.most_similar("image")


# In[ ]:


w2v.most_similar("segmentation")


# In[ ]:


w2v.most_similar("pneumothorax")


# In[ ]:


w2v.most_similar("xray")


# In[ ]:


w2v.most_similar("cnn")


# We can see that the class woman - man + king still holds true.

# In[ ]:


w2v.most_similar(positive=['woman', 'king'], negative=['man'])


# And we can also do things like cnn - convolution + rnn = lstm

# In[ ]:


w2v.most_similar(positive=['cnn', 'rnn'], negative=['convolution'])


# In[ ]:




