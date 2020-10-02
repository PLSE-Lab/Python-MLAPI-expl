#!/usr/bin/env python
# coding: utf-8

# # Lib Import

# In[ ]:


import keras
print('Keras Version', keras.__version__)
print('Lib Import Completed!')


# # Keras-Bert Offline Import

# In[ ]:


get_ipython().run_line_magic('time', '')
get_ipython().system('ls ../input/keras-bert/keras-transformer-master/keras-transformer-master')
get_ipython().system('pip install ../input/keras-bert/keras-layer-normalization-master/keras-layer-normalization-master')
get_ipython().system('pip install ../input/keras-bert/keras-position-wise-feed-forward-master/keras-position-wise-feed-forward-master')
get_ipython().system('pip install ../input/keras-bert/keras-embed-sim-master/keras-embed-sim-master')
get_ipython().system('pip install ../input/keras-bert/keras-self-attention-master/keras-self-attention-master')
get_ipython().system('pip install ../input/keras-bert/keras-multi-head-master/keras-multi-head-master')
get_ipython().system('pip install ../input/keras-bert/keras-pos-embd-master/keras-pos-embd-master')
get_ipython().system('pip install ../input/keras-bert/keras-transformer-master/keras-transformer-master')
get_ipython().system('pip install ../input/keras-bert/keras-bert-master/keras-bert-master')
print('Lib Offline Import Completed!')


# Install completed! Enjoy it!
