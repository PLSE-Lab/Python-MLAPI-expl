#!/usr/bin/env python
# coding: utf-8

# In[ ]:


shakespeare_file_path = "/kaggle/input/shakespeare_complete_works.txt"


# In[ ]:


get_ipython().system('pip install gpt-2-finetuning==0.10')


# In[ ]:


# Available sizes: 124M, 355M, 774M
get_ipython().system('download_gpt2_model 355M')


# In[ ]:


MODEL = '355M'


# In[ ]:


import os
import tensorflow as tf
import numpy as np

from gpt_2_finetuning.interactive_conditional_samples import interact_model
from gpt_2_finetuning.train import train


# In[ ]:


train(dataset_path=shakespeare_file_path,
      model_name=MODEL,
      n_steps=10000,
      save_every=5000,
      sample_every=1000,
      mem_saving_gradients=True,
      print_loss_every=1000,
      max_checkpoints_to_keep=2)


# In[ ]:


get_ipython().system('rm -rf models')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls checkpoint/run1')


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


## Interact example
# interact_model(model_name=MODEL,
#                length=100,
#                top_k=40)


# In[ ]:


## Encode example
# from gpt_2_finetuning.load_dataset import load_dataset
# from gpt_2_finetuning.encoder import get_encoder

# enc = get_encoder(model_name)
# chunks = load_dataset(enc, shakespeare_file_path, combine=50000, encoding='utf-8')
# enc.encode("PyCon is awesome")
# enc.decode([20519])

