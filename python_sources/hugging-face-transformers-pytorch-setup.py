#!/usr/bin/env python
# coding: utf-8

# This kernel provides the required datasets and commands to setup Hugging Face Transformers setup in offline mode. You can find the required github codebases in the datasets.
# 1. [sacremoses dependency](https://github.com/alvations/sacremoses) - https://www.kaggle.com/axel81/sacremoses
# 2. [transformers](https://github.com/huggingface/transformers) - https://www.kaggle.com/axel81/transformers
# 
# 

# Sacremoses is a dependency for Hugging Face transformer. As we don't have internet access we have to install it manually.

# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master')


# ## Install transformers package

# In[ ]:


get_ipython().system('pip install ../input/transformers/transformers-master')


# ## Ready to use

# In[ ]:


import transformers


# Next Steps:
# 
# 1. Post a baseline using Hugging Face Transformers
# 2. Improve on baseline
