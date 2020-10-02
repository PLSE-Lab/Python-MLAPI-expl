#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

get_ipython().system('git clone https://github.com/bkkaggle/grover.git')
os.chdir('grover/')
get_ipython().system('pip install -r requirements-gpu.txt')
get_ipython().system('python download_model.py base')


# In[ ]:


get_ipython().system('python generate.py --title="Why Bitcoin is a great investment" --author="Paul Krugman" --date="08-31-2019" --domain="nytimes.com"')


# In[ ]:


os.chdir('../')
get_ipython().system('rm -rf grover')

