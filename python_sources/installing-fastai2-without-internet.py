#!/usr/bin/env python
# coding: utf-8

# This kernel will show you how to install fastai2 without internet on kaggle kernels, this will be helpful if you want to use fastai2 on kernel only competitions.
# **If you find this Kernel helpful don't forget to upvote this kernel and the [dataset](https://www.kaggle.com/vijayabhaskar96/fastai2-wheels)**

# Add this [dataset](https://www.kaggle.com/vijayabhaskar96/fastai2-wheels) and copy,paste and run the below cell in your kernel to install all the dependencies required to install fastai2(as of 25th June 2020)

# In[ ]:


get_ipython().system('pip install /kaggle/input/fastai2-wheels/fastcore-0.1.18-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/fastai2-wheels/fastai2-0.0.17-py3-none-any.whl')


# Importing fastai2 vision to test installation.

# In[ ]:


from fastai2.vision.all import *


# It works!

# In[ ]:


import fastai2
fastai2.__version__


# In[ ]:




