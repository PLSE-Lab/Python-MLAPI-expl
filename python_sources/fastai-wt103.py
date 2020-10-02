#!/usr/bin/env python
# coding: utf-8

# This notebook is to download the latest Fastai wt103 models, pre-trained on wiki English corpus.
# See https://docs.fast.ai/text.html

# In[ ]:


from fastai.text import *


# In[ ]:


untar_data(URLs.WT103_FWD , data=False, dest=".")


# In[ ]:


get_ipython().system('ls')


# In[ ]:


untar_data(URLs.WT103_BWD , data=False, dest=".")


# In[ ]:




