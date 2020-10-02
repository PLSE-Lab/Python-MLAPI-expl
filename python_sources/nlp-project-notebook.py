#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


dow_path = '/kaggle/input/eng-680-class-project/DJIA_price.csv'
news_path = '/kaggle/input/eng-680-class-project/headline_news.csv'


# In[ ]:


df_dow = pd.read_csv(dow_path)
df_news = pd.read_csv(news_path)


# In[ ]:


df_dow.head()


# In[ ]:


df_news.head()


# In[ ]:




