#!/usr/bin/env python
# coding: utf-8

# #A quick Overview
# 

# This kernel is an exploratory analysis of the dataset you can fork and use as a starting point for other kernels. It also contains some words cloud of the main sources contained in the dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/wordgame_20170628.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# The dataset consists of 334036 rows with 4 columns:
# 
#  1. **author**: is an ID of the author of an article from which the two words have been taken;
#  2. **word1**: is the first word;
#  3. **word2**: is the second word;
#  4. **source**: is the source (i.e. magazine) from which the article containing the two words have been taken.

# In[ ]:


data['author'].unique().shape


# Here follows the authors top 10

# In[ ]:


data['author'].value_counts().head(10)


# there are 10 different sources

# In[ ]:


data['source'].unique()


# and here is how the articles re distributed among the sources

# In[ ]:


data['source'].value_counts()


# In[ ]:





# In[ ]:


import numpy as np # linear algebra
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS


# ##The Wrong Planet Words Cloud

# In[ ]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data[data['source']=='wrongplanet']['word1']))


# In[ ]:


fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ##The Gog Words Planet

# In[ ]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data[data['source']=='gog']['word1']))


# In[ ]:


fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ## The sas Words Planet

# In[ ]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data[data['source']=='sas']['word1']))


# In[ ]:


fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:




