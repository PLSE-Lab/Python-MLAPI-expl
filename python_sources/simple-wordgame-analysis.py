#!/usr/bin/env python
# coding: utf-8

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


data.info()


# In[ ]:


data.author.value_counts()


# In[ ]:


data.author.value_counts()[:10].plot.barh(figsize=(10,10))#top 10 author id


# In[ ]:


data.source.value_counts()


# **simple wordclouds :**

# **WORDCLOUD OF GAG SOUECE :**

# In[ ]:


# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud, STOPWORDS

text = (str(data[data['source']=='gog']['word1']))
wordclou = WordCloud(
                      relative_scaling = 1.0,
                      stopwords = 'to of'
                      ).generate(text)
plt.imshow(wordclou)
plt.axis("off")
plt.title('GOG WORD1')
plt.show()

text = (str(data[data['source']=='gog']['word2']))
wordcloud = WordCloud(
                      relative_scaling = 1.0,
                      stopwords = 'to of'
                      ).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title('GOG WORD2')
plt.show()


# **WORDCLOUD OF SAS SOUECE :**

# In[ ]:


# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud, STOPWORDS

text = (str(data[data['source']=='sas']['word1']))
wordclou = WordCloud(
                      relative_scaling = 1.0,
                      stopwords = 'to of'
                      ).generate(text)
plt.imshow(wordclou)
plt.axis("off")
plt.title('SAS WORD1')
plt.show()

text = (str(data[data['source']=='sas']['word2']))
wordcloud = WordCloud(
                      relative_scaling = 1.0,
                      stopwords = 'to of'
                      ).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title('SAS WORD2')
plt.show()


# **WORDCLOUD OF WRONGPLANET SOUECE**

# In[ ]:


# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud, STOPWORDS

text = (str(data[data['source']=='wrongplanet']['word1']))
wordclou = WordCloud(
                      relative_scaling = 1.0,
                      stopwords = 'to of'
                      ).generate(text)
plt.imshow(wordclou)
plt.axis("off")
plt.title('WRONGPLANET WORD1')
plt.show()

text = (str(data[data['source']=='wrongplanet']['word2']))
wordcloud = WordCloud(
                      relative_scaling = 1.0,
                      stopwords = 'to of'
                      ).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title('WRONGPLANET WORD2')
plt.show()

