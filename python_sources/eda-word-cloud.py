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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Load Data**

# In[ ]:


vocab = pd.read_csv('../input/vocabulary.csv')
print(vocab.head())


# In[ ]:


sample_sub = pd.read_csv('../input/sample_submission.csv')
print(sample_sub.head())


# In[ ]:


print(os.listdir("../input/frame-sample/frame/"))


# In[ ]:


print('Total number of names :',vocab["Name"].nunique())


# In[ ]:


Vertical1 = vocab.groupby('Vertical1')
print(Vertical1.describe().head())


# In[ ]:


print(vocab.WikiDescription.head(10))


# In[ ]:


vocab.dtypes


# In[ ]:


#vocab.astype({'WikiDescription': 'String'}).dtypes
vocab['WikiDescription'] = vocab['WikiDescription'].apply(str)


# In[ ]:




text = " ".join(WD for WD in vocab.WikiDescription)
print ("There are {} words in the combination of all review.".format(len(text)))


# In[ ]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

wordcloud = WordCloud( max_words=500, background_color="white").generate(text)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# > **Over-Use of used ? :)**
