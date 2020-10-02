#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


# In[ ]:


df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding='latin-1')
df.head()


# In[ ]:


df.shape


# In[ ]:


df['Artist.Name'].value_counts().head(10)


# In[ ]:


df['Genre'].value_counts().head(10)


# In[ ]:


artists = pd.DataFrame(df['Artist.Name'].value_counts().head(10)).reset_index()
artists.columns = ['Artist', 'Artist_count']
genre = pd.DataFrame(df['Genre'].value_counts().head(10)).reset_index()
genre.columns = ['Genre', 'Genre_count']
artists


# In[ ]:


genre


# In[ ]:


plt.figure(figsize=(34,12))
sns.countplot(df.Genre)


# In[ ]:


plt.figure(figsize=(50,16))
sns.countplot(df['Artist.Name'])


# In[ ]:


sns.scatterplot(x=df.Genre,y=df.Popularity,data=df)


# In[ ]:


#top Genres listened.
string=str(df.Genre)
plt.figure(figsize=(12,8))
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




