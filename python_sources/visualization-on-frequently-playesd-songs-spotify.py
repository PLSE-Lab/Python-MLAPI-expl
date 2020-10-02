#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import all required libraries for reading data,analysing an visualizing data
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


filename='/kaggle/input/data-analytics-to-study-music-streaming-patterns/spotify.csv'
df=pd.read_csv(filename,encoding='ISO-8859-1')
df.head() 


# In[ ]:


data=df.iloc[:,4:]
data.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.columns


# In[ ]:


sns_plot=sns.pairplot(data, hue="Genre")
sns_plot.savefig("output.jpg")


# In[ ]:


sns.pairplot(data)
plt.plot()
plt.show()


# In[ ]:


plt.figure(figsize=(25,5))
sns_plot1=sns.countplot(data=df, x="Genre", label="count")


# In[ ]:


X = data.head()
X.head()


# In[ ]:


data_corr = X.corr()
data_corr


# In[ ]:


plt.figure(figsize=(8, 8))
sns.heatmap(data_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
           cmap= 'coolwarm')


# In[ ]:


data.groupby('Genre').size().plot.bar()


# In[ ]:


df.groupby('Artist').size().plot.bar()


# In[ ]:


plt.figure(figsize=(20,5))
df.groupby('Album').size().plot.bar()


# In[ ]:


plt.figure(figsize=(25,5))
df.groupby('Title').size().plot.bar()


# In[ ]:


from collections import Counter
Counter(data['Genre'])


# In[ ]:


import squarify 
plt.figure(figsize=(14,8))
squarify.plot(sizes=df.Genre.value_counts(), label=data["Genre"], alpha=.8 )
plt.axis('off')
plt.show()


# In[ ]:


import squarify 
plt.figure(figsize=(14,8))
squarify.plot(sizes=df.Album.value_counts(), label=data["Genre"], alpha=.8 )
plt.axis('off')
plt.show()


# In[ ]:


#WordCloud


# In[ ]:


from wordcloud import WordCloud, STOPWORDS


# In[ ]:


string=str(data.Genre)
plt.figure(figsize=(5,6))
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


string=str(df.Album)
plt.figure(figsize=(12,8))
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


string=str(df.Artist)
plt.figure(figsize=(5,6))
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




