#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.


# In[ ]:


sns.set_palette("Set3",10)
sns.set_context('talk')
magazine=pd.read_csv("../input/archive.csv")
magazine[magazine['Country']=='China']


# In[ ]:


magazine.groupby(['Country']).size().plot(kind='bar')


# In[ ]:


magazine.Honor.value_counts().plot.pie(figsize=(8,8))


# In[ ]:


magazine.groupby(['Country']).size().sort_values(ascending=False)


# In[ ]:


magazine.groupby('Name').size().sort_values(ascending=False)[:1]
magazine[magazine['Name']=='Franklin D. Roosevelt']


# In[ ]:


US_p=magazine[magazine['Title']=='President of the United States'] 
US_name=US_p.groupby(['Category','Name']).size()
US_name


# In[ ]:


magazine.groupby(['Category']).size().plot(kind='bar')


# In[ ]:


from wordcloud import WordCloud
wordList=[]
Context=magazine.Context.unique()
for word in Context:
    wordList.append(word)

wc=WordCloud(background_color="white",max_font_size=80, random_state=3, relative_scaling=.5)
wc.generate(str(wordList))
plt.figure(figsize=(15,20))
plt.imshow(wc)
plt.axis('off')

