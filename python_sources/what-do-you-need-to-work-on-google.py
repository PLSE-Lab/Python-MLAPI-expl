#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# In 2018, Google is completing 20 years old. This young adult company is known in the whole world by his products, such as the search engine, Gmail, Drive and so on and, with its products, they made a revolution through millions of people's lives. One consequence of being so popular is that a lot of people want to work at Google. And they have opportunities many areas. Maybe, it is possible to get a shortcut (for those who want to work at Google) by knowing what are the most wanted prerequisites and preferred qualifications by Google in their job offerings.

# **So, what do I need to have or do to work on Google?**

# For answering the title's question, we must first identify what columns do we have in our database.

# In[ ]:


df = pd.read_csv('../input/job_skills.csv')
df.head()


# Ok, we have "Category", "Minimum Qualifications" and "Preferred Qualifications". I'll search for all categories in the base, and show the number of vacancies for each area.

# In[ ]:


categories = df.groupby('Category')
categories.size()
num_categories = []

for category in df.Category.unique():
    num_categories.append(categories.size()[category])

d = {'Category': df.Category.unique(), 'Vacancies': num_categories }
cat_df = pd.DataFrame(data=d)
bp = cat_df.plot(kind='bar', figsize=(15, 10), x='Category')


# I believe that it is possible to see what are the most important requirement and diferentials by analyzing the frequence of the words in the "Minimum Qualifications" and "Preferred Qualifications". So, for example, if "Big Data" appears many times in the "Minimum Qualifications", it is an important requisite for the job.

# In[ ]:


def makecloud(column, category, color):
    words = pd.Series(df.loc[df['Category'] == category][column]).str.cat(sep=' ')
    wc = WordCloud(stopwords=STOPWORDS, colormap=color, background_color='White', width=800, height=400).generate(words)
    plt.figure(figsize=(16,18))
    plt.imshow(wc)
    plt.axis('off')
    plt.title(category + " " + column);
    
def makeclouds(index, color):
    makecloud('Minimum Qualifications', df.Category.unique()[index], color)
    makecloud('Preferred Qualifications', df.Category.unique()[index], color)


# Now, we must print the word cloud for "Minimum Qualifications" and "Preferred Qualifications" for each job category.

# In[ ]:


makeclouds(0, 'Reds')


# In[ ]:


makeclouds(1, 'Greens')


# In[ ]:


makeclouds(2, 'Blues')


# In[ ]:


makeclouds(3, 'Reds')


# In[ ]:


makeclouds(4, 'Greens')


# In[ ]:


makeclouds(5, 'Blues')


# In[ ]:


makeclouds(6, 'Reds')


# In[ ]:


makeclouds(7, 'Greens')


# In[ ]:


makeclouds(8, 'Blues')


# In[ ]:


makeclouds(9, 'Reds')


# In[ ]:


makeclouds(10, 'Greens')


# In[ ]:


makeclouds(11, 'Blues')


# In[ ]:


makeclouds(12, 'Reds')


# In[ ]:


makeclouds(13, 'Greens')


# In[ ]:


makeclouds(14, 'Blues')


# In[ ]:


makeclouds(15, 'Reds')


# In[ ]:


makeclouds(16, 'Greens')


# In[ ]:


makeclouds(17, 'Blues')


# In[ ]:


makeclouds(18, 'Reds')


# In[ ]:


makeclouds(19, 'Greens')


# In[ ]:


makeclouds(20, 'Blues')


# In[ ]:


makeclouds(21, 'Reds')


# In[ ]:


makeclouds(22, 'Greens')

