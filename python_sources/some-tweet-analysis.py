#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


data = pd.read_csv('../input/tweets.csv')


# In[3]:


data.head()


# So we have 28 columns. Let's check the names

# In[4]:


data.columns


# How many missing values do we have (NaN)?

# In[5]:


missing_values_count = data.isnull().sum()

# Number of NaN in the first ten columns
missing_values_count[0:10]


# In[6]:


# how many total missing values do we have?
total_cells = np.product(data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100


# That's a lot!!!

# Let's print some information of the first row

# In[7]:


print('id: ',data['id'][0])
print('source: ',data['source_url'][0])
print('favourite count: ',data['favorite_count'][0])
print('retweet_count: ',data['retweet_count'][0])
print('entities: ',data['entities'][0])
print('lang: ',data['lang'][0])
print('text: ', data['text'][0])
print('place: ', data['place_name'][0])


# In[8]:


lang_list=[]
for t in data['lang']:
    if t not in lang_list:
        lang_list.append(t)
        
print("Languages of the tweets:")
for t in lang_list:
    print(t)
    


# In[9]:


percent = np.zeros(len(lang_list))

for t in data['lang']:
    for index in range(len(lang_list)):
        if t == lang_list[index]:
            percent[index] += 1
            pass

percent /= 100


pie_chart = pd.Series(percent, index=lang_list, name='Languages')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));


# In[10]:


pie_chart.plot.barh(fontsize=11, figsize=(6, 6))


# In[11]:


data.text.str.split(expand=True).stack().value_counts()[:10]


# In[24]:


words = {}
total = 0

for word in data.text.str.split(expand=True).stack():
    if word in words:
        words[word] += 1
    else:
        words[word] = 1
    total += 1    

#print the 20 most frequency words 
x=[];y=[]    
sorted_words = sorted(words, key = words.get, reverse=True)
print("FREQUENCY OF WORDS")
print(" ")
count=0
for w in sorted_words:
    count += 1 
    print(w, ' ', words[w]/total)
    x.append(w)
    y.append(words[w])
    if count == 20:
        break


# In[50]:


import matplotlib.pyplot as plt 

y_pos = np.arange(len(x))
plt.title('Most common words')
plt.barh(y_pos,y,align = 'center')
plt.yticks(y_pos, x)
plt.show()


# In[21]:


hashtags ={}
for word in words:
    if word.startswith('#'):
        if word in hashtags:
            hashtags[word] += 1
        else:
            hashtags[word] = 1

sorted_hashtags = sorted (hashtags,key = hashtags.get, reverse=True)             
print (sorted_hashtags[:10])


# In[ ]:




