#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("iso-8859-1"))
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/inaug_speeches.csv', encoding='iso-8859-1')
data= data.ix[57,]


# In[ ]:


# Making the data frame 'text' into a pandas series for wordcloud
#texts = pd.Series(data['text'].tolist()).astype(str)
texts=data['text']
dw = ['will', 'America', 'American', 'people', 'country', 'nation', 'one', 'every', 'Every','world',
      'now','never', 'new','great','dream','today','many','president','across','President','thank','Thank',
      'protected','right','make','ns','whether','first']
for w in dw:
    texts = texts.replace(w,'')

    
# Building a wordcloud!
cloud = WordCloud(
                 width=900,
                 height=800,
                 min_font_size=0.1,
                 background_color='black',
                 colormap='plasma'
                 ).generate(''.join(texts))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)
plt.axis('off')


# lol not much you can do with 4 string values and one datetime

# In[ ]:


speakers = data['Name']
txt = data['text']
for i in range(len(data)):
    print('The length of ', speakers[i],'speech was ', len(txt[i]), ' characters')


# In[ ]:


print('Max is ', len(max(txt)), ' characters')


# Thank you
# =
