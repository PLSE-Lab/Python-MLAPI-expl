#!/usr/bin/env python
# coding: utf-8

# Just a fun little side project I was working on, if you are interested in word clouds feel free to check this out!

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from os import path
from PIL import Image
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


df = pd.read_csv('../input/austin-animal-center-intakes/rows.csv')
df = df[df["Animal Type"] == 'Dog']


# In[5]:


text = df['Breed'][0]
wordcloud = WordCloud().generate(text)


# In[6]:


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[7]:


text = " ".join(review for review in df['Breed'])
print ("There are {} words in the combination of all review.".format(len(text)))
stopwords = set(STOPWORDS)
stopwords.update(["Mix", "mix"])
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)


# In[8]:


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[9]:



dog_mask = np.array(Image.open("../input/maskimage/pbsil.png"))
transformed_dog_mask = dog_mask 
#dog_mask


# In[11]:


wc = WordCloud(background_color="white", max_words=1000, mask=transformed_dog_mask,
               stopwords=stopwords, contour_width=.1, contour_color='black', colormap = 'bone')

# Generate a wordcloud
wc.generate(text)

# store to file
wc.to_file("dog_cloud.png")

# show
plt.figure(figsize=[30,20])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

