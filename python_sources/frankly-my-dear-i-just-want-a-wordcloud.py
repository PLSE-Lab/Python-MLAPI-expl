#!/usr/bin/env python
# coding: utf-8

# Gone with the wind is one of my favorite movies from childhood. And I wanted to learn Andreas Mueller's word cloud package for a while because unfortunately enough, I've a lot of way to go when it comes to data visualization. So here goes the combination. 
# ![](https://images6.alphacoders.com/464/464170.jpg)

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


# read the data
gone_wind = open('../input/text-mining-gone-with-the-wind/GoneWithTheWind.txt',errors='ignore').read()


# In[ ]:


import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# mask address 
from PIL import Image
mask_img = np.array(Image.open('../input/gone-with-the-wind-images/4.jpg'))


# In[ ]:


plt.figure(figsize=(8,8))
plt.imshow(mask_img)
plt.axis('off')


# # Generate the wordcloud

# In[ ]:


plt.figure(figsize=(8,8))
wc = WordCloud(max_words=200,
               background_color = 'white',
               mask = mask_img,
               stopwords=set(STOPWORDS),
               random_state=42, mode = 'RGB', 
               ).generate(gone_wind)
plt.title("Gone With The Wind - Word Cloud",fontsize=20)
plt.axis('off')
plt.imshow(wc)


# In[ ]:




