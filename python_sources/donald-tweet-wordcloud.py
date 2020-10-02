#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Read data
data = pd.read_csv('../input/Donald-Tweets!.csv')
data.info()


# In[ ]:


# remove two last null values
data['Unnamed: 10'].replace(np.nan, '0', inplace=True)
data['Unnamed: 11'].replace(np.nan, '0', inplace=True)


# Making a wordcloud 
# ===

# In[ ]:


wordcld = pd.Series(data['Tweet_Text'].tolist()).astype(str)
# Most frequent words in the data set. Using a beautiful wordcloud
cloud = WordCloud(width=900, height=900,
                  stopwords=('https', 'https co', 'co'), 
                  colormap='hsv').generate(''.join(wordcld.astype(str)))
plt.figure(figsize=(15, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# Cool
# ==
