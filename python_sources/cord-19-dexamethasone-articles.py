#!/usr/bin/env python
# coding: utf-8

# # Dexamethasone - a new hope for treatment of COVID-19?
# 
# Wikipedia: [https://en.wikipedia.org/wiki/Dexamethasone](http://en.wikipedia.org/wiki/Dexamethasone)

# # Filter articles by keywords

# In[ ]:


# init
import os
import numpy as np
import pandas as pd
import time
import warnings 

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

warnings.filterwarnings('ignore')


# ### Import data

# In[ ]:


# load metadata
t1 = time.time()
df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
t2 = time.time()
print('Elapsed time:', t2-t1)


# ### Search for specific keyword in abstracts

# In[ ]:


# define keyword
my_keyword = 'dexamethason'


# In[ ]:


def word_finder(i_word, i_text):
    found = (str(i_text).lower()).find(str(i_word).lower()) # avoid case sensitivity
    if found == -1:
        result = 0
    else:
        result = 1
    return result

# partial function for mapping
word_indicator_partial = lambda text: word_finder(my_keyword, text)
# build indicator vector (0/1) of hits
keyword_indicator = np.asarray(list(map(word_indicator_partial, df.abstract)))


# In[ ]:


# number of hits
print('Number of hits for keyword <', my_keyword, '> : ', keyword_indicator.sum())


# In[ ]:


# add index vector as additional column
df['selection'] = keyword_indicator

# select only hits from data frame
df_hits = df[df['selection']==1]


# In[ ]:


# show results
df_hits


# In[ ]:


# show all abstracts
n = df_hits.shape[0]
for i in range(0,n):
    print(df_hits.title.iloc[i],":\n")
    print(df_hits.abstract.iloc[i])
    print('\n')


# In[ ]:


# make available for download
df_hits.to_csv('hits.csv')


# # Wordcloud of abstracts

# In[ ]:


text = " ".join(abst for abst in df_hits.abstract)


# In[ ]:


stopwords = set(STOPWORDS)


# In[ ]:


wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,
                      width = 600, height = 400,
                      background_color="white").generate(text)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

