#!/usr/bin/env python
# coding: utf-8

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
my_keyword = 'main protease'


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


# look at an example: metadata first
df_hits.iloc[0]


# In[ ]:


# look at an example: the abstract itself
df_hits.abstract.iloc[0]


# ### Use another keyword

# In[ ]:


# define keyword
my_keyword2 = 'vaccine'


# In[ ]:


# partial function for mapping
word_indicator_partial2 = lambda text: word_finder(my_keyword2, text)
# build indicator vector (0/1) of hits
keyword_indicator2 = np.asarray(list(map(word_indicator_partial2, df_hits.abstract)))


# In[ ]:


# number of hits
print('Number of hits for keywords <', my_keyword, '> + <', my_keyword2, '> : ', keyword_indicator2.sum())


# In[ ]:


# add index vector as additional column
df_hits['selection'] = keyword_indicator2

# select only hits from data frame
df_hits2 = df_hits[df_hits['selection']==1]


# In[ ]:


df_hits2


# In[ ]:


# look at an example: metadata first
df_hits2.iloc[0]


# In[ ]:


# look at an example: the abstract itself
df_hits2.abstract.iloc[0]


# In[ ]:


# save selection to CSV file for further evaluations
df_hits2.to_csv('selection.csv')


# # Wordcloud of abstracts

# In[ ]:


text = " ".join(abst for abst in df_hits2.abstract)


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

