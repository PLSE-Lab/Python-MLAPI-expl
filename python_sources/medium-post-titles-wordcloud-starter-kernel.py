#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[ ]:


medium = pd.read_csv("../input/medium_post_titles.csv")


# In[ ]:


medium.dtypes


# In[ ]:


medium.shape


# In[ ]:


medium1k = medium.sample(1000)


# In[ ]:


medium1k.head(10)


# In[ ]:


text = medium1k.title.values


# In[ ]:


wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))


# In[ ]:


fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:




