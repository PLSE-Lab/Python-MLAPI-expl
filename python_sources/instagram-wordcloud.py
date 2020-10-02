#!/usr/bin/env python
# coding: utf-8

# #### Instagram wordcloud
# 
# Asked my followers to describe what means for them the term Data Science, and made a WordCloud out of the data they provided me with.

# In[ ]:


import numpy as np
import pandas as pd
import os
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


insta_data = pd.read_csv('../input/insta_data.csv')


# In[ ]:


insta_data


# In[ ]:


get_ipython().run_line_magic('pinfo', 'WordCloud')


# In[ ]:


text = " ".join(definition for definition in insta_data.description)


# In[ ]:


wordcloud = WordCloud(width=2000,height=2000,stopwords=STOPWORDS,background_color='white',colormap='RdPu',repeat=True).generate(text)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:




