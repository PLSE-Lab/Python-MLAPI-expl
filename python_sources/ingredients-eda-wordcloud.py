#!/usr/bin/env python
# coding: utf-8

# # Loading the libraries

# In[ ]:


# Load libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# # Loading dataset

# In[ ]:


df = pd.read_json('../input/train.json')
df.head()


# # WordCloud
# ----
# Creating a **WordCloud** of ingredients for each **Cuisine**
# 
# [WordCloud Python Documentation](https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html)

# In[ ]:


'''
Input: dataframe and cuisine
Output: WordCloud image
Processing: Create a list of all ingredients in given cuisine
            Create wordcluod based on the count of each type of ingredient
'''

def word_cloud(df, cuisine):
    
    # Read the whole text.
    # For selected cuisine
    lst = []
    for each in df[df['cuisine'] == cuisine]['ingredients']:
        lst = lst + each
        
    text = ' '.join(lst)
    
    # Create word cloud
    wordcloud = WordCloud(background_color='white', max_words=100, width=2000, height=1000).generate(text)

    # Display the generated image:
    plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# ## [Greek Cuisine](https://en.wikipedia.org/wiki/Greek_cuisine)

# In[ ]:


# greek
word_cloud(df, 'greek')


# ## [Cuisine of the Southern United States](https://en.wikipedia.org/wiki/Cuisine_of_the_Southern_United_States)

# In[ ]:


# southern_us
word_cloud(df, 'southern_us')


# ## [Indian Cuisine](https://en.wikipedia.org/wiki/Indian_cuisine)

# In[ ]:


# indian
word_cloud(df, 'indian')


# ## [Jamaican Cuisine](https://en.wikipedia.org/wiki/Jamaican_cuisine)

# In[ ]:


# jamaican
word_cloud(df, 'jamaican')


# ## [Filipino Cuisine](https://en.wikipedia.org/wiki/Filipino_cuisine)

# In[ ]:


# filipino
word_cloud(df, 'filipino')


# ## [Spanish Cuisine](https://en.wikipedia.org/wiki/Spanish_cuisine)

# In[ ]:


# spanish
word_cloud(df, 'spanish')


# ## [Italian Cuisine](https://en.wikipedia.org/wiki/Italian_cuisine)

# In[ ]:


# italian
word_cloud(df, 'italian')


# ## [Mexican Cuisine](https://en.wikipedia.org/wiki/Mexican_cuisine)

# In[ ]:


# mexican
word_cloud(df, 'mexican')


# ## [Chinese Cuisine](https://en.wikipedia.org/wiki/Chinese_cuisine)

# In[ ]:


# chinese
word_cloud(df, 'chinese')


# ## [British Cuisine](https://en.wikipedia.org/wiki/British_cuisine)

# In[ ]:


# british
word_cloud(df, 'british')


# ## [Thai Cuisine](https://en.wikipedia.org/wiki/Thai_cuisine)

# In[ ]:


# thai
word_cloud(df, 'thai')


# ## [Vietnamese Cuisine](https://en.wikipedia.org/wiki/Vietnamese_cuisine)

# In[ ]:


# vietnamese
word_cloud(df, 'vietnamese')


# ## [Cajun Creole Cuisine](https://en.wikipedia.org/wiki/Louisiana_Creole_cuisine)

# In[ ]:


# cajun_creole
word_cloud(df, 'cajun_creole')


# ## [Brazilian Cuisine](https://en.wikipedia.org/wiki/Brazilian_cuisine)

# In[ ]:


# brazilian
word_cloud(df, 'brazilian')


# ## [Japanese Cuisine](https://en.wikipedia.org/wiki/Japanese_cuisine)

# In[ ]:


# japanese
word_cloud(df, 'japanese')


# ## [Irish Cuisine](https://en.wikipedia.org/wiki/Irish_cuisine)

# In[ ]:


# irish
word_cloud(df, 'irish')


# ## [Korean Cuisine](https://en.wikipedia.org/wiki/Korean_cuisine)

# In[ ]:


# korean
word_cloud(df, 'korean')


# ## [Moroccan Cuisine](https://en.wikipedia.org/wiki/Moroccan_cuisine)

# In[ ]:


# moroccan
word_cloud(df, 'moroccan')


# ## [Russian Cuisine](https://en.wikipedia.org/wiki/Russian_cuisine)

# In[ ]:


# russian
word_cloud(df, 'russian')


# # Bibliography
# ----
# **WordCloud**
# 
# https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
# 
# ----
# **Cuisines - Wikipedia**
# 
# 
# https://en.wikipedia.org/wiki/Greek_cuisine
# 
# https://en.wikipedia.org/wiki/Cuisine_of_the_Southern_United_States
# 
# https://en.wikipedia.org/wiki/Indian_cuisine
# 
# https://en.wikipedia.org/wiki/Jamaican_cuisine
# 
# https://en.wikipedia.org/wiki/Filipino_cuisine
# 
# https://en.wikipedia.org/wiki/Spanish_cuisine
# 
# https://en.wikipedia.org/wiki/Italian_cuisine
# 
# https://en.wikipedia.org/wiki/Mexican_cuisine
# 
# https://en.wikipedia.org/wiki/Chinese_cuisine
# 
# https://en.wikipedia.org/wiki/British_cuisine
# 
# https://en.wikipedia.org/wiki/Thai_cuisine
# 
# https://en.wikipedia.org/wiki/Vietnamese_cuisine
# 
# https://en.wikipedia.org/wiki/Louisiana_Creole_cuisine
# 
# https://en.wikipedia.org/wiki/Brazilian_cuisine
# 
# https://en.wikipedia.org/wiki/Japanese_cuisine
# 
# https://en.wikipedia.org/wiki/Irish_cuisine
# 
# https://en.wikipedia.org/wiki/Korean_cuisine
# 
# https://en.wikipedia.org/wiki/Moroccan_cuisine
# 
# https://en.wikipedia.org/wiki/Russian_cuisine

# In[ ]:




