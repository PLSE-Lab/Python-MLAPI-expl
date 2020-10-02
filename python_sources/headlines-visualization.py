#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing important library
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


articles = pd.read_csv("../input/ArticlesApril2017.csv")
articles.head(5)


# now cleaning the dataset. I am gona a use headline feature for visualization
# 

# In[3]:


headlines = articles["headline"]
headlines.head(3)


# In[5]:


CV = CountVectorizer(stop_words = "english", lowercase = "TRUE", token_pattern= '[A-z]+')
wnl = WordNetLemmatizer()
cleaned = []
for i in range(len(headlines)):
    cleaned.append(''.join([wnl.lemmatize(headlines[i])]))
transformed = CV.fit_transform(cleaned)
print(CV.get_feature_names())


# In[6]:


from wordcloud import WordCloud

wordcloud = WordCloud(random_state=42,max_words=500,
                          max_font_size=40,).generate(str(CV.get_feature_names()))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:




