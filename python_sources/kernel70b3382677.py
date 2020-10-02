#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


# In[ ]:


ratings = pd.read_excel("/kaggle/input/parisdata/rewiews808K.xlsx")


# In[ ]:


ratings.head()


# In[ ]:


blob = TextBlob(ratings['text'].iloc[0])
blob.tags


# In[ ]:





# In[ ]:


testimonial = TextBlob("terrible to use. What stupid!")


# In[ ]:





# In[ ]:


for sentence in blob.sentences:
    print(sentence, sentence.sentiment.polarity)


# In[ ]:


blob.sentiment


# In[ ]:


def get_sentiment(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity


# In[ ]:


ratings['text'] = ratings['text'].apply(lambda x: str(x))


# In[ ]:


wdummy = ratings['text'].apply(lambda x: get_sentiment(x))


# In[ ]:


wdummy.shape


# In[ ]:


ratings['sentiment'] = wdummy


# In[ ]:


ratings.head()


# In[ ]:


ratings_copy = ratings


# In[ ]:


ratings_copy['sentiment'] = ratings_copy['sentiment'] + 1
ratings_copy['sentiment'] = ratings_copy['sentiment'] / 2


# In[ ]:


ratings_copy.tail()


# In[ ]:


ratings.to_csv("ratings.csv", index=None)
ratings_copy.to_csv("ratings_copy.csv", index=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




