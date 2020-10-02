#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


# In[ ]:


# Because train.text contains NaN
# https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/138213
train.dropna(subset=['text'], inplace=True)


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


positive_words = []
negative_words = []
neutral_words = []

for row in (train.query('sentiment=="positive"')['selected_text']):
    positive_words += row.split(' ')

for row in (train.query('sentiment=="negative"')['selected_text']):
    negative_words += row.split(' ')

for row in (train.query('sentiment=="neutral"')['selected_text']):
    neutral_words += row.split(' ')

data = [
    ' '.join(positive_words),
    ' '.join(negative_words),
    ' '.join(neutral_words)
]

stopWords = stopwords.words("english")
vectorizer = TfidfVectorizer(stop_words=stopWords)
X = vectorizer.fit_transform(data).toarray()

tfidf_df = pd.DataFrame(X.T, index=vectorizer.get_feature_names(),
                        columns=['positive', 'negative', 'neutral'])


# In[ ]:


tfidf_df.shape


# # Top 10 positive words

# In[ ]:


tfidf_df.sort_values('positive', ascending=False).head(10)


# # Top 10 negative words

# In[ ]:


tfidf_df.sort_values('negative', ascending=False).head(10)


# # Top 10 neutral words

# In[ ]:


tfidf_df.sort_values('neutral', ascending=False).head(10)


# In[ ]:




