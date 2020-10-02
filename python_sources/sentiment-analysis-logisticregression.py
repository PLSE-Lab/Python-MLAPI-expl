#!/usr/bin/env python
# coding: utf-8

# In[7]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Step 1: Load data and pre-process

# In[8]:


df = pd.read_csv('../input/Devotion_Reviews.csv')
df.head()


# Store the reivews and recommendations into another dataframe, converting string(True, False) to ingeters(0,1)

# In[9]:


df_review = df[['text', 'recommended']].copy()
df_review['recommended'] = df_review['recommended'].astype(dtype=np.int64)
df_review.head()


# Data Cleaning

# In[10]:


def pre_process(text):
    # lowercase
    text = text.lower()
    # tags
    text = re.sub('&lt;/?.*?&gt;',' &lt;&gt; ',text)
    # special characters and digits
    text=re.sub('(\\d|\\W)+',' ',text)
    
    return text

df_review['text'] = df_review['text'].apply(lambda x:pre_process(x))
df_review.head()


# ## Vectorization

# In[13]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
cv.fit(df_review['text'])
X = cv.transform(df_review['text'])

y = df_review['recommended']


# ## Build Classifier

# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)


# Find the best value of C in logistic regression

# In[15]:


for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print('Accuracy for C=%s: %s'
         % (c, accuracy_score(y_test, lr.predict(X_test))))


# Here I choose C=1 to build the final model.

# In[16]:


final_model = LogisticRegression(C=1)
final_model.fit(X, y)
print('Final Model Accuracy: %s' %accuracy_score(y_test, final_model.predict(X_test)))


# ## Find the most significant words

# In[17]:


feature_to_coef = {
    word: coef for word, coef in zip(
    cv.get_feature_names(), final_model.coef_[0])
}

print('Positive Words')
for best_positive in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1],
    reverse=True)[:10]:
    print(best_positive)
    


# In[18]:


print('Negative Words')
for best_negative in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1])[:10]:
    print(best_negative)


# ## Questions:
# 
# 1. Why 'negative' appears to be an important factor in the positive reviews.
# 2. Need to do more data cleaning. Since there are nonsense words in the list: 've'
# 3. This only analyzes the single word. What about word pairs?
# 4. Next is to implement TF-IDF
# 5. Different algorithms: SVM

# In[ ]:




