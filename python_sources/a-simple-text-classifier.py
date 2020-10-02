#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# ## Reading the file

# In[ ]:


data = pd.read_csv('../input/reviews.csv', delimiter=',')


# ## Converting into numeric values

# In[ ]:


def func(x):
    if x == 'True':
        return 1.0
    if x == 'False':
        return 0.0

data['Recommends'] = data['Recommends'].apply(func)


# ## Populating an artifical Good column
# 
# If a restaurant is recommended or the review is more than 3, then it is considered a good restaurant. This feature was created to simplify the classification problem into a binary classification problem.

# In[ ]:


def is_good(review, recommends):
    if recommends == 1.0 or review > 3:
        return 1
    return 0

data['Good'] = list(map(is_good, data['Review'], data['Recommends']))


# ## Splitting the training and testing data

# In[ ]:


from sklearn import model_selection

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(data['Review Text'], data['Good'], test_size=0.1, random_state=42)


# ## TD-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, ngram_range=(1,3), stop_words='english')

features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, n_jobs=5)
clf.fit(features_train, labels_train)


# In[ ]:


pred = clf.predict(features_test)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(labels_test, pred)


# In[ ]:





# In[ ]:




