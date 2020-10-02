#!/usr/bin/env python
# coding: utf-8

# # This is a very simple starting example for working with NLP

# In[ ]:


import numpy as np 
import pandas as pd 
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# ## Data loading

# In[ ]:


base_dir = '../input/nlp-getting-started/'
train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
sub_df = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))


# In[ ]:


train_df.head()


# In[ ]:


train_df.isna().sum()


# ## TF-IDF preprocessing

# In[ ]:


X = train_df["text"]
y = train_df["target"]
X_test = test_df["text"]
X.shape, y.shape, X_test.shape


# Using both TRAIN and TEST datasets to train tf-idf, it gives some more details about existing words

# In[ ]:


X_for_tf_idf = pd.concat([X, X_test])
tfidf = TfidfVectorizer(stop_words = 'english')
tfidf.fit(X_for_tf_idf)
X = tfidf.transform(X)
X_test = tfidf.transform(X_test)
del X_for_tf_idf


# ## Training and Evaluating

# Let's split TRAIN data to train and validation

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# Using GridSearchCV to find the best 'gamma' for SVM

# In[ ]:


parameters = { 
    'gamma': [0.7, 1, 'auto', 'scale']
}
model = GridSearchCV(SVC(kernel='rbf'), parameters, cv=4, n_jobs=-1).fit(X_train, y_train)


# Let's use accuracy and f1 to evaluate the model

# In[ ]:


y_val_pred = model.predict(X_val)
accuracy_score(y_val, y_val_pred), f1_score(y_val, y_val_pred)


# Confusion matrix for more details:

# In[ ]:


confusion_matrix(y_val, y_val_pred)


# ## Creating submission

# In[ ]:


y_test_pred = model.predict(X_test)


# In[ ]:


sub_df["target"] = y_test_pred
sub_df.to_csv("submission.csv",index=False)

