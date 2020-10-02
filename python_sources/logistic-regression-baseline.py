#!/usr/bin/env python
# coding: utf-8

# # Disasters, real or fake?
# # Logistic regression baseline
# Started on 16 Jan 2020

# ##### Comments:
# * In this kernel, I use logistic regression as the binary classifier.
# * I shall start with the twitter text only. My purpose is to create a baseline model. Further on, I will explore adding other features, and using other models to see how much improvement can be made to the classification task.
# * To process the text data, I will simply use Count Vectorizer.

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Read "train.csv" and "test.csv into pandas

# In[ ]:


train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


# In[ ]:


train_df.info()


# In[ ]:


train_df.head()


# # Examine the train data

# In[ ]:


# check the class distribution for the target label in train_df?
train_df['target'].value_counts()


# * The class distribution looks quite balanced, with about 40% 'disaster' tweets.

# # Define X and y from train data for use in tokenization by Vectorizers

# In[ ]:


X = train_df['text']
y = train_df['target']


# # Split train data into a training and a validation set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


# In[ ]:


# examine the class distribution in y_train and y_test
print(y_train.value_counts(),'\n', y_val.value_counts())


# # Vectorize the data

# In[ ]:


# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = CountVectorizer(lowercase=True, stop_words='english', token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')
vect


# In[ ]:


# learn the vocabulary in the training data, then use it to create a document-term matrix
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix created from X_train
X_train_dtm


# In[ ]:


# transform the test data using the earlier fitted vocabulary, into a document-term matrix
X_val_dtm = vect.transform(X_val)
# examine the document-term matrix from X_test
X_val_dtm


# # Build and evaluate the disaster tweet classification model using Logistic Regression

# In[ ]:


# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=8)
logreg


# In[ ]:


# tune hyperparameter
from sklearn.model_selection import GridSearchCV
grid_values = {'C':[0.01, 0.1, 1.0, 3.0, 5.0]}
grid_logreg = GridSearchCV(logreg, param_grid=grid_values, scoring='neg_log_loss', cv=5)
grid_logreg.fit(X_train_dtm, y_train)
grid_logreg.best_params_


# In[ ]:


# set with recommended parameter
logreg = LogisticRegression(C=1.0, random_state=8)
# train the model using X_train_dtm & y_train
logreg.fit(X_train_dtm, y_train)


# In[ ]:


# make class predictions for X_test_dtm
y_pred_val = logreg.predict(X_val_dtm)


# In[ ]:


# compute the accuracy of the predictions
from sklearn import metrics
metrics.accuracy_score(y_val, y_pred_val)


# In[ ]:


# compute the accuracy of predictions with the training data
y_pred_train = logreg.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)


# In[ ]:


# look at the confusion matrix for y_test
metrics.confusion_matrix(y_val, y_pred_val)


# In[ ]:


# compute the predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_val_dtm)
y_pred_prob[:10]


# In[ ]:


# compute the log loss number
metrics.log_loss(y_val, y_pred_prob)


# # Train the Logistic Regression model with the entire dataset from "train.csv"

# In[ ]:


# Learn the vocabulary in the entire training data, and create the document-term matrix
X_dtm = vect.fit_transform(X)
# Examine the document-term matrix created from X_train
X_dtm


# In[ ]:


# Train the Logistic Regression model using X_dtm & y
logreg.fit(X_dtm, y)


# In[ ]:


# Compute the accuracy of training data predictions
y_pred_train = logreg.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)


# # Make predictions on the test data and compute the probabilities for submission

# In[ ]:


test = test_df['text']
# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_dtm = vect.transform(test)
# examine the document-term matrix from X_test
test_dtm


# In[ ]:


# make author (class) predictions for test_dtm
LR_y_pred = logreg.predict(test_dtm)
print(LR_y_pred)


# In[ ]:


# calculate predicted probabilities for test_dtm
LR_y_pred_prob = logreg.predict_proba(test_dtm)
LR_y_pred_prob[:10]


# # Create submission file

# In[ ]:


submission['target'] = LR_y_pred


# In[ ]:


submission


# In[ ]:


# Generate submission file in csv format
submission.to_csv('submission.csv', index=False)


# ### Thank you for reading this.
# ### Please upvote if you find it useful. Cheers!
