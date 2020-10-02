#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# # Building Vectors

# In[ ]:


#Create instance
count_vectorizer = feature_extraction.text.CountVectorizer()


# In[ ]:


#Fit Transform train data
train_vectors = count_vectorizer.fit_transform(train_df["text"])


# In[ ]:


#Only transform test data - so that train and test use the same vectors
test_vectors = count_vectorizer.transform(test_df["text"])


# # Model

# In[ ]:


clf = linear_model.RidgeClassifier()


# In[ ]:


scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores.mean()


# In[ ]:


clf.fit(train_vectors, train_df["target"])


# # Predict

# In[ ]:


#get sample file for creating submission file
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


sample_submission["target"] = clf.predict(test_vectors)


# In[ ]:


sample_submission.head()


# # Submit

# In[ ]:


#Got to the Output section of this Kernel -> click on Submit to Competition
sample_submission.to_csv("submission.csv", index=False)

