#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[ ]:


import numpy as np 
import pandas as pd 
import random
import re


# # Prepare Data

# In[ ]:


test_data =  pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


# In[ ]:


data = {}
for a, b in zip(test_data['text'], test_data['sentiment']):
    data[a] = b


# In[ ]:


def seperate(data):
    X = []
    Y = []
    
    for K, V in data.items():
        K = K.lower()
        
        X.append(K)
        Y.append(V)
    return X,Y


# In[ ]:


X, Y = seperate(data)


# # Vectorization

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn.pipeline import Pipeline
model = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
])


# # Train & Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.80)


# # Model Fit

# In[ ]:


model.fit(X_train, y_train)


# # Accuracy

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, model.predict(X_test))


# # Confusion Matrix

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


# In[ ]:


plot_confusion_matrix(model, X_test, y_test)
plt.show()

