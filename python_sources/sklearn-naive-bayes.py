#!/usr/bin/env python
# coding: utf-8

# **Naive Bayes is  super fast and performs surprisingly well.**

# In[ ]:


import pandas as pd


# In[ ]:


# read the data into a pandas datafrome
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print(df_train.shape)
print(df_test.shape)


# ### Define X and y

# In[ ]:


X = df_train.drop('label', axis=1)
y = df_train['label']

X_test = df_test

print(X.shape)
print(y.shape)
print(X_test.shape)


# ### Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X, y)

y_pred = nb.predict(X_test)


# ### Create a submission file

# In[ ]:


# The index should start from 1 instead of 0
df = pd.Series(range(1,28001),name = "ImageId")

ID = df

submission = pd.DataFrame({'ImageId':ID, 
                           'Label':y_pred, 
                          }).set_index('ImageId')

submission.to_csv('mnist_nb.csv', columns=['Label']) 


# In[ ]:




