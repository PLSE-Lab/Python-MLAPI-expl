#!/usr/bin/env python
# coding: utf-8

# Exercices from the book "Practical Machine Learning and Image Processing" (Himanshu Singh)
# ### Algorithms covered:
# * Decision Trees
# * Support Vector Machines
# * Logistic Regression

# # Loading and setting the datasets

# In[ ]:


import pandas as pd
train_set = pd.read_csv('../input/digit-recognizer/train.csv')
train_set


# In[ ]:


y_train = train_set['label']
y_train


# In[ ]:


train_set.drop('label', axis=1, inplace=True)
X_train = train_set
y_train = pd.Categorical(y_train)


# # The 3 Algos

# #### Importing...

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


# In[ ]:


logreg = LogisticRegression()
dt = DecisionTreeClassifier()
svc = LinearSVC()


# #### Fiting...

# In[ ]:


model_logreg = logreg.fit(X_train, y_train)
model_dt = dt.fit(X_train, y_train)
model_svc = svc.fit(X_train, y_train)


# # Testing the accuracy of models on the training set
# ... of course this thing here are being done the wrong way ...

# In[ ]:


from sklearn.metrics import accuracy_score

pred_on_train1 = model_logreg.predict(X_train)
pred_on_train2 = model_dt.predict(X_train)
pred_on_train3 = model_svc.predict(X_train)

print("Decision Tree Accuracy is: ", accuracy_score(pred_on_train1, y_train)*100)
print("Logistic Regression Accuracy is: ", accuracy_score(pred_on_train2, y_train)*100)
print("Support Vector Machine Accuracy is: ", accuracy_score(pred_on_train3, y_train)*100)


# # Predicting

# In[ ]:


X_test = pd.read_csv('../input/digit-recognizer/test.csv')

pred_logreg = model_logreg.predict(X_test)
pred_dt = model_dt.predict(X_test)
pred_svc = model_svc.predict(X_test)


# In[ ]:




