#!/usr/bin/env python
# coding: utf-8

# **SIMPLE CODE TO UNDERSTAND THE WORKFLOW**

# Importing all the libraries

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import model_selection


# Importing the training and test dataset

# In[9]:


train = pd.read_csv("../input/train.csv")
print(train.shape)

test = pd.read_csv("../input/test.csv")
print(test.shape)


# In[10]:


print(train.columns)
train.describe()

print(test.columns)
test.describe()


# Extracting the target variable and the features

# To split the Features and Target values of both Training and Test dataset

# In[11]:




train_x = train.iloc[:,2:]
train_y_ = train.iloc[:,1]
train_y = []
for i in train_y_:
    train_y.append(i)
    
test_x = test.iloc[:,1:]


#print(test_x)


# Scaling of the data

# In[20]:


scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
#test_x = scaler.fit_transform(test_x)


# Cross Validation to reduce overfitting of the data

# Applying Regression layer 

# In[22]:



regression = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)
regression.fit(train_x,train_y)

scores = model_selection.cross_val_score(regression,train_x,train_y,scoring="accuracy",cv=50)   # To Cross validate and remodel it with less features
# cv - number of runs to find cross validated model

test_y_ = regression.predict_proba(test_x)
print("Training Accuracy score: ",regression.score(train_x,train_y))
test_y = []
#for i in test_y_[:,1]:
#    test_y.append(i)
    
#print(len(train_y))
print((test_y_[:,1]))

#score = accuracy_score(train_y,test_y)
#print(score)


# Submission of code

# In[18]:


submission = pd.DataFrame({"id":test["id"],"target":test_y_[:,1]})

#Visualize the first 5 rows
submission.head()

filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

