#!/usr/bin/env python
# coding: utf-8

# The Logistic Regression for Multiclass classification on sklearn Digits dataset. The Notebook shows the classification of digits by training and predicting the data and plotting the same on Heat Map to visualize it.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.datasets import load_digits


# In[ ]:


digits=load_digits()


# In[ ]:


dir(digits)


# In[ ]:


plt.matshow(digits.images[0])# to show images in pixel
plt.gray() # show images in black and white


# In[ ]:


for i in range(0,10):
    plt.matshow(digits.images[i]) # shoxws image from 1-9


# In[ ]:


digits.target


# In[ ]:


digits.target_names


# In[ ]:


#features
X= digits.data
#label
Y=digits.target


# In[ ]:


#test train and split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=1)# train/test is 80/20.


# In[ ]:


# test and train data is split, lets choose a classifier
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[ ]:


#training
lr.fit(X_train,Y_train)


# In[ ]:


#prediction 
y_predict=lr.predict(X_test)


# In[ ]:


#model score
lr.score(X_test,Y_test)


# In[ ]:


#lets see the confusion matrix for the same
from sklearn.metrics import confusion_matrix
cn=confusion_matrix(Y_test,y_predict)
cn


# In[ ]:


#lets visualise the same on Heatmap
plt.figure(figsize=(6,6))
sns.heatmap(cn,annot=True)

