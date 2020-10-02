#!/usr/bin/env python
# coding: utf-8

# **BOSTON HOUSE PRICING **
# 
# *CONCEPT USED: KNN (k nearest neigbours)*
# 
# 
# the program is just for understanding the most basic ML algorithm that is KNN 
# 
# 
# DIFFICULTY LEVEL : **EASY** 
# 
# 
# The algorithm estimates the cost of the house.
# 
# **KNN is not the right approch for this particular problem.**
# 
# thats why the mean squared error is as high as 25%
# 
# but this is just an example showing how we can use KNN for regression problems. 
#  The main purpose of KNN is basically for classification problems.
#  
#  KNN is the easiest algorithm to start undderstanding the ML algo
#  

# In[ ]:


#IMPORT FILE NECESSARY TO RUN THE CODE
#dataset is imported in order to get the boston house dataset from sklearn 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error


# In[ ]:


boston=datasets.load_boston()
x=boston.data[:,:]
y=boston.target
print(x.shape,y.shape)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=42)


# In[ ]:


def dis(v,w):
    return np.sqrt(np.sum((v-w)**2))


# In[ ]:


def knn_r(tr, tr_lab, te , k):
    distances = []
    
    for i in range(tr.shape[0]):
        distances.append(dis(tr[i], te))
    
    distances = np.array(distances)
    inds = np.argsort(distances)
    
    distances = distances[inds]
    tr_lab_2 = tr_lab[inds]
    value = np.average(tr_lab_2[:k])
    
    return value


# In[ ]:


def knn_reg(tr , tr_lab, te , te_lab , k):
    preds = []
    for i in range(te.shape[0]):
        value = knn_r(tr, tr_lab, te[i] , k)
        preds.append(value)
    
    preds  = np.array(preds)
    err = mean_squared_error(te_lab , preds)
    return err


# In[ ]:


acc = knn_reg(xtrain , ytrain , xtest , ytest ,5)
print ("MEAN SQUARED ERROR:",acc)

