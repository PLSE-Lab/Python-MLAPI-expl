#!/usr/bin/env python
# coding: utf-8

# ## I'm new to English and Pyton so if I have anything misunderstanding, please tell me 
# 
# # This notebook will introduce you the way to make 'good' LinearSCVmodel and LogisticRegressionmodel by using some 'C'

# first, import libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# and next,read data

# In[ ]:


data=pd.read_csv('../input/fish-market/Fish.csv')


# Check data

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# define the features and target

# In[ ]:


X=data.drop(['Species'],axis=1)


# In[ ]:


##to make multi calss model so convert fish name to int
y= data['Species'].map({'Perch': 0, 'Bream': 1,'Roach': 2,'Pike': 3,'Smelt': 4,'Parkki':5,'Whitefish':6})


# to avoid bad effection, standarize features

# In[ ]:


X_std=StandardScaler().fit(X).transform(X)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X_std,y,random_state=0)


# make LinearSVC model and get the best accuracy by using some C 

# In[ ]:


LinearSVC_result=[]
def get(_X,_y,_X_test,_y_test):
    for i in {1,10,50,100,1000}:
        model=LinearSVC(C=i,random_state=0)
        model.fit(_X,_y)
        LinearSVC_result.append(model.score(_X_test,_y_test))
    return LinearSVC_result


# In[ ]:


get(X_train,y_train,X_test,y_test)


# next make LogisticRegression

# In[ ]:


LogisticRegression_result=[]
def get(_X,_y,_X_test,_y_test):
    for i in {1,10,50,100,1000}:
        model=LogisticRegression(C=i,random_state=0)
        model.fit(_X,_y)
        LogisticRegression_result.append(model.score(_X_test,_y_test))
    return LogisticRegression_result


# In[ ]:


get(X_train,y_train,X_test,y_test)


# In my case, best socre is 0.925, and I found that score of LogisticRegression change dinamically by changing 'C'

# Thank you for reading my notebook, if you have any questions please ask me. 

# In[ ]:




