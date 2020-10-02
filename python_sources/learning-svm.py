#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Dataset:
# The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html]. The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics. The ID field contains the patient identifiers. The characteristics of the cell samples from each patient are contained in fields Clump to Mit. The values are graded from 1 to 10, with 1 being the closest to benign. The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).
# 

# In[ ]:


df=pd.read_csv("../input/cell_samples.csv")


# In[ ]:


df.head()


# In[ ]:


import matplotlib.pyplot as plt


# ## Checking for datatypes

# In[ ]:



df.dtypes


# ## It looks like that 'BareNucl' is not of int type. Hence, to apply any algorithm we need to change it to a the 'int' type.

# In[ ]:


df=df[pd.to_numeric(df["BareNuc"],errors='coerce').notnull()]


# In[ ]:


df.dtypes


# In[ ]:


df["BareNuc"]=df["BareNuc"].astype('int')


# In[ ]:


X = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
y=df["Class"]


# # Train/Test data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


from sklearn import svm


# In[ ]:


from sklearn.metrics import jaccard_similarity_score as jsc


# # Applying Support Vector Machines from sklearn library and checking for different kernels and comparing them with the help of their accuracy score which is measured using jaccard_similarity_score.

# In[ ]:


clf=svm.SVC(kernel='rbf').fit(x_train,y_train)


# In[ ]:


yhat=clf.predict(x_test)


# In[ ]:


yhat


# In[ ]:


print("The accuracy score for our model when we use RBF kernel is:",jsc(y_test,yhat)*100,"%")


# ## For RBF kernel the accuracy score is approx: 96.5853%

# In[ ]:


clf=svm.SVC(kernel='linear').fit(x_train,y_train)


# In[ ]:


yhat1=clf.predict(x_test)


# In[ ]:


yhat1


# 

# In[ ]:


print("The accuracy score for our model when we use Linear kernel is:",jsc(y_test,yhat1)*100,"%")


# ## The accuracy score for our model when we use Linear kernel is: 96.09756097560975 %
# 

# In[ ]:


clf=svm.SVC(kernel='sigmoid').fit(x_train,y_train)


# In[ ]:


yhat2=clf.predict(x_test)
yhat2


# In[ ]:


print("The accuracy score for our model when we use Polynomial kernel is:",jsc(y_test,yhat2)*100,"%")


# ## The accuracy score for our model when we use Polynomial kernel is: 32.19512195121951 %
# 

# # Therefore, polynomial kernel performs worst in our case and the RBF kernel performs best in our case

# In[ ]:




