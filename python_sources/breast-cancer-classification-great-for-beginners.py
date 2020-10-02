#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Business Case :** 
# 
# Our objective is to identify whether the blood sample tumor cell, taken for test is cancerous or benign (harmless). The normal procedure starts from a path lab where doctor takes a sample of blood see it through microscope and based on the features of the cells, take a call on whether the cell is cancerous or not. 

# <img src="https://image.shutterstock.com/image-photo/breast-cancer-core-biopsy-ductal-600w-551260165.jpg" width="600px">

# **Requirement translated to machine learning question **
# 
# We receive the data and based on its features we need to be able to come up with a model which could classify each example as either a cancerous or benign (harmless). We are going to use SVC for the job (Support vector classification) 

# In[ ]:


from sklearn.datasets import load_breast_cancer
cancerdata = load_breast_cancer()
cancerdata.keys()


# In[ ]:


# Lets see the description of the dataset 
print(cancerdata['DESCR'])


# Our data has total 10 distinct features, their mean standard error and worst or larfest of thest value thus total 30 features. We have 569 such image example data. We also have results whether it is cancerous or not. These are the features.
# 
#        - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry 
#         - fractal dimension ("coastline approximation" - 1)
#         

# In[ ]:


cancerdata['data'].shape # Tells number of example and features


# In[ ]:


# Creating dataframe from our data 
df = pd.DataFrame(np.c_[cancerdata['data'],cancerdata['target']], columns = np.append(cancerdata['feature_names'],['target']))
df.head()


# In[ ]:


import seaborn as sns
distinctfeature = cancerdata['feature_names'][:10] # We know first 10 features are mean features
#sns.pairplot(df,hue='target',vars=['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness'])
sns.pairplot(df,hue='target',vars=distinctfeature)


# We can see except few features pretty much all features are clearly separable 

# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(df.corr()) # Checking the correlation in the data 


# We can see thre are few features which are highly correlated except others, we can choose to discard them, treat them or proceed, Lets proceed without modifying the data and check how it goes.

# **Proceeding to model building stage **

# In[ ]:


X = df.drop(['target'],axis=1)
y = df['target']


# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
clf_svc = SVC()
clf_svc.fit(X_train,y_train)


# Checkout the parameters that are passed to support vector classifier. two of the most important parameters are C, gamma, kernel and degree
# 
# * C : regularization parameter (just like lambda in regression)
# * gamma : gamma controls the variation (variance) of the gaussian distribution
# * degree : for non linear classification (polynomial function)

# In[ ]:


baseline_ypred = clf_svc.predict(X_test)
cm = confusion_matrix(y_test,baseline_ypred)
print(cm)


# In[ ]:


print(classification_report(y_test,baseline_ypred))


# So the Baseline model appears to be poor with accuracy of 60%, we try and improove the accuracy of the model.
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainScaled = sc.fit_transform(X_train)
X_testScaled = sc.transform(X_test)

# standard scaler X = (X - mean ) / StandardDeviation


# In[ ]:


clf_svc.fit(X_trainScaled,y_train)
scaled_ypred = clf_svc.predict(X_testScaled)
print(classification_report(y_test,scaled_ypred))
print(confusion_matrix(y_test,scaled_ypred))


# ### BAM !!! 
# what just happend !! We signinifactly see improovement in the accuracy of the model. 

# In[ ]:


# Lets just change the split and rerun the test. 
XScaled = sc.transform(X)
# using the same Xtrain y train nomenclature 
X_train,X_test,y_train,y_test = train_test_split(XScaled,y,test_size=0.4)
clf_svc.fit(X_train,y_train)
scaled_ypred = clf_svc.predict(X_test)
print(classification_report(y_test,scaled_ypred))
print(confusion_matrix(y_test,scaled_ypred))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,scaled_ypred), annot=True)


# #### Result :  
# By just feature scaling we can obtain a model with great accuracy in classifying the breast cancer cells data in to cancerous of harmless.

# In[ ]:




