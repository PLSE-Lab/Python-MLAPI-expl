#!/usr/bin/env python
# coding: utf-8

# # Predicting Behavior based on protein expression using SVM

# Context
# Expression levels of 77 proteins measured in the cerebral cortex of 8 classes of control and Down syndrome mice exposed to context fear conditioning, a task used to assess associative learning.
# Content
# 
# The data set consists of the expression levels of 77 proteins/protein modifications that produced detectable signals in the nuclear fraction of cortex. There are 38 control mice and 34 trisomic mice (Down syndrome), for a total of 72 mice. In the experiments, 15 measurements were registered of each protein per sample/mouse. Therefore, for control mice, there are 38x15, or 570 measurements, and for trisomic mice, there are 34x15, or 510 measurements. The dataset contains a total of 1080 measurements per protein. Each measurement can be considered as an independent sample/mouse.
# 
# The eight classes of mice are described based on features such as genotype, behavior and treatment. According to genotype, mice can be control or trisomic. According to behavior, some mice have been stimulated to learn (context-shock) and others have not (shock-context) and in order to assess the effect of the drug memantine in recovering the ability to learn in trisomic mice, some mice have been injected with the drug and others have not.
# 
# Classes:
# 
#     c-CS-s: control mice, stimulated to learn, injected with saline (9 mice)
# 
#     c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)
# 
#     c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
# 
#     c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice)
# 
#     t-CS-s: trisomy mice, stimulated to learn, injected with saline (7 mice)
# 
#     t-CS-m: trisomy mice, stimulated to learn, injected with memantine (9 mice)
# 
#     t-SC-s: trisomy mice, not stimulated to learn, injected with saline (9 mice)
# 
#     t-SC-m: trisomy mice, not stimulated to learn, injected with memantine (9 mice)
# 
# Attribute Information
# 
# [1] Mouse ID
# 
# [2:78] Values of expression levels of 77 proteins; the names of proteins are followed by N indicating that they were measured in the nuclear fraction. *For example: DYRK1A_n*
# 
# [79] Genotype: control (c) or trisomy (t)
# 
# [80] Treatment type: memantine (m) or saline (s)
# 
# [81] Behavior: context-shock (CS) or shock-context (SC)
# 
# [82] Class: c-CS-s, c-CS-m, c-SC-s, c-SC-m, t-CS-s, t-CS-m, t-SC-s, t-SC-m

# In[179]:


import pandas as pd
import numpy as  np


# In[186]:


#reading data from file
data = pd.read_csv('../input/Data_Cortex_Nuclear.csv')


# In[187]:


#Display all the columns
data.columns


# In[188]:


#Drop unwanted Columns
data = data.drop(['MouseID','Treatment', 'Genotype', 'class'],axis=1)
#Drop all the columns which have more than or equal to 10 missing values
temp_data = (data.isnull().sum()  < 10)
columns_with_missing_lte_10 =[]
for i in range(temp_data.shape[0]):
    if temp_data.iloc[i] == True:
        columns_with_missing_lte_10.append(temp_data.index[i])
data = data[columns_with_missing_lte_10]


# In[189]:


data.columns


# In[190]:


#Replace Blank values with NaN
columns = data.columns
X_data = data[columns[:-1]]
y_data = data[columns[-1]]
X_data.replace('',np.NaN,inplace=True)


# In[191]:


#Fill missing values with mean
from sklearn.preprocessing import Imputer


# In[192]:


imputer = Imputer()


# In[193]:


imputer.fit(X_data)


# In[194]:


X_data = pd.DataFrame(columns=X_data.columns,data=imputer.transform(X_data))


# In[195]:


X_data.isnull().sum()


# In[196]:


#Train and test 
from sklearn.model_selection  import train_test_split


# In[197]:


X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, test_size=0.2)


# In[198]:


from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
import scipy


# In[201]:


clf  =  SVC()


# In[204]:


#Randomized Grid Search to optimize hyperparameters
s = RandomizedSearchCV(clf,param_distributions={'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
  'kernel': ['rbf','linear']},)


# In[205]:


#Train model
s.fit(X_train,y_train)


# In[206]:


#Test Score
print("Train score ",s.score(X_train,y_train))


# In[207]:


#Train Score
print("Test score ",s.score(X_test,y_test))


# In[208]:


#Best hyperparameters
s.best_params_


# In[211]:


#Predict first 10 values
s.predict(X_test[:10])


# In[212]:


#Actual first 10 classes
y_test[:10]


# In[ ]:




