#!/usr/bin/env python
# coding: utf-8

# # Classification Heart Disease
# 
# We'll try classification method for heart disease

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/heart.csv')


# In[ ]:


data.info()


# as we can see there is no null data in these columns

# In[ ]:


data.head()


# > 1. age 
# > 2. sex 
# > 3. chest pain type (4 values) 
# > 4. resting blood pressure 
# > 5. serum cholestoral in mg/dl 
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved 
# > 9. exercise induced angina 
# > 10. oldpeak = ST depression induced by exercise relative to rest 
# > 11. the slope of the peak exercise ST segment 
# > 12. number of major vessels (0-3) colored by flourosopy 
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# # Data Visualization

# In[ ]:


plt.figure(figsize=(14,6))
sns.set_style('whitegrid')
sns.countplot(x='target',data=data)


# In[ ]:


plt.figure(figsize=(14,6))
sns.set_style('dark')
sns.countplot(x='target',hue='sex',data=data,palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(14,6))
sns.set_style('dark')
sns.countplot(x='target',hue='thal',data=data)


# In[ ]:


data['age'].plot(kind='hist',bins=30,color='red',figsize= (16,7))


# ## Train and test Split

# In[ ]:


X=data.drop(columns=['target'],axis=1)
y=data['target']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=101)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr_model= LogisticRegression()
lr_model.fit(X_train,y_train)


# In[ ]:


lr_pred=lr_model.predict(X_test)


# # Evaluation
# 
# We can check precision,recall,f1-score using classification report!

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,lr_pred))


# **Not so bad!!**

# # Support Vector Classification (SVC)

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc_model = SVC()


# In[ ]:


svc_model.fit(X_train,y_train)


# In[ ]:


svc_pred = svc_model.predict(X_test)


# In[ ]:


print(classification_report(y_test,svc_pred))


# Woah! Notice that we are classifying everything into a single class! This means our model needs to have it parameters adjusted (it may also help to normalize the data).
# 
# We can search for parameters using a GridSearch!

# # Gridsearch
# 
# Finding the right parameters (like what C or gamma values to use) is a tricky task! But luckily, we can be a little lazy and just try a bunch of combinations and see what works best! This idea of creating a 'grid' of parameters and just trying out all the possible combinations is called a Gridsearch, this method is common enough that Scikit-learn has this functionality built in with GridSearchCV! The CV stands for cross-validation which is the
# 
# GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested. 

# In[ ]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.model_selection import GridSearchCV


# One of the great things about GridSearchCV is that it is a meta-estimator. It takes an estimator like SVC, and creates a new estimator, that behaves exactly the same - in this case, like a classifier. You should add refit=True and choose verbose to whatever number you want, higher the number, the more verbose (verbose just means the text output describing the process).

# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# What fit does is a bit more involved then usual. First, it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to built a single new model using the best parameter setting.

# In[ ]:


# May take awhile!
grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# Then you can re-run predictions on this grid object just like you would with a normal model.

# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(classification_report(y_test,grid_predictions))


# ### It's better!!
