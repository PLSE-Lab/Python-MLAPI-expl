#!/usr/bin/env python
# coding: utf-8

# In[122]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[123]:


X_test = pd.read_csv("../input/test.csv", index_col="PassengerId")
data = pd.read_csv("../input/train.csv", index_col="PassengerId")

from sklearn.model_selection import train_test_split

data.dropna(axis= 0, subset=["Survived"], inplace=True)

# get the target
y = data.Survived
data.drop(axis=1,columns="Survived", inplace=True)

data.drop(axis = 1, columns="Name", inplace = True)

high_cardinality_categorical = [col for col in data.columns if data[col].dtype == "object" and data[col].nunique() > 10]

low_cardinality_categorical = [col for col in data.columns if data[col].dtype == "object" and data[col].nunique() < 10]

data.drop(high_cardinality_categorical, axis = 1, inplace=True)
oh_col = pd.get_dummies(data[low_cardinality_categorical])
data.drop(axis = 1, columns = low_cardinality_categorical, inplace = True)
data = pd.concat([data, oh_col], axis = 1)


# In[124]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy= "median")
imputed_data = pd.DataFrame(imputer.fit_transform(data))
imputed_data.index = data.index
imputed_data.columns = data.columns

data = imputed_data


# In[125]:


# Load libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# Set random seed
np.random.seed(0)

# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
search_space = {"classifier__n_estimators": [50, 100, 150],
"classifier__max_features": [6, 7,8 ],
"classifier__max_leaf_nodes":[200,30,50,100]
               }
# Create grid search
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=True, n_jobs = -1)
# Fit grid search
best_model = gridsearch.fit(data, y)

best_model = best_model.best_estimator_


# In[ ]:





# In[127]:


X_test.drop(axis = 1,columns= high_cardinality_categorical,inplace=True)


# In[128]:



X_test.drop("Name", axis = 1, inplace = True)


# In[129]:


oh_test = pd.get_dummies(X_test[low_cardinality_categorical])

X_test.drop(axis = 1, columns= low_cardinality_categorical,inplace = True)

X_test = pd.concat( [X_test, oh_test], axis = 1)


# In[ ]:





# In[130]:


imputed_X_test = pd.DataFrame(imputer.transform(X_test))

imputed_X_test.index = X_test.index
imputed_X_test.columns = X_test.columns

X_test = imputed_X_test


# In[131]:


y_pred = best_model.predict(X_test)


# In[132]:





# In[ ]:


output = pd.DataFrame({"PassengerId": X_test.index, "Survived": y_pred})
output.to_csv("submission.csv", index= False)


# In[ ]:




