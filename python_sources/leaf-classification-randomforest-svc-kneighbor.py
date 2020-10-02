#!/usr/bin/env python
# coding: utf-8

# ## **I. PROJECT OBJECTIVE**
# 
# The objective of this machine learning project is to use extracted leaf features, including shape, margin, and texture, to accurately identify 99 species of plants.

# ## **II. DATA ANALYSIS**
# * [1. Import libraries](#ImportLib)
# * [2. Load & initially explore data](#LoadData)
# * [3. Preprocess data](#Preprocess)
# * * [3.1 Label encoder](#labelEncoder)
# * * [3.2 Feature scaling](#featureScaling)
# * * [3.3 Split train-validation set](#split)
# * * [3.4 Save test set id & features with scaling](#testSet)
# * [4. Model selection, model training & fine tuning](#model)
# * [5. Deploy models on validation set & choose the best one](#valid)
# * [6. Deploy the chosen model on test set](#test)

# <a id='ImportLib'></a>

# ## **1. Import libraries**

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, GridSearchCV

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='LoadData'></a>

# ## **2. Load & initially explore data**

# In[ ]:


train = pd.read_csv('../input/leaf-classification/train.csv', delimiter=',')
test = pd.read_csv('../input/leaf-classification/test.csv', delimiter=',')


# In[ ]:


train.head()


# In[ ]:


train.info() # 990 samples, 192 features


# In[ ]:


train['species'].nunique() # 99 unique species


# In[ ]:


train['species'].value_counts() # each species has 10 samples in training set


# In[ ]:


test.head()


# In[ ]:


test.info() # Target: classify 594 test samples into 99 species


# <a id='Preprocess'></a>

# ## **3. Preprocess data**

# <a id='labelEncoder'></a>

# ### **3.1 Label encoder**

# In[ ]:


le = LabelEncoder().fit(train['species'])


# In[ ]:


# encode species in training set
train['label'] = le.transform(train['species'])


# In[ ]:


# drop id & species columns. seperate labels in training set
labels = train['label']
train_df = train.drop(columns=['id','species','label'])


# <a id='featureScaling'></a>

# ### **3.2 Feature scaling**

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_df)
train_scale = pd.DataFrame(scaler.transform(train_df))


# <a id='split'></a>

# ### **3.3 Split train - validation set**

# In[ ]:


# create train & validation set. In this case, we dont want just simple random sampling but stratification because of large number of classes (99)
# stratification will make sure there's an equal number of samples per class in training set
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for train_index, val_index in sss.split(train_scale,  labels):
    x_train, x_val = train_scale.iloc[train_index], train_scale.iloc[val_index]
    y_train, y_val = labels.iloc[train_index], labels.iloc[val_index]


# In[ ]:


print(x_train.shape,y_train.shape)
print(x_val.shape, y_val.shape)


# In[ ]:


y_train.value_counts() # each class has 8 samples in training set


# <a id='testSet'></a>

# ### **3.4 Save test set id & features with scaling**

# In[ ]:


test_id = test['id']
test_features = test.drop('id', axis=1)
test_features_scale = scaler.transform(test_features)


# <a id='model'></a>

# ## **4. Model selection, model training & fine tuning**

# Three classifiers are considered in this notebook: Random forest classifier, Support vector machine classifier and KNeighbors classifier.
# 
# GridSearchCV is used to fine-tune some hyperparameters

# In[ ]:


cv_sets = ShuffleSplit(n_splits=10,test_size=0.20,random_state=42)
classifiers = [RandomForestClassifier(), SVC(), KNeighborsClassifier()]
params = [{'n_estimators' : [3,10,30], 'max_features':[2,4,6,8]},
          {'kernel':('linear','poly','sigmoid','rbf'),'C':[0.01,0.05,0.025,0.07,0.09,1.0], 'gamma':['scale'], 'probability':[True]},
          {'n_neighbors': [3,5,7,9]}]


# In[ ]:


best_estimators = []
for classifier, param in zip(classifiers, params):
    grid = GridSearchCV(classifier,param,cv=cv_sets)
    grid = grid.fit(x_train,y_train)
    best_estimators.append(grid.best_estimator_)


# In[ ]:


best_estimators 


# <a id='valid'></a>

# ## **5. Deploy models on validation set & choose the best one**

# In[ ]:


for estimator in best_estimators:
    estimator.fit(x_train, y_train)
    name = estimator.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    print('**Training set**')
    train_predictions = estimator.predict(x_train)
    acc = accuracy_score(y_train, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    train_predictions = estimator.predict_proba(x_train)
    ll = log_loss(y_train, train_predictions)
    print("Log Loss: {}".format(ll))
    
    print('**Validation set**')
    train_predictions = estimator.predict(x_val)
    acc = accuracy_score(y_val, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    train_predictions = estimator.predict_proba(x_val)
    ll = log_loss(y_val, train_predictions)
    print("Log Loss: {}".format(ll))
    
print("="*30)


# KNeighbors Classifier performs well on both training & validation set. The other two classifiers overfit on training set.

# <a id='test'></a>

# ## **6. Deploy the chosen model on test set**

# In[ ]:


pred = best_estimators[2].predict_proba(test_features_scale) # KNeighbors classifer model


# In[ ]:


submission = pd.DataFrame(pred, index = test_id, columns = le.classes_ )


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:




