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


# # IMPORTING LIBRARIES

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # LOADING DATA SET 

# In[ ]:


avocado = pd.read_csv("../input/avocado.csv")


# # DATA EXPLORATION 

# In[ ]:


print ('There are',len(avocado.columns),'columns:')
for x in avocado.columns:
    print(x+' ',end=',')


# In[ ]:


avocado.head()


# In[ ]:


avocado.tail()


# In[ ]:


avocado.info()


# no null values in the dataset.

# Dataset is uniformly distributed as both categories conventional as well as organic type has approximately same entries

# In[ ]:


avocado['type'].value_counts()


# In[ ]:


sns.heatmap(avocado.isnull(),yticklabels=False)


# In[ ]:


avocado.columns.values


# In[ ]:


sns.jointplot(x='Large Bags',y='Small Bags',data=avocado)


# In[ ]:


sns.jointplot(x='XLarge Bags',y='Large Bags',data=avocado)


# In[ ]:


sns.jointplot(x='Small Bags',y='XLarge Bags',data=avocado)


# In[ ]:


sns.countplot(avocado['type'])


# Dropping Unnamed: 0 and Date column from avocado dataset

# In[ ]:


avocado = avocado.drop(['Unnamed: 0','Date'],axis=1)


# Dataset after dropping Unnamed: 0 and Date columns

# In[ ]:


avocado.head()


# # IMPORTING SOME MORE LIBRARIES TO PREDICT THE 'TYPE'

# In[ ]:


from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


# # Feature Hashing 
# From feature_extraction we import Feature Hasher. Since region category contains 54 different regions so its not possible to direct pass this category while training the model, so Feature Hashing allows us to overcome this problem by letting us to encode them in a unique codes for each region, and then passing it to our model will increase our model's accuracy!

# Different types of region.

# In[ ]:


len(avocado['region'].value_counts())


# In[ ]:


fh = FeatureHasher(n_features=5,input_type='string')


# In[ ]:


hashed_features = fh.fit_transform(avocado['region']).toarray()


# In[ ]:


avocado = pd.concat([avocado,pd.DataFrame(hashed_features)],axis=1)


# Head of new dataset.

# In[ ]:


avocado.head()


# Dropping Region column as we have created hashed features out of them so this category is now of no use while training our model.

# In[ ]:


avocado = avocado.drop('region',axis=1)


# In[ ]:


X = avocado.drop('type',axis=1)
y = avocado['type']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # PREDICTION OF "TYPE" COLUMN

# # USING RANDOM FOREST CLASSIFIER.

# In[ ]:


rfc = RandomForestClassifier(n_estimators=100)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


pred1 = rfc.predict(X_test)


# Our  trained model is giving 100% accuracy!!!

# In[ ]:


print(classification_report(y_test,pred1))


# Our model is giving (false positive) + (false negatives) = 3.
# Total 3 inaccurate predictions which is pretty awesome.

# In[ ]:


print(confusion_matrix(y_test,pred1))


# #  USING KNearestNeighbors

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred2 = knn.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred2))


# In[ ]:


print(confusion_matrix(y_test,pred2))


# # USING SVM

# Manually sending list of parameters C and gamma so to get a best combination of parameters .

# In[ ]:


params = {'C':[1,10,100,1000,10000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[ ]:


grid = GridSearchCV(SVC(),params,verbose=3)


# In[ ]:


grid.fit(X_train,y_train)


# Best Parameters for C and gamma

# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


pred3 = grid.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred3))


# In[ ]:


print(confusion_matrix(y_test,pred3))


# So at last we have tried Random Forest Classifier, KNN and SVM but most efficient among them is Random Forest Classifier giving 100% accuracy, KNN is also not too bad as it has 98% accuracy but Support Vector Classifier(SVC) is not that much efficient in predicting the values having only 54% accuracy for conventional and 100% accuracy for organic type.
# Recommended Model for prediction of TYPE(coventional and organic categories) columns is Random Forest Classifier.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




