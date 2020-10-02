#!/usr/bin/env python
# coding: utf-8

# # **STEP #1: PROBLEM STATEMENT**
# 
# *Predicting if the cancer diagnosis is benign or malignant based on several observations/features
# *30 features are used, examples:
# 
#   - radius (mean of distances from center to points on the perimeter)
#   - texture (standard deviation of gray-scale values)
#   - perimeter
#   - area
#   - smoothness (local variation in radius lengths)
#   - compactness (perimeter^2 / area - 1.0)
#   - concavity (severity of concave portions of the contour)
#   - concave points (number of concave portions of the contour)
#   - symmetry 
#   - fractal dimension ("coastline approximation" - 1)
# Datasets are linearly separable using all 30 input features
# 
# Number of Instances: 569
# Class Distribution: 212 Malignant, 357 Benign
# Target class:
#    - Malignant
#    - Benign

# # **STEP #2: IMPORTING DATA**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # Import Pandas for data manipulation using dataframes, data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.


# In[ ]:


# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[ ]:


cancer


# In[ ]:


cancer.keys()


# In[ ]:


print(cancer['DESCR'])


# In[ ]:


print(cancer['target_names'])


# In[ ]:


print(cancer['target'])


# In[ ]:


print(cancer['feature_names'])


# In[ ]:


print(cancer['data'])


# In[ ]:


cancer['data'].shape


# In[ ]:


# pandas dataframe
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# In[ ]:


# let's take a look at our data
df_cancer.head()


# In[ ]:


df_cancer.tail()


# In[ ]:


# now visualization time
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )


# In[ ]:


sns.countplot(df_cancer['target'], label = "Count") 


# In[ ]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[ ]:


# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# # **STEP #4: MODEL TRAINING (FINDING A PROBLEM SOLUTION)**

# In[ ]:


# Let's drop the target label column
X = df_cancer.drop(['target'],axis=1)


# In[ ]:


X


# In[ ]:


y = df_cancer['target']
y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)


# # **STEP #5: EVALUATING THE MODEL**

# In[ ]:


y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)


# In[ ]:


sns.heatmap(cm, annot=True)


# In[ ]:


print(classification_report(y_test, y_predict))


# # **STEP #6: IMPROVING THE MODEL**

# In[ ]:


min_train = X_train.min()
min_train


# In[ ]:


range_train = (X_train - min_train).max()
range_train


# In[ ]:


X_train_scaled = (X_train - min_train)/range_train
X_train_scaled


# In[ ]:


sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[ ]:


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# In[ ]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[ ]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[ ]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")


# In[ ]:


print(classification_report(y_test,y_predict))


# # **IMPROVING THE MODEL - PART 2**

# In[ ]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[ ]:


grid.fit(X_train_scaled,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_predictions = grid.predict(X_test_scaled)


# In[ ]:


cm = confusion_matrix(y_test, grid_predictions)


# In[ ]:


sns.heatmap(cm, annot=True)


# In[ ]:


print(classification_report(y_test,grid_predictions))


# # ** Conclusion**
# 
# Machine learning techniques (SVM) was able to classify tumors into Malignant/Benign with %97 accuracy
# 
# If you liked this kernel please up vote :)

# In[ ]:




