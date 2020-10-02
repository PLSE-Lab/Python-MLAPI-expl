#!/usr/bin/env python
# coding: utf-8

# In[116]:


# This Python 3 environment comes with many helpful analytics libraries installed# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Import the data

# In[77]:


data = pd.read_csv('../input/heart.csv')
data.head()


# ## Descriptive Statistics
# ### Age

# In[3]:


f, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.bar(data.age.value_counts().index, data.age.value_counts().values)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[4]:


print('The minimum age in the dataset is {}, the maximum age is {} and the mean is {}'.format(np.min(data.age.values), np.max(data.age.values), np.mean(data.age.values)))


# ### Sex
# Sex (1 = male; 0 = female)

# In[5]:


f, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.bar(data.sex.value_counts().index, data.sex.value_counts().values)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


# In[6]:


print('There are {} men in the dataset and {} women'.format(np.count_nonzero(data.sex.values== 1), np.count_nonzero(data.sex.values==0)))


# In[7]:


print('There are {} men with hearth disease (target == 1) and {} men "healthy"'.format(np.count_nonzero(data.sex.values & data.target.values), np.count_nonzero(data.sex.values & 1-data.target.values)))
print('There are {} women with hearth disease (target == 1) and {} women "healthy"'.format(np.count_nonzero(1-data.sex.values & data.target.values), np.count_nonzero(1-data.sex.values & 1-data.target.values)))


# ## Columns Features
# 1. Age (age in years)
# 2. Sex (1 = male; 0 = female)
# 3. CP (chest pain type)
# 4. TRESTBPS (resting blood pressure (in mm Hg on admission to the hospital))
# 5. CHOL (serum cholestoral in mg/dl)
# 6. FPS (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. RESTECH (resting electrocardiographic results)
# 8. THALACH (maximum heart rate achieved)
# 9. EXANG (exercise induced angina (1 = yes; 0 = no))
# 10. OLDPEAK (ST depression induced by exercise relative to rest)
# 11. SLOPE (the slope of the peak exercise ST segment)
# 12. CA (number of major vessels (0-3) colored by flourosopy)
# 13. THAL (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 14. TARGET (1 or 0)

# In[8]:


np.min(data.values[:,12])


# In[9]:


#Let's see some of the people with target == 1
data.head()


# In[ ]:





# In[10]:


#Let's see some of the people with target == 0
data.tail()


# ## Dummy Sex Variable

# In[69]:


dummy_sex = pd.get_dummies(data["sex"],prefix="sex")
dummy_sex.columns.values[0] = "Women"
dummy_sex.columns.values[1] = "Men"
dummy_sex.head()


# In[78]:


data = data.drop(["sex"],axis = 1)


# In[79]:


data = pd.concat([dummy_sex,data],axis=1)


# In[80]:


data.head()


# In order to have an idea on how the data can be classified, first some basic models are going to be run. 

# ## Separate the categorical data from the continuous

# In[91]:


data_cat = data[["Women","Men","age","cp","fbs","restecg","exang","slope","ca","thal","target"]]
data_cat.head()


# In[92]:


data_con = data[["trestbps","chol","thalach","oldpeak","target"]]
data_con.head()


# ## First Classifier
# 
# Linear classifier only taking the sex.

# In[93]:


X = data.values[:,0:2].astype(float)
Y = data.values[:,14].astype(int)


# In[94]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)


# In[95]:


clasificador = LinearSVC(C=0.0001)
clasificador.fit(X_train, y_train)
print(clasificador.score(X_test, y_test))
plot_decision_regions(X_test, y_test, clf=clasificador, legend=2)


# **This is clearly not a good classifier**

# ## Second Classifier
# 
# Using only the continous variables.

# In[101]:


X = data_con.values[:,0:-1].astype(float)
Y = data.values[:,-1].astype(int)


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)


# In[103]:


clasificador = LinearSVC(C=0.0001)
clasificador.fit(X_train, y_train)
print(clasificador.score(X_test, y_test))


# **This is definetely better but there is still room for improvement**

# ## Third Classifier
# Using only the categorical variables.

# In[104]:


X = data_cat.values[:,0:-1].astype(float)
Y = data.values[:,-1].astype(int)


# In[105]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)


# In[106]:


clasificador = LinearSVC(C=0.0001)
clasificador.fit(X_train, y_train)
print(clasificador.score(X_test, y_test))


# 

# ## Fourth Classifier
# With all of the features

# In[107]:


X = data.values[:,0:-1].astype(float)
Y = data.values[:,-1].astype(int)


# In[108]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)


# In[109]:


clasificador = LinearSVC(C=0.0001)
clasificador.fit(X_train, y_train)
print(clasificador.score(X_test, y_test))


# ## Let's run more models at once
# 
# Models for the continuous data only

# In[128]:


X = data_con.values[:,0:-1].astype(float)
Y = data_con.values[:,-1].astype(int)


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)


# In[130]:


std = StandardScaler()
pca = PCA(n_components=5)
clas = LinearSVC(C=200.0, max_iter=10000, tol=0.01)
clas2 = SVC(C=200.0, kernel='rbf')
pipe = Pipeline([('std', std), ('pca', pca), ('clas', clas)])
param_grid = {'pca__n_components': [4, 3, 2],
              'clas__C': [0.1, 1.0, 10.0, 100.0, 1000.0],
              'clas': [clas, clas2]}
search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)


# In[131]:


search.fit(X_train, y_train)


# In[132]:


search.best_estimator_.score(X_test, y_test)


# In[133]:


search.best_params_ # Best parameters for the continuous data


# ## Let's run more models at once
# 
# Models for all of the data

# In[135]:


X = data.values[:,0:-1].astype(float)
Y = data.values[:,-1].astype(int)


# In[136]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)


# In[137]:


search.fit(X_train, y_train)


# In[143]:


search.best_estimator_.score(X_test, y_test)


# In[145]:


search.best_params_ # Best parameters for the continuous data


# In[141]:


pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]


# In[ ]:




