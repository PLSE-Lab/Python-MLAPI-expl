#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import pandas as pd 
from pandas.plotting import scatter_matrix
import numpy as np
plt.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')


# Get the data for processing

# In[ ]:


# import the data for prediction 
df = pd.read_csv('../input/abalone-dataset/abalone.csv')
df.head()


# In[ ]:


df.head()


# In[ ]:


# looking into our data, there are no outliers
df.info()


# In[ ]:


# let's explore sex, it is a categorical variable 
df['Sex'].value_counts()


# Visualizing the dataset

# In[ ]:


# visualizing the features
df.hist(bins=50, figsize=(20, 15))
plt.show()


# In[ ]:


# Find the r value aka standard correlation coefficient 
corr_matrix = df.corr()
corr_matrix['Rings']


# In[ ]:


# plot variables against each other to find the relationship 
attributes = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
scatter_matrix(df[attributes], figsize=(15, 12))


# In[ ]:


# convert text labels to integer labels
sex_label = LabelEncoder()
df['Sex'] = sex_label.fit_transform(df['Sex'])
df.head()


# In[ ]:


df.describe()


# In[ ]:


# define the features and the labels
# dropping the sex column due to the lack of correlation 

X = df.drop(['Rings', 'Sex'], axis=1)
y = df['Rings']
# divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


# In[ ]:


X_train.hist(bins=50, figsize=(20, 15))
plt.show()


# In[ ]:


y_train.hist(bins=50)
plt.show()


# Scale our data 

# In[ ]:


# standardize our data 
# standardization is less affected by outliers
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


sklearn.metrics.SCORERS.keys()


# Try several models using grid search and evaluate them

# In[ ]:


# fine tune hyperparamaters
knn_grid = [
    {'n_neighbors': [i+1 for i in range(10)] }, 
]
knn = KNeighborsRegressor()
knn_search = GridSearchCV(knn, knn_grid, scoring='neg_mean_squared_error', cv=5,
                          return_train_score=True, n_jobs=-1)
knn_search.fit(X_train, y_train)


# In[ ]:


knn_search.best_params_


# In[ ]:


result = knn_search.cv_results_
for mean_acc, params in zip(result['mean_test_score'], result['params']):
    print(mean_acc, params)


# In[ ]:


# try other models
kernel = ['linear', 'rbf', 'poly', 'sigmoid']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma}


# In[ ]:


svm = SVR()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=20, verbose=1)
svm_search.fit(X_train, y_train)


# In[ ]:


svm_search.best_params_


# In[ ]:


result = svm_search.cv_results_
for mse, params in zip(result['mean_test_score'], result['params']):
    print(mse, params)


# In[ ]:


ensemble_grid =  {'n_estimators': [(i+1)*10 for i in range(20)],
                 'criterion': ['mse', 'mae'],
                 'bootstrap': [True, False]}

ensemble = RandomForestRegressor()
ensemble_search = RandomizedSearchCV(ensemble, ensemble_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=10, verbose=1)
ensemble_search.fit(X_train, y_train)


# In[ ]:


ensemble_search.best_params_


# In[ ]:


result = ensemble_search.cv_results_
for mse, params in zip(result['mean_test_score'], result['params']):
    print(mse, params)


# In[ ]:


svm_reg = svm_search.best_estimator_
svm_reg.fit(X_train, y_train)


# In[ ]:


svm_pred = svm_reg.predict(X_test)
print(np.sqrt(mean_squared_error(svm_pred, y_test)))


# In[ ]:


knn_reg = knn_search.best_estimator_
knn_reg.fit(X_train, y_train)


# In[ ]:


knn_pred = knn_reg.predict(X_test)
print(np.sqrt(mean_squared_error(knn_pred, y_test)))


# In[ ]:


ensemble_reg = ensemble_search.best_estimator_
ensemble_reg.fit(X_train, y_train)


# In[ ]:


ensemble_pred = ensemble_reg.predict(X_test)
print(np.sqrt(mean_squared_error(ensemble_pred, y_test)))

