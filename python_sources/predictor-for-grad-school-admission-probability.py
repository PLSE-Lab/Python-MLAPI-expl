#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.drop(columns=['Serial No.'], inplace=True)


# In[ ]:


val_set = data[-100:]


# In[ ]:


df = data[:400]


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(df.drop(columns=['Chance of Admit ']))


# In[ ]:


df.drop(columns=['Chance of Admit ']).corr()


# In[ ]:


df.corr()['Chance of Admit ']


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


df['University Rating'].unique()


# In[ ]:


df['Research'].unique()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='Chance of Admit '),
    df['Chance of Admit '],
    test_size=0.2,
    random_state=10
)


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:


#Create a custom transformer that does the following
#OHE for University Rating and Research


# In[ ]:


#input pandas DataFrame, not numpy ndarray
#output is numpy ndarray
class InitialTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, ohe_attr):
        self.ohe_attr = ohe_attr
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        oh_encoder = LabelBinarizer()
        oh_attr = []
        for i in X[self.ohe_attr].columns:
            oh_attr.append(oh_encoder.fit_transform(X[i]))
        return np.concatenate([X.drop(columns=self.ohe_attr).values, np.concatenate(oh_attr, axis=1)], axis=1)


# In[ ]:


#test the transformer
init_transformer = InitialTransformation(['University Rating'])
aaa = init_transformer.fit_transform(X_train)
aaa.shape


# In[ ]:


#Use StandardScaler to scale all values for linear regression
scaler = StandardScaler()


# In[ ]:


#test scaler on custom transformer output
aa = scaler.fit_transform(aaa)
print(aa.shape)
aa[:1]


# In[ ]:


#make a pipeline for transformation
from sklearn.pipeline import Pipeline

data_transform = Pipeline([
    ('custom', InitialTransformation(['University Rating'])),
    ('scaler', StandardScaler())
])


# In[ ]:


#test the pipeline
bbb = data_transform.fit_transform(X_train)
bbb.shape


# In[ ]:


#Plot the learning curve to check if at the basic state the data is under or over fitting


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


# In[ ]:


def plot_learning_curve(df, model):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns='Chance of Admit '),
        df['Chance of Admit '],
        test_size=0.2,
        random_state=73
    )
    X_train_tr = data_transform.fit_transform(X_train)
    X_test_tr = data_transform.fit_transform(X_test)
    train_errors = []
    test_errors = []
    for i in range(1, len(X_train_tr)):
        linreg = model.fit(X_train_tr[:i], y_train[:i])
        y_train_preds = linreg.predict(X_train_tr[:i])
        y_preds = linreg.predict(X_test_tr)
        train_errors.append(mean_squared_error(y_train[:i], y_train_preds, squared=False))
        test_errors.append(mean_squared_error(y_test, y_preds, squared=False))
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(len(X_train_tr)-1), train_errors, label='Training error')
    plt.plot(np.arange(len(X_train_tr)-1), test_errors, label='Testing error')
    plt.legend()


# In[ ]:


plot_learning_curve(df, LinearRegression())


# In[ ]:


plot_learning_curve(df, Ridge())


# In[ ]:


plot_learning_curve(df, Lasso())


# In[ ]:


#Use Ridge
#Use GridSearchCV to find alpha and other hyperparameters


# In[ ]:


X_train_prepped = data_transform.fit_transform(X_train)
X_test_prepped = data_transform.fit_transform(X_test)


# In[ ]:


param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}


# In[ ]:


grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    scoring='neg_root_mean_squared_error'
).fit(X_train_prepped, y_train)


# In[ ]:


pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')


# In[ ]:


cols = [i for i in X_train.columns if i != 'University Rating']
for i in range(1,6):
    cols.append('UR_{}'.format(i))


# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(cols, grid_search.best_estimator_.coef_)
plt.axhline(0, color='r')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


print("Training RMSE: {}".format(grid_search.best_score_))
print("Testing RMSE: -{}".format(mean_squared_error(y_test, grid_search.best_estimator_.predict(X_test_prepped), squared=False)))


# In[ ]:


#Remove University Rating and SOP
#For the same estimator, find RMSE
#Compare best estimator RMSE with feature removed RMSE


# In[ ]:


gs_resultant = grid_search.best_estimator_


# In[ ]:


X_train_short = X_train.drop(columns=['University Rating', 'SOP'])
X_test_short = X_test.drop(columns=['University Rating', 'SOP'])


# In[ ]:


X_train_short_scaled = scaler.fit_transform(X_train_short)
X_test_short_scaled = scaler.fit_transform(X_test_short)


# In[ ]:


gs_ridge_red_feat = gs_resultant.fit(X_train_short_scaled, y_train)


# In[ ]:


print("Training error with reduced features on the GridSearchCV best model: -{}".format(mean_squared_error(y_train, gs_ridge_red_feat.predict(X_train_short_scaled), squared=False)))
print("Testing error with reduced features on the GridSearchCV best model: -{}".format(mean_squared_error(y_test, gs_ridge_red_feat.predict(X_test_short_scaled), squared=False)))


# In[ ]:


grid_search_cull = GridSearchCV(
    Ridge(),
    param_grid,
    scoring='neg_root_mean_squared_error'
).fit(X_train_short_scaled, y_train)


# In[ ]:


pd.DataFrame(grid_search_cull.cv_results_).sort_values(by='rank_test_score')


# In[ ]:


grid_search_cull.best_estimator_.coef_


# In[ ]:


mean_squared_error(y_test, grid_search_cull.best_estimator_.predict(X_test_short_scaled), squared=False)


# In[ ]:


#use fewer features model. i.e. X_train_short
#use the model from the first grid search survey that produced the lowest training RMSE: gs_ridge_red_feat
#run it in the validation set


# In[ ]:


val_set.info()


# In[ ]:


y_val = val_set['Chance of Admit ']


# In[ ]:


X_val = val_set.drop(columns=['University Rating', 'SOP', 'Chance of Admit '])


# In[ ]:


X_val_scaled = scaler.fit_transform(X_val)


# In[ ]:


mean_squared_error(y_val, gs_ridge_red_feat.predict(X_val_scaled), squared=False)


# In[ ]:




