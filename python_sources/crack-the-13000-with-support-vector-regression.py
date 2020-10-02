#!/usr/bin/env python
# coding: utf-8

# # 0. Intro
# 
# In this Kernel only Support Vector Regression with 4 different Kernels is performed to crack the score of 13000.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1. Data Preperation

# Import Data, 
# 
# Remove Columns with much missing values, 
# 
# Impute remaining missing values, 
# 
# One Hot Encode categorical data and 
# 
# scale numerical data to Standard Deviation 1 and Mean 0.

# In[ ]:


# Data paths
dir_df = '../input/home-data-for-ml-course'
train_dir = os.path.join(dir_df, 'train.csv')
test_dir = os.path.join(dir_df, 'test.csv')

# read in data
df_train = pd.read_csv(train_dir)
df_test = pd.read_csv(test_dir)
print('shape of df_train:', df_train.shape)
print('shape of df_test:', df_test.shape)
display(df_train.head())
display(df_test.head())


# In[ ]:


# check for null values
print('null values of train:')
print(df_train.isnull().sum()[0:81])
print('\n\n null values of test:')
print(df_test.isnull().sum()[0:20])


# In[ ]:


# remove following features (columns), because they have so many null values:
# ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

df_train_drop = df_train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
df_test_drop = df_test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)

print('shape of df_train_drop', df_train_drop.shape)
print('shape of df_test_drop', df_test_drop.shape)


# In[ ]:


# split df_train_drop in X (input) and y (output)
# and call df_test_drop without Id X_test
X, y = df_train_drop.iloc[ : ,1:75], df_train_drop['SalePrice']
X_test = df_test_drop.drop('Id', axis = 1)
print('shape of X:', X.shape)
print('shape of y:', y.shape)
print('shape of X_test:', X_test.shape)


# In[ ]:


# split in categorical and numerical data
# because we like to transform them differently
# Note: You can check the data type of the different features with X.dtypes
cat_features = (X.dtypes == 'object')
cat_list = [i for i in cat_features]
cont_features = (X.dtypes != 'object')
cont_list = [i for i in cont_features]

X_cat = X.iloc[ : , cat_list]
X_cont = X.iloc[ : , cont_list]
X_test_cat = X_test.iloc[ : , cat_list]
X_test_cont = X_test.iloc[ : , cont_list]

print('shape of X_cat:', X_cat.shape)
print('shape of X_cont:', X_cont.shape)
print('shape of X_test_cat:', X_test_cat.shape)
print('shape of X_test_cont:', X_test_cont.shape)


# In[ ]:


# Now we one hot encode the categorical data
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
# We fit the OneHotEncoder on training plus test data
# and replace Null Values of our dataset with an empty string
# in this way we make shure to get no error message, because
# of unknown values (maybe not the most elegant way I do this)
X_full_cat = np.vstack([X_cat.fillna(""), X_test_cat.fillna("")])

# fit OneHotEncoder on concatenate dataset
ohe.fit(X_full_cat)

# one hot transform training and test set
X_cat_ohe = ohe.transform(X_cat.fillna("")).toarray()
X_test_cat_ohe = ohe.transform(X_test_cat.fillna("")).toarray()

print('shape of X_cat_ohe:', X_cat_ohe.shape)
print('shape of X_test_cat_ohe:', X_test_cat_ohe.shape)


# In[ ]:


# impute mean values to numerical data for NANs (Null Values) and normalize to std 1 and mean 0

# SimpleImputer is a simple way to replace NANs with some apropriate value
# by default with the mean value of the respective column
from sklearn.impute import SimpleImputer
si = SimpleImputer()
si.fit(X_cont)
X_cont_imp = si.transform(X_cont)
X_test_cont_imp = si.transform(X_test_cont)

# Normalize
# StandardScalar is a simple way for scaling all features to standard deviation 1 and mean 0
# but of course you can also do this 'by hand'
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_cont_imp)
X_cont_imp_std = ss.transform(X_cont_imp)
X_test_cont_imp_std = ss.transform(X_test_cont_imp)

# check if data is scaled correctly
print('mean and std of X:', np.mean(X_cont_imp_std, axis = 0), np.std(X_cont_imp_std, axis = 0))
print('mean and std of X_test:', np.mean(X_test_cont_imp_std, axis = 0), np.std(X_test_cont_imp_std, axis = 0))

print('shape of X_cont_imp:', X_cont_imp.shape)
print('shape of X_test_cont_imp:', X_test_cont_imp.shape)


# In[ ]:


# finally we combine categorical and numerical data
X_train = np.hstack([X_cat_ohe, X_cont_imp_std])
X_final_test = np.hstack([X_test_cat_ohe, X_test_cont_imp_std])
print('shape of X_train', X_train.shape)
print('shape of X_final_test', X_final_test.shape)


# # 2. Determine Best Parameters by a Grid Search

# In[ ]:


# to perform a grid search, we use GridSearchCV which is a simple and good option
# at first we have to define a Parameter grid, which contains the 
# parameters, which should be tested during Search

# we try 4 different options for the kernel namly 'linear', 'rbf', 
# 'sigmoid' and 'poly'. The do not have all the same parameters. So
# for example it would be meaningless to give the linear kernel a 
# degree element (for details about the parameters have a look at 
# the documentation of SVR)

# Note: the parameters in the grid below are already adjusted well,
# but normaly you have to adjust them a few (or many) times, until 
# you get good predictions.
# You can try to adjust them more properly than I, if you like. 

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

svr = SVR()

# setup grid
grid = [{'kernel': ['linear'],
        'C': [950, 1000, 1.15e3],
        'epsilon': [40, 50, 60]},
        {'kernel': ['rbf'],
         'C': [3e5, 3.5e5, 4e5],
         'epsilon': [40, 50, 60],
         'gamma': [0.002, 0.0025, 0.003]},
        {'kernel': ['sigmoid'],
         'C': [700, 750, 800],
         'epsilon': [0, 0.000001, 0.00001],
         'gamma': [0.4, 0.5, 0.6],
         'coef0': [-21, -20, -19]},
         {'kernel': ['poly'],
         'C': [1.2, 1.3, 1.4],
         'epsilon': [3050, 3100, 3150],
         'gamma': [0.09, 0.1, 0.11],
         'coef0': [6, 7, 8],
         'degree': [3, 4, 5]}]

# initialize GridSearchCV, note that you have to specify:
# how often cross validation should be done (cv) and a 
# scoring parameter (scoring) (look at GridSearchCV documentation
# for details)
# n_jobs = -1 just means that all available kernels should be used
gs = GridSearchCV(svr,
                 param_grid = grid,
                 scoring = 'neg_mean_absolute_error',
                 cv = 10,
                 n_jobs = -1)

# fitting Grid Search takes around 20 Minutes or so
gs.fit(X_train,y)


# In[ ]:


# check the results of our grid search and may adjust parameters in the grid
tmp = pd.DataFrame(gs.cv_results_).loc[gs.cv_results_['rank_test_score'] < 5, ['rank_test_score', 'mean_test_score', 'std_test_score',
                                                                         'param_kernel', 'param_C', 'param_epsilon', 
                                                                         'param_gamma', 'param_coef0', 'param_degree']]

# shows the different kernels seperately
# is done, because I like to adjust the parameters
# of all 4 kernels, not only for the best one
display(tmp[tmp['param_kernel'] == 'linear'])
display(tmp[tmp['param_kernel'] == 'rbf'])
display(tmp[tmp['param_kernel'] == 'sigmoid'])
display(tmp[tmp['param_kernel'] == 'poly'])
# you may have to change the number of evaluated 
# models which should be shown


# In[ ]:


# Best Parameters: (for each kernel respectively)
# Kernel:          'linear', 'rbf',    'sigmoid', 'poly'
# C:                1000,     350000,   750,       1.3
# epsilon:          50,       60,       0,         3100
# gamma:            NaN,      0.0025,   0.5,       0.1
# coef0:            NaN,      NaN,      -20,       7
# degree:           NaN,      NaN,      NaN,       4
# best_score (mae): 15471,    14010,    25293,     13713


# Now you have two Options, the simple one is to just take the best Model of our Grid Search and make predictions on the test set (beats already the 13000)
# 
# For this uncomment the following code cell and scip part 3. of this kernel

# In[ ]:


#svr_best = gs.best_estimator_

## prediction on test set
#pred_best = svr_best.predict(X_final_test)
#pred_best


# Option two is to go on with Part 3 and build an ensemble of the best Models of your Grid Search (one model for each kernel)

# # 3. Determine weights for Model Averaging

# In[ ]:


C_linear = 1000
epsilon_linear = 50

C_rbf = 350000
epsilon_rbf = 60
gamma_rbf = 0.0025

C_sig = 750
epsilon_sig = 0
gamma_sig = 0.5
coef0_sig = -20

C_poly = 1.3
epsilon_poly = 3100
gamma_poly = 0.1
coef0_poly = 7
degree_poly = 4

svm_poly = SVR(kernel = 'poly', C = C_poly, epsilon = epsilon_poly,
               gamma = gamma_poly, coef0 = coef0_poly, degree = degree_poly)
svm_rbf = SVR(kernel = 'rbf', C = C_rbf, epsilon = epsilon_rbf,
              gamma = gamma_rbf)
svm_linear = SVR(kernel = 'linear', C = C_linear, epsilon = epsilon_linear)
svm_sig = SVR(kernel = 'sigmoid', C = C_sig, epsilon = epsilon_sig,
              gamma = gamma_sig, coef0 = coef0_sig)


# In[ ]:


# split in tr and val set
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size = 0.2, random_state = 15)
print(y_tr)


# In[ ]:


# Fit Models on tr set
svm_poly.fit(X_tr, y_tr)
svm_rbf.fit(X_tr, y_tr)
svm_linear.fit(X_tr, y_tr)
svm_sig.fit(X_tr, y_tr)


# In[ ]:


# Predict on val set
pred_svm_poly = svm_poly.predict(X_val)
pred_svm_rbf = svm_rbf.predict(X_val)
pred_svm_lin = svm_linear.predict(X_val)
pred_svm_sig = svm_sig.predict(X_val)
# pred_svm_poly
# pred_svm_rbf
# pred_svm_lin
# pred_svm_sig


# In[ ]:


# Determine best weights for ensemble with Nelder-Mead-Algorithm
from scipy.optimize import minimize

def dummyfunction(x = [0.3, 0.3]):
    w0 = x[0]
    w1 = x[1]
    w2 = np.abs(1-np.sum(x))
    w3 = 0                   # gets a 0 weight, so set it 0 from beginning
    tmp_pred = (w0*pred_svm_poly + w1*pred_svm_rbf + w2*pred_svm_lin + w3*pred_svm_sig)
    return np.sum(np.abs(tmp_pred - y_val))        # use mean absolute error, can also use mean squared error

res = minimize(dummyfunction, x0 = [0.25, 0.25], method = 'Nelder-Mead', tol = 1e-6)
print(res)
1-np.sum(res.x)


# In[ ]:


# fit models on whole training data
svm_poly = SVR(kernel = 'poly', C = C_poly, epsilon = epsilon_poly,
               gamma = gamma_poly, coef0 = coef0_poly, degree = degree_poly)
svm_rbf = SVR(kernel = 'rbf', C = C_rbf, epsilon = epsilon_rbf, gamma = gamma_rbf)
svm_lin = SVR(kernel = 'linear', C = C_linear, epsilon = epsilon_linear)

svm_poly.fit(X_train, y)
svm_rbf.fit(X_train, y)
svm_lin.fit(X_train, y)

# make predictions on test set
pred_svm_poly = svm_poly.predict(X_final_test)
pred_svm_rbf = svm_rbf.predict(X_final_test)
pred_svm_lin = svm_lin.predict(X_final_test)
#pred_svm_poly
#pred_svm_rbf
#pred_svm_lin


# In[ ]:


# average preditions with determined weights
weight_svm_poly = res.x[0]
weight_svm_rbf = res.x[1]
weight_svm_lin = np.abs(1 - np.sum(res.x))
pred = (weight_svm_poly * pred_svm_poly + weight_svm_rbf * pred_svm_rbf + weight_svm_lin * pred_svm_lin)


# 4. Submit Predictions

# In[ ]:


# submit without model averaging
#output_ohne = pd.DataFrame({'Id': df_test.Id,
#                            'SalePrice': pred_best})
#output_ohne.to_csv('submission_ohne', index = False)


# In[ ]:


# submit prediction
output = pd.DataFrame({'Id': df_test.Id,
                      'SalePrice': pred})
output.to_csv('submission_mit_averaging.csv', index = False)


# In[ ]:


# check output
output

