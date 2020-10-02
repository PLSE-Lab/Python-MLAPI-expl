#!/usr/bin/env python
# coding: utf-8

# **        This kernel uses [vecstack]( https://github.com/vecxoz/vecstack) ensemble for stacking different models.       
#         I use[ scikit optimize](https://scikit-optimize.github.io/) to find the best parameters for 2nd level model (xgboost).         
#         If you don't want to use scikit optimize you can comment out the relevant portions and use your own
#         parameter values for the 2nd level model.
#         I created a train and test set based on  Olivier's 46 features for training the model. Please go through his excellent [post](https://www.kaggle.com/ogrellier/santander-46-features) for details
# **

# In[ ]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor

from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from vecstack import stacking

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os


# In[ ]:


temp = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')
test_ID = temp['ID'].values
del temp


# In[ ]:


# give the path to where the training and test set are present
train = pd.read_csv('../input/46-features-lgb/46feat_trainData.csv.gz',compression='gzip')
# train['log_target'] = np.log1p(train['target'])
test = pd.read_csv('../input/46-features-lgb/46feat_testData.csv.gz',compression='gzip')


# In[ ]:


train.head(2)


# In[ ]:


X_train, X_test = train_test_split(train, test_size=0.2, random_state=5)


# In[ ]:


y_train = X_train['log_target']
y_test = X_test['log_target']

t2 = 'log_target'
foo = train.columns
foo = list(foo)
foo.remove(t2)
colNames = foo


# In[ ]:


# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
# Initialize 1-st level models.
models = [    
    CatBoostRegressor(iterations=200,
                            learning_rate=0.03,
                            depth=4,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            random_seed=99,
                            od_type='Iter',
                            od_wait=50,
                     logging_level='Silent'),
    
    CatBoostRegressor(iterations=500,
                            learning_rate=0.06,
                            depth=3,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            random_seed=99,
                            od_type='Iter',
                            od_wait=50,
                     logging_level='Silent'),
    
    XGBRegressor(eta=0.1,reg_lambda=1,reg_alpha=10),
    
    XGBRegressor(eta=0.02,reg_lambda=1,reg_alpha=10,n_estimators=300),
    
    XGBRegressor(eta=0.002,max_depth=15,n_estimators=200),
]


# In[ ]:


# mode 1 Compute stacking features to determine optimal parameters for 2nd level model
S_train, S_test = stacking(models, X_train[colNames], y_train, X_test[colNames], 
    regression = True, metric = mean_absolute_error, n_folds = 5, 
    shuffle = True,random_state = 0, verbose = 2)


# In[ ]:


print(X_train[colNames].shape)
print(X_test[colNames].shape)
print(S_test.shape)
print(S_train.shape)


# In[ ]:


dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform',name='learning_rate')
dim_estimators = Integer(low=50, high=500,name='n_estimators')
dim_max_depth = Integer(low=1, high=6,name='max_depth')


# In[ ]:


dimensions = [dim_learning_rate,
              dim_estimators,
              dim_max_depth]


# In[ ]:


default_parameters = [1e-2,300,4]


# In[ ]:


def createModel(learning_rate,n_estimators,max_depth):       

    model = XGBRegressor(n_estimators=n_estimators,
                          learning_rate=learning_rate,
                          max_depth=max_depth,
                          random_state=0)
   # Fit 2-nd level model
    model = model.fit(S_train, y_train)
    
    # Predict
    y_pred = model.predict(S_test)

    # Final prediction score
    lv = mean_absolute_error(y_test, y_pred)
    
    return lv
    


# In[ ]:


@use_named_args(dimensions=dimensions)
def fitness(learning_rate,n_estimators,max_depth):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    n_estimators:      Number of estimators.
    max_depth:         Maximum Depth of tree.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('estimators:', n_estimators)
    print('max depth:', max_depth)


    
    lv= createModel(learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth = max_depth)
    
   # Print the rmse.
    print()
    print("Error: {}".format(lv))
    print()

    return lv


# In[ ]:


error = fitness(default_parameters)


# In[ ]:


# use only if you haven't found out the optimal parameters for xgb. else comment this block.
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=20,
                           x0=default_parameters)


# In[ ]:


plot_convergence(search_result)
plt.show()


# In[ ]:


# optimal parameters found using scikit optimize. use these parameter to initialize the 2nd level model.
search_result.x


# In[ ]:


# mode 2 Compute stacking features on the test set. 
# this mode assumes that you have already found out the best parameters using skopt for the 2nd level model
# if not then uncomment the upper block (comment this one) and find the best paramters and 
# then uncomment this block and run on the test set.

S_train, S_test = stacking(models, X_train[colNames], y_train, test[colNames], 
    regression = True, metric = mean_absolute_error, n_folds = 5, 
    shuffle = True, random_state = 0, verbose = 2)


# In[ ]:


print(test.shape)
print(X_train[colNames].shape)
print(S_test.shape)
print(S_train.shape)


# In[ ]:


# Initialize 2-nd level model

model = XGBRegressor(n_estimators=search_result.x[1],
                          learning_rate=search_result.x[0],
                          max_depth=search_result.x[2],
                          random_state=0)
    
# Fit 2-nd level model
model = model.fit(S_train, y_train)


# In[ ]:


# Predict
y_pred = model.predict(S_test)


# In[ ]:


print(y_pred.shape)


# In[ ]:


result = pd.DataFrame({'ID':test_ID
                       ,'target':np.expm1(y_pred)})

result.to_csv('46feat_stacked_ensemble_regr_models.csv', index=False)


# In[ ]:


result.tail()

