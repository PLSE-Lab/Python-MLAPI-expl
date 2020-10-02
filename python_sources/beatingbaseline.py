#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


#Define data
train = pd.read_csv('/kaggle/input/elte-photometric-redshift-estimation-2020/train.csv')
test = pd.read_csv('/kaggle/input/elte-photometric-redshift-estimation-2020/testX.csv')
sample = pd.read_csv('/kaggle/input/elte-photometric-redshift-estimation-2020/sample.csv')


# ### Brief explanation of data:
#  - Target column (variable): ```train.redshift```, can be found in train data.
#  - Column ```test.ID``` in test data actually is replication of the index of DataFrame length and can be ignored for a while. We can add it in submission made dublicating index value.
#  
# ### Beating baselines
#  - Basically we have usual ```Regression``` task, cause our target value is continuous. So we can firstly try to deploy all the standart algorithms for Regression.  
#  
# ### Algorithms. Simple Ones.
# 
# 1. SVM 
# 2. Stochastic Gradient Descent
# 3. KNN
# 4. Dicision Trees

# ### BaseLine
# 
# The main part is ready as well as parameters dictionary.<br>
# The ```output``` will return list of 4 values based on 4 best param_grid of algorithms. One of them will perform better than others. <br>
# Actually ```this baseline can be freely used by you```. In days I will finish rest part with output and transform_to_csv.<br> 
# 
# <strong>Be considerable the calculations can take time, especially with full data capacity. To decrease it, customize params_dict with less hyper parameters</strong>
# 
# <h3>Happy Coding</h3>

# In[ ]:


import logging 

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


class BasePipeLine:
    def __init__(self, train, test, sample):
        self.train = train
        self.test = test
        self.sample = sample
        self.KNNR = KNeighborsRegressor()
        self.DTR = DecisionTreeRegressor()
        self.SVR = SVR()
        self.SGDR = SGDRegressor()
        self.algorithms = [self.KNNR, self.DTR, self.SVR, self.SGDR]
#         self.indexes = [alg.__class__.__name__ for alg in self.algorithms]
#         self.columns = ['score', 'best_params']
        self.params_dict = {
            'KNeighborsRegressor': {'n_neighbors':list(range(3, 11))},
            'SVR' : {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.3]},
            'DecisionTreeRegressor' : {"min_samples_split": [10, 20], "max_depth": [2, 6]},
            'SGDRegressor': {'alpha': 10.0 ** -np.arange(1, 7), 'penalty': ['l2', 'l1', 'elasticnet'], 'learning_rate': ['constant', 'optimal', 'invscaling']}
        }
            
            
    def data_transform(self):
        X = self.train.drop(['redshift'], axis=1)
        y = self.train['redshift']
        X_test = self.test.drop(['ID'], axis=1)
        y_test = self.sample.drop(['ID'], axis=1).values
        return X, y, X_test, y_test
    
    
    def grid_search_cv(self, X_train, y_train):
        best_models = [] 
        for algorithm in self.algorithms:
            clf = GridSearchCV(estimator=algorithm, param_grid=self.params_dict['{}'.format(algorithm.__class__.__name__)], cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
            clf.fit(X_train, y_train)
            best_models.append(clf.best_estimator_)
        return best_models
    
    
    def make_prediction(self, best_models, X_test, y_test):
        evaluation = []
        for model in best_models:
            predictions = model.predict(X_test)
            evaluation.append(mean_squared_error(y_test, predictions))
        return evaluation
        #return predictions 
    
    
    def transform_to_submission_csv(self, predictions):
        submission_df = pd.DataFrame(columns=['ID', 'redshift'])
        submission_df['redshift'] = predictions
        submission_df['ID'] = submission_df['redshift'].index
        submission_df.to_csv('/kaggle/working/submission.csv', index = False)
    
    
    def run(self):
        X, y, X_test, y_test = BasePipeLine(train, test, sample).data_transform()
        logging.info('Transformation finished')
        
        best_models = BasePipeLine.grid_search_cv(self, X, y)
        logging.info('Fitting models finished')
        
        BasePipeLine.make_prediction(best_models, X_test, y_test)
        #predictions = BasePipeLine.make_prediction(best_model, X_test, y_test)
        logging.info('Evaluating Finished')
        
#         BasePipeLine.transform_to_submission_csv(predictions)
#         logging.info('BasePipeLine Calculations Finished')


# In[ ]:


#BasePipeLine(train[:100], test[:100], sample[:100]).run()

