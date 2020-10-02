#!/usr/bin/env python
# coding: utf-8

# # How To Use Bayesian Optimization
# In this kernel I'll try to demonstrate how easy it is to use 'hyperopt' to do hyperparams search using bayesian optimization.
# This is not supposed to be an in-depth tutorial but more a simple notebook to show how to use this great searching method.
# 
# 
# I intentionally won't do any EDA or feature extraction from the data.
# I'll do a simple one hot encoding to categorical features and run a model !
# 
# 
# #### We actually need only 2 things:
# 1. The parameters' values space to search
# 2. An objective function to minimize

# In[45]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import category_encoders
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn import metrics
from hyperopt import hp, tpe, fmin, space_eval
import os

from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.cluster import FeatureAgglomeration

np.random.seed(123)


# ### Load the train data and the test data:

# In[46]:


train = pd.read_csv(os.path.join('..', 'input', 'train.csv'), index_col='ID')
train.head()


# In[47]:


test = pd.read_csv(os.path.join('..', 'input', 'test.csv'), index_col='ID')
test.head()


# In[48]:


train.info()


# In[49]:


def test_model(x_train, x_test, y_train, y_test, model):
    """ fit the model and print the train and test result """
    np.random.seed(1)
    model.fit(x_train, y_train)
    print('train score: ', model.score(x_train, y_train))
    print('test score: ', model.score(x_test, y_test))


# In[50]:


# Split to X and y and then to train and test sets:
X = train.drop('y', axis=1)
y = train['y']
x_train, x_test, y_train, y_test = train_test_split(X, y)


# One hot encoding to the categorical columns in the data:

# In[ ]:


# One hot encoding to the categorical columns in the data:
one_hot = category_encoders.OneHotEncoder(cols=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'], drop_invariant=True, use_cat_names=True)
x_train_one_hot = one_hot.fit_transform(x_train)
x_test_one_hot = one_hot.transform(x_test)


# #### Test a first inilized model for a baseline

# In[ ]:


test_model(x_train_one_hot, x_test_one_hot, y_train, y_test, model=SVR())


# Our init model is not bad, lets do a simple hyperparams search

# In[ ]:


def get_model(args):
    """Construct the mode based on the args choosen in the current step of the bayesian optimization process"""
    feature_selector = args['selection']
        
    model = Pipeline([
        ('scaler', args['scaler']()),
        ('selection', feature_selector['selection_algo'](**feature_selector['selection_params'])),
        ('clf', args['clf'](**args['clf_params']))
    ])

    return model


# In[ ]:


def objective_func(args, x_train=x_train_one_hot, y_train=y_train):
    """
    Run a cross validation on the train data and return the mean test score.
    This function output will be value the bayesian optimization process will try to minimize.
    """
    np.random.seed(123)
    model = get_model(args)

    cv_results = cross_validate(estimator=model, X=x_train, y=y_train, n_jobs=-1, scoring='r2',
                                cv=KFold(n_splits=4))
    return - cv_results['test_score'].mean() # minus is because we optimize to the minimum


# #### A few notes about the search space:
# - You need to specify for each parameter it's distribution.<br/>I offen user uniformal distribution if I'm not sure which is the right distribution (**Do you know a better way? I'll be happy to learn, please leave a comment**
# - I'm considering the choise of which data scaler to use as a hyperparameter
# - I assume I need some feature selection but I'm not sure which method will be the best.<br/>So I have three different methods which have different params and all this will be considered as hyperparam as well.
# - There is more options and maybe better models to use ..

# In[ ]:


search_space = {
    'scaler': hp.choice('scaler', [StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler]),
    'selection':  hp.choice('selection',[
        {
        'selection_algo': SelectKBest,
        'selection_params': 
            {
            'k': hp.choice('k', ['all'] + list(range(1, x_train_one_hot.shape[1]))),
            'score_func': hp.choice('score_func', [f_regression, mutual_info_regression])
            }
        },
        {
            'selection_algo': PCA,
            'selection_params': {'n_components': hp.uniformint('n_components', 1, x_train_one_hot.shape[1])}
        },
        {
            'selection_algo': FeatureAgglomeration,
            'selection_params': {'n_clusters': hp.uniformint('n_clusters', 1, x_train_one_hot.shape[1])}
        }
    ]),

    'clf': SVR,
    'clf_params': 
        {
            'kernel': hp.choice('kernel', ['rbf', 'poly', 'linear']),
            'C': hp. uniform('C', 0.0001, 30)
        }

}


# In[ ]:


np.random.seed(123)
best_space = fmin(objective_func, space=search_space, algo=tpe.suggest, max_evals=100)
best_model =  get_model(space_eval(search_space, best_space))
print(best_model)


# In[ ]:


space_eval(search_space, best_space)


# In[ ]:


test_model(x_train_one_hot, x_test_one_hot, y_train, y_test, model=best_model)


# **Great** improvement only by searching some hyperparms (100 evaluations, which in my opinion is a low amount) .
# 
# Of course a simple grid search would find the same params as well and if you are any lucky even random search would. But it would be a question of running time.<br/>
# I believe that this bayesian way improves the random searching and offers a bit better searching method.

# In[ ]:


# Run on the real test
# X_one_hot = one_hot.fit_transform(X)
# test_one_hot = one_hot.transform(test)

# best_model.fit(X_one_hot, y)
# pd.DataFrame({'ID':test.index, 'y': best_model.predict(test_one_hot)}).to_csv(r'subs.csv', index=False)

