#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly_express as px

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


plt.style.use('ggplot')
g = sns.countplot(x="target", data=data, palette="bwr")
sns.despine()
g.figure.set_size_inches(12,7)
plt.show()


# # Splitting the data into Cross-Validation and Train

# In[ ]:


X = data.iloc[:,:-1]
y = data['target']
X.shape


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)


# # Running Hyperopt

# ## 1. Importing the Required Library for Hyperopt

# In[ ]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


# ## 2. Create the objective function
# 
# Here we create an objective function which takes as input a hyperparameter space:
# - defines a classifier, in this case XGBoost. Just try to see how we take the parameters from the space. For example `space['max_depth']` 
# - We fit the classifier to the train data
# - We predict on cross validation set
# - We calculate the required metric we want to maximize or minimize
# - Since we only minimize using `fmin` in hyperopt, if we want to minimize `logloss` we just send our metric as is. If we want to maximize accuracy we will try to minimize `-accuracy`

# In[ ]:


from sklearn.metrics import accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np
def objective(space):
    # Instantiate the classifier
    clf = XGBClassifier(n_estimators =1000,colsample_bytree=space['colsample_bytree'],
                           learning_rate = .3,
                            max_depth = int(space['max_depth']),
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            gamma = space['gamma'],
                           reg_lambda = space['reg_lambda'])
    
    eval_set  = [( X_train, y_train), ( X_test, y_test)]
    
    # Fit the classsifier
    clf.fit(X_train, y_train,
            eval_set=eval_set, early_stopping_rounds=10,verbose=False)
    
    # Predict on Cross Validation data
    pred = clf.predict(X_test)
    
    # Calculate our Metric - accuracy
    accuracy = accuracy_score(y_test, pred>0.5)

    # return needs to be in this below format. We use negative of accuracy since we want to maximize it.
    return {'loss': -accuracy, 'status': STATUS_OK }


# ## 3. Create the Space for your classifier
# 
# Now, we create the search space for hyperparameters for our classifier.
# 
# To do this we end up using many of hyperopt built in functions which define verious distributions. As you can see we use uniform distribution between 0.7 and 1 for our subsample hyperparameter. It is much better than defining a parameter value using ranges for sure. You can also define a lot of other distributions too. 
# 
# 

# In[ ]:


space ={'max_depth': hp.quniform("x_max_depth", 4, 16, 1),
        'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
        'subsample': hp.uniform ('x_subsample', 0.7, 1),
        'gamma' : hp.uniform ('x_gamma', 0.1,0.5),
        'colsample_bytree' : hp.uniform ('x_colsample_bytree', 0.7,1),
        'reg_lambda' : hp.uniform ('x_reg_lambda', 0,1)
    }


# ## 4. Run Hyperopt

# In[ ]:


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
print(best)


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score

clf = XGBClassifier(x_colsample_bytree= 0.8743861143035889, x_gamma= 0.15403994099351054, 
                         x_max_depth= 7.0, x_min_child =5.0, x_reg_lambda= 0.015889530822374764, 
                         x_subsample= 0.7716293823039047)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print('Confusion Matrix :\n',pd.crosstab(y_test,y_pred))

print('Accuracy Socre :',accuracy_score(y_test,y_pred))

print('\nClassification Report :\n',classification_report(y_test,y_pred))


# ### Selecting model using TPOT
# <p><a href="https://github.com/EpistasisLab/tpot">TPOT</a> is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.</p>
# <p><img src="https://assets.datacamp.com/production/project_646/img/tpot-ml-pipeline.png" alt="TPOT Machine Learning Pipeline"></p>
# <p>TPOT will automatically explore hundreds of possible pipelines to find the best one for our dataset. Note, the outcome of this search will be a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">scikit-learn pipeline</a>, meaning it will include any pre-processing steps as well as the model.</p>
# <p>We are using TPOT to help us zero in on one model that we can then explore and optimize further.</p>

# In[ ]:


# Import TPOTClassifier and roc_auc_score
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Instantiate TPOTClassifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}.{transform}')


# In[ ]:


# Importing modules
from sklearn.linear_model import LogisticRegression

# Instantiate LogisticRegression
logreg = LogisticRegression(C=10.0, class_weight=None,
                                         dual=False, fit_intercept=True,
                                         intercept_scaling=1,
                                         l1_ratio=None, max_iter=100,
                                         multi_class='warn', n_jobs=None,
                                         penalty='l2', random_state=None,
                                         solver='warn', tol=0.0001,
                                         verbose=0, warm_start=False)

# Train the model
logreg.fit(X_train, y_train)

# AUC score for tpot model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')


# In[ ]:


y_pred = logreg.predict(X_test)
print('Accuracy score :',accuracy_score(y_pred,y_test))


# In[ ]:




