#!/usr/bin/env python
# coding: utf-8

# First of all I have used ideas from these websites: 
# 
# https://www.statisticshowto.datasciencecentral.com/lasso-regression/
# 
# 
# In this kernel we are going to use Lasso regression which is a type of linear regression that produces [sparse models](https://www.quora.com/Why-need-to-find-sparse-models-in-machine-learning). 
# 
# Lasso regression performs L1 regularization, which adds a penalty equal to the absolute value of the magnitude of the coefficients. 

# In[1]:


# Loading the packages
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer 
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Loading the training dataset
df_train = pd.read_csv("../input/train.csv")


# In[3]:


y = df_train["target"]
# We exclude the target and id columns from the training dataset
df_train.pop("target");
df_train.pop("id")
colnames1 = df_train.columns


# We are going to standardize the explanatory variables by removing the mean and scaling to unit variance. The reason for that is to help convergence of the technique used in the optimization. 

# In[4]:


scaler = StandardScaler()
scaler.fit(df_train)
X = scaler.transform(df_train)
df_train = pd.DataFrame(data = X, columns=colnames1)


# We are going to perform a grid search in order to find a good value for the hyperparameters $\lambda$ of Lasso regression. The following web page is a good reference: 
# 
# [Tuning ML Hyperparameters](https://alfurka.github.io/2018-11-18-grid-search/)

# In[5]:


# Find best hyperparameters (roc_auc)
random_state = 0
clf = LogisticRegression(random_state = random_state)
param_grid = {'class_weight' : ['balanced'], 
              'penalty' : ['l1'],  
              'C' : [0.0001, 0.0005, 0.001, 
                     0.005, 0.01, 0.05, 0.1, 0.5, 1, 
                     10, 100, 1000, 1500, 2000, 2500, 
                     2600, 2700, 2800, 2900, 3000, 3100, 3200  
                     ] , # This hyperparameter is lambda 
              'max_iter' : [100, 1000, 2000, 5000, 10000] }

# Make an roc_auc scoring object using make_scorer()
scorer = make_scorer(roc_auc_score)

grid = GridSearchCV(estimator = clf, param_grid = param_grid , 
                    scoring = scorer, verbose = 1, cv=20,
                    n_jobs = -1)


X = df_train.values

grid.fit(X,y)

print("Best Score:" + str(grid.best_score_))

best_parameters = grid.best_params_


# In[6]:


# We are going to print the hyperparameters of the best model 
best_clf = grid.best_estimator_
print(best_clf)


# In[7]:


model = LogisticRegression(C=0.1, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='warn', n_jobs=None, penalty='l1', random_state=0,
          solver='warn', tol=0.0001, verbose=0, warm_start=False);

model.fit(X, y);


# Finally, we are going to generate the submission file. 

# In[8]:


df_test = pd.read_csv("../input/test.csv")
df_test.pop("id");
X = df_test 
X = scaler.transform(X)
df_test = pd.DataFrame(data = X, columns=colnames1)  
X = df_test.values

y_pred = model.predict_proba(X)
y_pred = y_pred[:,1]  


# In[9]:


# submit prediction
smpsb_df = pd.read_csv("../input/sample_submission.csv")
smpsb_df["target"] = y_pred
smpsb_df.to_csv("submission.csv", index=None)

