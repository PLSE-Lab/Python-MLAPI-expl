#!/usr/bin/env python
# coding: utf-8

# We have applied cross validation using different techniques:
# - **LogisticRegressionCV**
# - **Custom cross validation**
# - **cross_val_score**
# - **GridSearchCV**
# - **RandomSearchCV**
# 
# In all cases the evaluated model (Logistic Regression) has a not very big variance (the score is very similar in the different folds)

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# In[2]:


data_train = pd.read_csv('../input/train.csv')


# In[3]:


data_train.head()


# In[4]:


data_train.shape


# In[5]:


data_train.isnull().sum().any()


# In[6]:


data_train['target'].value_counts()


# In[7]:


X = data_train.drop(['ID_code', 'target'], axis=1)
y = data_train['target'].values


# In[8]:


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Let's check if pca can help:

# In[9]:


scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(0.99, random_state=1)
pca.fit(X_scaled)


# In[10]:


pca.explained_variance_ratio_[:198].sum()


# We need a lot of features to explain most of the variance, so pca looks like no very useful here:

# In[11]:


pca.n_components_


# We are going to split our data into a training and testing sets:

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, 
                                                    y, 
                                                    test_size=0.1, 
                                                    stratify=y, 
                                                    random_state=1)


# ## Cross validation with LogisticRegressionCV
# This method is specific for LogisticRegression, although other models have also their own "CV" version. This method let's explore differnt values of hyperparameters at the same time.

# In[13]:


kfolds = StratifiedShuffleSplit(n_splits=5, random_state=1)


# In[14]:


model_lr = LogisticRegressionCV(Cs=10, cv=kfolds, random_state=1, class_weight='balanced', scoring='roc_auc')


# In[15]:


model_lr.fit(X_train, y_train)


# In[16]:


model_lr.score(X_test, y_test)


# In[17]:


model_lr.C_


# ## Custom cross validation
# At the same time that we create each model in the cross validation process we evaluate it on the validation data and create predictions for the testing set. We will ensamble those predictions and we'll evaluate them.

# In[18]:


from tqdm import tqdm
from sklearn.metrics import roc_auc_score

#model_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1)
model_lr = LogisticRegression(C=0.0001, solver='lbfgs', random_state=1)
model = model_lr
scores = []
scores_xtest = []
preds_xtest = np.empty((X_test.shape[0], 5))
for i, (id_train, id_val) in tqdm(enumerate(kfolds.split(X_train, y_train))):
    model.fit(X_train[id_train], y_train[id_train])
    pred_i = model.predict_proba(X_train[id_val])[:, 1]
    score_i = roc_auc_score(y_train[id_val], pred_i)
    scores.append(score_i)
    pred_xtest = model.predict_proba(X_test)[:, 1]
    preds_xtest[:, i] = pred_xtest
    score_xtest = roc_auc_score(y_test, pred_xtest)
    scores_xtest.append(score_xtest)


# In[19]:


print(scores)


# In[20]:


print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(scores), np.std(scores)))


# If we obtain the mean of the scores of all build models during the cross validation, on the testing data, we have:

# In[21]:


print("Mean test auc: {:.6f} +/- {:.6f}".format(np.mean(scores_xtest), np.std(scores_xtest))) 


# This is quite similar to what we obtain if we get a mean ensamble of all of the predictions on the testing data and we obtain the score:

# In[22]:


mean_preds_test = preds_xtest.mean(axis=1)


# In[23]:


roc_auc_score(y_test, mean_preds_test)


# Other alternative is to build a model using all the training data and apply it on the testing data (the result in this case is quite similar):

# In[24]:


model.fit(X_train, y_train)


# In[25]:


full_preds_test = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, full_preds_test)


# ## Cross validation with cross_val_score

# In[26]:


cv_scores = cross_val_score(model_lr, X_train, y_train, scoring='roc_auc', cv=kfolds, n_jobs=-1)


# In[27]:


cv_scores


# In[28]:


print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))


# ## GridSearchCV
# This helps also to explore the hyperparameter space

# In[29]:


from sklearn.model_selection import GridSearchCV

model = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=1)
param_grid = {'C': np.logspace(-4, 4, base=10, num=4)}
grid = GridSearchCV(model, cv=kfolds, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')


# In[30]:


grid.fit(X_train, y_train)


# In[31]:


grid.best_params_


# In[32]:


grid.best_score_


# In[33]:


cv_results = list(zip(grid.cv_results_['mean_test_score'], 
                      grid.cv_results_['std_test_score']))

for i, param in enumerate(grid.cv_results_['params']):
    print("C: {:.4f} => mean validation auc: {:.4f} +/- {:.4f}"
          .format(param['C'], cv_results[i][0], cv_results[i][1]))


# In[34]:


grid.best_estimator_


# Logistic Regression score method return the mean accuracy:

# In[35]:


grid.best_estimator_.score(X_test, y_test)


# So, if we want to get the auc metric we can do:

# In[36]:


grid.score(X_test, y_test)


# The grid object can be used as an estimator because by default the refit param is True. So we can have predicions directly:

# In[37]:


(grid.best_estimator_.predict(X_test) != grid.predict(X_test)).sum()


# ## RandomSearchCV
# This chooses randomly different n_iter groups of hyperparameters from the defined space and try their performance with cross validation

# In[47]:


from sklearn.model_selection import RandomizedSearchCV

model = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=1)
param_grid = {'C': np.logspace(-4, 4, base=10, num=100)}
grid_random = RandomizedSearchCV(model, cv=kfolds, n_iter=4, 
                                 param_distributions=param_grid, n_jobs=-1, 
                                 random_state=1, scoring='roc_auc')


# In[48]:


grid_random.fit(X_train, y_train)


# In[49]:


grid_random.best_params_


# In[52]:


grid_random.best_score_


# In[50]:


grid_random.cv_results_['params']


# In[53]:


cv_results = list(zip(grid_random.cv_results_['mean_test_score'], 
                      grid_random.cv_results_['std_test_score']))

for i, param in enumerate(grid_random.cv_results_['params']):
    print("C: {:.4f} => mean validation auc: {:.4f} +/- {:.4f}"
          .format(param['C'], cv_results[i][0], cv_results[i][1]))


# In[ ]:




