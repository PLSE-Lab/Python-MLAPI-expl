#!/usr/bin/env python
# coding: utf-8

# Apply Backward Feature Selection and Forward Feature Selection 
# to the dataset that is given in the following link:
# http://archive.ics.uci.edu/ml/datasets/madelon

# In[482]:


import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,precision_score,recall_score,f1_score
print(os.listdir("../input/"))
warnings.filterwarnings("ignore")


# In[483]:


madelon_test = '../input/madelon_test.data'
madelon_train = '../input/madelon_train.data'
madelon_train_labels = '../input/madelon_train.labels'
madelon_valid = '../input/madelon_valid.data'
madelon_valid_labels = '../input/madelon_valid.labels'


# In[484]:


train = pd.read_csv(madelon_train,delimiter=' ',header=None)
print('There are', train.shape[0], 'rows and', train.shape[1], 'columns in the dataset.')


# In[485]:


train_labels = pd.read_csv(madelon_train_labels, delimiter=' ', header=None, names=['target'])
print('There are', train_labels.shape[0], 'rows and', train_labels.shape[1], 'column in the dataset.')


# In[486]:


valid = pd.read_csv(madelon_valid,delimiter=' ',header=None)
print('There are', valid.shape[0], 'rows and', valid.shape[1], 'columns in the dataset.')


# In[487]:


valid_labels = pd.read_csv(madelon_valid_labels,delimiter=' ',header=None)
print('There are', valid_labels.shape[0], 'rows and', valid_labels.shape[1], 'columns in the dataset.')


# In[488]:


test = pd.read_csv(madelon_test,delimiter=' ',header=None)
print('There are', test.shape[0], 'rows and', test.shape[1], 'columns in the dataset.')


# In[489]:


train = train.iloc[:,:-1]
valid = valid.iloc[:,:-1]
test = test.iloc[:,:-1]


# In[490]:


knn = LogisticRegression()
forward = SequentialFeatureSelector(knn,
                                    k_features = 5,
                                    forward = True,
                                    floating = False,
                                    scoring = 'accuracy',
                                    verbose = 2,
                                    n_jobs=4,
                                    cv = 0)


# In[491]:


forward.fit(train.values,train_labels)
forward_cols = list(forward.k_feature_idx_)
knn.fit(train.loc[:,forward_cols],train_labels)


# In[492]:


print(forward_cols)


# In[493]:


pred_train_forward= knn.predict(train.loc[:,forward_cols])
print('Training accuracy on selected features: %.3f' % accuracy_score(train_labels, pred_train_forward ))
print('Training precision on selected features: %.3f' % precision_score(train_labels,pred_train_forward ,average='micro'))
print('Training recall on selected features: %.3f' % recall_score(train_labels,pred_train_forward ,average='micro'))
print('Training f1_score on selected features: %.3f' % f1_score(train_labels,pred_train_forward ,average='micro'))


# In[494]:


pred_valid_forward= knn.predict(valid.loc[:,forward_cols])
print('Validation accuracy on selected features: %.3f' % accuracy_score(valid_labels, pred_valid_forward))
print('Validation precision on selected features: %.3f' % precision_score(valid_labels,pred_valid_forward,average='micro'))
print('Validation recall on selected features: %.3f' % recall_score(valid_labels,pred_valid_forward,average='micro'))
print('Validation f1_score on selected features: %.3f' % f1_score(valid_labels,pred_valid_forward,average='micro'))


# https://www.thedynamatrix.com/machine-learning/backward-elemination

# In[495]:


import statsmodels.formula.api as sm
X = np.append(arr = np.ones((len(train.values),1)).astype(int), values = train, axis = 1)
X_opt = X[:, :]
regressor_OLS = sm.OLS(endog=train_labels, exog=X_opt).fit()
#regressor_OLS.summary()


# In[496]:


condition = lambda x: int(x.split('x')[1])
arr = regressor_OLS.pvalues
while len(arr) > 5:
    cols = list(arr.index)
    cols_arr = np.array(list(map(condition,cols[1:])))
    X_opt = X[:,cols_arr]
    regressor_OLS = sm.OLS(endog=train_labels, exog=X_opt).fit()
    arr = arr.drop(labels=arr.idxmax())               


# In[497]:


cols_app = list(map(lambda x : x -1,list(map(condition,arr.index))))
knn.fit(train.loc[:,cols_app],train_labels)


# In[498]:


cols_app


# In[499]:


pred_train_backward= knn.predict(train.loc[:,cols_app])
print('Training accuracy on selected features: %.3f' % accuracy_score(train_labels, pred_train_backward ))
print('Training precision on selected features: %.3f' % precision_score(train_labels,pred_train_backward ,average='micro'))
print('Training recall on selected features: %.3f' % recall_score(train_labels,pred_train_backward ,average='micro'))
print('Training f1_score on selected features: %.3f' % f1_score(train_labels,pred_train_backward ,average='micro'))


# In[500]:


pred_valid_backward= knn.predict(valid.loc[:,cols_app])
print('Validation accuracy on selected features: %.3f' % accuracy_score(valid_labels, pred_valid_backward ))
print('Validation precision on selected features: %.3f' % precision_score(valid_labels,pred_valid_backward ,average='micro'))
print('Validation recall on selected features: %.3f' % recall_score(valid_labels,pred_valid_backward ,average='micro'))
print('Validation f1_score on selected features: %.3f' % f1_score(valid_labels,pred_valid_backward ,average='micro'))

