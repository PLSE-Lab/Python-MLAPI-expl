#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:




# Import required librarues

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

# Import svm, multilinear regression, decision tree and xgboost

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score


# In[ ]:





# In[ ]:


train = pd.read_csv("/kaggle/input/learn-together/train.csv")
test = pd.read_csv("/kaggle/input/learn-together/test.csv")


# In[ ]:


# get target
y = train['Cover_Type']

# get features (TODO feature extraction)
X = train.drop(['Cover_Type'],axis=1)
test_X = test



# split data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.0001, random_state=42)

train_X = train_X.drop(['Id'], axis = 1)
val_X = val_X.drop(['Id'], axis = 1)
test_X = test_X.drop(['Id'], axis = 1)


# In[ ]:


train_X.describe()


# In[ ]:


val_X.describe()


# In[ ]:


test_X.describe()


# In[ ]:


scaler_train = RobustScaler(quantile_range = (25, 75))
train_X = pd.DataFrame(scaler_train.fit_transform(train_X), index = train_X.index, columns = train_X.columns)
train_X.describe()


# In[ ]:


scaler_val = RobustScaler(quantile_range = (25, 75))
val_X = pd.DataFrame(scaler_train.fit_transform(val_X), index = val_X.index, columns = val_X.columns)
val_X.describe()


# In[ ]:


scaler_test = RobustScaler(quantile_range = (25, 75))
test_X = pd.DataFrame(scaler_train.fit_transform(test_X), index = test_X.index, columns = test_X.columns)
test_X.describe()


# In[ ]:


len(train_X.columns), len(val_X.columns), len(val_X.columns)


# In[ ]:


len(train_X.index), len(val_X.index), len(test_X.index)


# ### Feature Selection using Sequential Backward Selection

# In[ ]:





# In[ ]:


### define the classifiers


classifier_rf = RandomForestClassifier(n_estimators = 400, min_samples_split = 2,
                                           min_samples_leaf = 1, max_features = 'sqrt',
                                           bootstrap = False, random_state=42)


# In[ ]:




efs_eclf = EFS(classifier_rf,
                    min_features=25,
                    max_features=35,
                    scoring = 'accuracy',
                    print_progress = True,
                    cv = 5, 
                    n_jobs = -1)
efs_eclf = efs_eclf.fit(train_X, train_y.values.ravel())


print('best combination (ACC: %.3f): %s\n' % (efs_eclf.k_score_, efs_eclf.k_feature_idx_))
print('all subsets:\n', efs_eclf.subsets_)
plot_sfs(efs_eclf.get_metric_dict(), kind='std_err');


# In[ ]:


train_X_sbs = classifier_rf.transform(train_X)
test_X_sbs = classifier_rf.transform(test_X)


# In[ ]:


eclf.fit(train_X_sbs.values, train_y.values.ravel())


# In[ ]:


test_ids = test["Id"]
test_pred = classifier_rf.predict(test_X_sbs.values)
test_pred


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'Id': test_ids,
                       'Cover_Type': test_pred})
output.to_csv('submission.csv', index=False)
