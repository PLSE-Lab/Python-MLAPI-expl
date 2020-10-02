#!/usr/bin/env python
# coding: utf-8

# ## Notebook Summary:

# ForestML: Part 1: EDA and Class Imbalance Handling
# 
# I ran through an almost complete cylce of making an ML model. But I realised that my Test accuracy became stagnant at ~0.801 +/- 0.002.
# In this Notebook I will systematically try to solve the class imbalance problem and redo all my model preparation from scratch.
# 
# 

# ## Record of notebooks I have worked on till now.

# https://www.kaggle.com/phsheth/preliminary-eda-feature-importance
# 
# https://www.kaggle.com/phsheth/ensemble-sequential-backward-selection
# 
# https://www.kaggle.com/phsheth/forestml-eda-and-stacking-evaluation-v2
# 
# https://www.kaggle.com/phsheth/forestml-part-2-feature-engg-random-forest <- Inducting Random Forest
# 
# https://www.kaggle.com/phsheth/forestml-part-3-feature-engg-xgboost <- Inducting XGBoost
# 
# https://www.kaggle.com/phsheth/forestml-part-4-feature-engg-extratrees < Inducting ExtraTrees
# 
# https://www.kaggle.com/phsheth/forestml-part-4-feature-engg-adaboost <- Inducting Adaboost
#  
# https://www.kaggle.com/phsheth/forestml-part-4-feature-engg-bagging-classifier <- Bagging did not work well
# 
# https://www.kaggle.com/phsheth/forestml-part-4-feature-engg-catboost <- CatBoost did not work well
# 
# https://www.kaggle.com/phsheth/forestml-part-6-stacking-eval-selected-fets-2 <- Using Borrowed Parameters
# 
# 
# https://www.kaggle.com/phsheth/forestml-part-7-stacking-selfets-pre-hp-tuning <- Removed Borrowed Parameters before HyperParameter Tuning
# 
# <Show kernels for HyperParameter Tuning
# 
# 
# https://www.kaggle.com/phsheth/forestml-part-6-stacking-eval-selfets-gmix <- Reference for Effect of Gmix
# 
# https://www.kaggle.com/phsheth/forestml-part-9-stacking-selfets-post-hp-tune <- Using Random Forest Tuned HyperParameters
# 
# https://www.kaggle.com/phsheth/forestml-part-9-1-stacking-selfets-rf-xgb <- Using Random Forest and XGBoost Tuned HyperParameters

# Test Result with Smoting: 0.7894
# https://www.kaggle.com/phsheth/forestml-part-1-eda-and-class-imbalance-handling/edit/run/20481229
# 
# Test Result whout Smoting: this run.

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


# ## Import required libraries

# In[ ]:




import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# In[ ]:


## Backup Imports
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import catboost as cb
import lightgbm as lgb

from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline


# In[ ]:





# ## Import the Raw Data

# In[ ]:


train = pd.read_csv("/kaggle/input/learn-together/train.csv")
test = pd.read_csv("/kaggle/input/learn-together/test.csv")


# ## Define X and y (features and labels respectively)

# In[ ]:


# Remove the Labels and make them y
y = train['Cover_Type']

# Remove label from Train set
X = train.drop(['Cover_Type'],axis=1)

# Rename test to text_X
test_X = test



# split data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

X = X.drop(['Id'], axis = 1)
train_X = train_X.drop(['Id'], axis = 1)
val_X = val_X.drop(['Id'], axis = 1)
test_X = test_X.drop(['Id'], axis = 1)


# ## View the dataframes

# In[ ]:


train_X.describe()


# In[ ]:


val_X.describe()


# In[ ]:


test_X.describe()


# ### Lets try Without Smoting

# In[ ]:



rfcfin = RandomForestClassifier(n_estimators = int(1631.3630739649345),
                                min_samples_split = int(2.4671165024828747),
                                min_samples_leaf = int(1.4052032266878376),
                                max_features = 0.23657708614689418,
                                max_depth = int(426.8410655510125),
                                bootstrap = int(0.8070235824535138),
                                random_state=42)
rfcfin.fit(X, y.ravel())


# In[ ]:


test_ids = test["Id"]
test_pred = rfcfin.predict(test_X.values)


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'Id': test_ids,
                       'Cover_Type': test_pred})
output.to_csv('submission.csv', index=False)


# In[ ]:




