#!/usr/bin/env python
# coding: utf-8

# I attempt to develop a model that provides the probability that a certain transaction is a fraud.

# References
# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt#reducing-memory-usage
# https://www.kaggle.com/shahules/tackling-class-imbalance

# ## Load Packages, Data

# In[ ]:


import time
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Transaction CSVs
train_transaction = pd.read_csv('../input/train_transaction.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')
# Identity CSVs - These will be merged onto the transactions to create additional features
train_identity = pd.read_csv('../input/train_identity.csv')
test_identity = pd.read_csv('../input/test_identity.csv')
# Sample Submissions
ss = pd.read_csv('../input/sample_submission.csv')


# ## Examine data

# In[ ]:


train_transaction.describe()


# In[ ]:


train_identity.describe()


# In[ ]:


train = train_transaction.merge(train_identity,how='left',left_index=True,right_index=True)
y_train = train['isFraud']


# In[ ]:


X_train, X_test, y_train, y_test= train_test_split( train.drop('isFraud',axis=1), y_train, test_size=.2 , random_state=1 )


# In[ ]:


train.isna().sum()[train.isna().sum() > 0]


# ### Check if there's a class imbalance

# In[ ]:


train_transaction.isFraud.value_counts() / train_transaction.shape[0]


# ## Next Steps:
# 
# There are a few questions to answer next, especially due to the  class imbalance:
# * Do I select features before or after addressing class imbalance
# * What metrics, beyond classification, should I consider using.
# * Do I need to consider certain algorithms due to the class imbalance?
# * Do I need to do data imputation for the missing data.
#     * particularly the identity data

# ### Metrics to consider
# 
# Naturally accuracy can be a misleading metric because we can have 96% accuracy without classifying any of the fraudalent transactions correctly. I'm going to consider the F1 score which is a weighted average of the Precision and Recall metric. To refresh
# * **Precision**: Is the measure of how many that were predicted to be a class were actually the class? In this case, I would look at hte precision of the minority class, fradualent transactions and see how many of the fradualent transactions were correctly classified
# * **Recall**: A measure of how many positive values in the original dataset were classified correctly. Another way to describe this is the number of true positives divided by the number of total positives in the original dataset. 
# 
# $$ F1 Score = 2 * \frac{precision * recall}{precision + recall}$$
# 
# While in some use cases, it may make sense to prioritize precision or recall (a treatment cheap so it's better to identify things as positive to be safe instead of focusing on precision). For the case of project, I am focusing on precision and recall.

# ### How to tackle class imbalance
# There are a variety of techniques include
# * SMOTE
# * Resampling the minority class, or undersampling the majority class
# 

# There may be bias inherently in the dataset due to the fact that only certain types of transcations will be considered fraudalent and caught in retrospect. Maybe there is something to be said about transactions that are fraudalent and not caught.

# In[ ]:




