#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Random permutation is needed for CatBoostEncoder to reduce leakage\ndef random_permutation(x):\n    perm = np.random.permutation(len(x)) \n    x = x.iloc[perm].reset_index(drop=True) \n    return x\n\ntrain = random_permutation(train)\ntest = random_permutation(test)\n\ntrain_ids = train.id\ntest_ids = test.id\n\ntrain.drop('id', 1, inplace=True)\ntest.drop('id', 1, inplace=True)\n\ntrain_targets = train.target\ntrain.drop('target', 1, inplace=True)")


# # Preprocessing

# Preprocessing strategy:
# 
# * For high-cardinality features (`nom_5` to `nom_9` and `ord_5`) use `CatBoostEncoder` target encoding (it performs a type of leave-one-out encoding)
# * For other features use `OneHotEncoder`

# In[ ]:


from category_encoders.cat_boost import CatBoostEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ## Missing values
# 
# Convert columns to strings and replace NAs with a non-NaN value to ensure that the one-hot encoder will treat missing values as their own class.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfor col in train.columns:\n    train[col] = train[col].astype(str)\n    train[col].fillna('NA', inplace=True)\n    test[col] = test[col].astype(str)\n    test[col].fillna('NA', inplace=True)")


# ## Encoders

# ### CatBoostEncoder for high-cardinality features

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# noms 5-9\nfor i in [5,6,7,8,9]:\n    cbe = CatBoostEncoder()\n    train[f'nom_{i}'] = cbe.fit_transform(train[f'nom_{i}'], train_targets)\n    test[f'nom_{i}'] = cbe.transform(test[f'nom_{i}'])\n\n# ord 5\ncbe = CatBoostEncoder()\ntrain['ord_5'] = cbe.fit_transform(train['ord_5'], train_targets)\ntest['ord_5'] = cbe.transform(test['ord_5'])")


# ### OneHotEncoder for other features

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nohe_cols = [\'bin_0\', \'bin_1\', \'bin_2\', \'bin_3\', \'bin_4\',\n            \'nom_0\', \'nom_1\', \'nom_2\', \'nom_3\', \'nom_4\',\n            \'ord_0\', \'ord_1\', \'ord_2\', \'ord_3\', \'ord_4\',\n            \'day\', \'month\']\n\n# ColumnTransformer enables applying OneHotEncoder to the entire dataframe\ntransformer = ColumnTransformer(\n    [\n        ("ohe",\n         OneHotEncoder(sparse=True, drop=\'first\'),\n         ohe_cols\n         )\n    ], remainder=\'passthrough\'\n)\ntrain = transformer.fit_transform(train)\ntest = transformer.fit_transform(test)')


# # Logistic Regression
# 
# I'll use LogisticRegressionCV to do automatic parameter selection, and give me a sense of cross-validated model performance.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.linear_model import LogisticRegressionCV\nclf = LogisticRegressionCV(cv=5, \n                           scoring='roc_auc', \n                           random_state=42, \n                           verbose=True, \n                           n_jobs=-1,\n                           max_iter = 1000)\nclf.fit(train, train_targets)")


# ### Model score averaged over folds

# In[ ]:


np.mean(clf.scores_[1])


# # Prepare submission

# In[ ]:


preds = clf.predict_proba(test)[:, 1]
preds = pd.DataFrame(list(zip(test_ids, preds)), columns = ['id', 'target'])
preds.sort_values(by=['id'], inplace = True)

preds.to_csv("./my_submission.csv", index=False)

