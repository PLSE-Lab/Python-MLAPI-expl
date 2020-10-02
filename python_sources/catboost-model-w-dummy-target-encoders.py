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

# ## Mapping values
# 
# For the binary and ordinal variables I will simply use a value mapping approach.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# bin variables\nbin_recode = {0: 0, 1: 1, 'F':0, 'T':1, 'N':0, 'Y':1}\nfor i in range(5):\n    train[f'bin_{i}'] = train[f'bin_{i}'].map(bin_recode)\n    test[f'bin_{i}'] = test[f'bin_{i}'].map(bin_recode)\n\n# ord_1\nlevels = { 'Novice':0, 'Contributor':1, \n          'Expert':2, 'Master':3, 'Grandmaster':4 }\ntrain['ord_1'] = train['ord_1'].map(levels)\ntest['ord_1'] = test['ord_1'].map(levels)\n\n# ord_2\ntemps = { 'Freezing':0, 'Cold':1, 'Warm':2, 'Hot':3, \n         'Boiling Hot':4, 'Lava Hot':5 }\ntrain['ord_2'] = train['ord_2'].map(temps)\ntest['ord_2'] = test['ord_2'].map(temps)\n\n# ord_3\nlowercase_letters = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,\n                     'g':7,'h':8,'i':9,'j':10,'k':11,\n                     'l':12,'m':13,'n':14,'o':15,'p':16,\n                     'q':17,'r':18,'s':19,'t':20,'u':21,\n                     'v':22,'w':23,'x':24,'y':25,'z':26}\ntrain['ord_3'] = train['ord_3'].map(lowercase_letters)\ntest['ord_3'] = test['ord_3'].map(lowercase_letters)\n\n# ord_4\nuppercase_letters = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,\n                     'G':7,'H':8,'I':9,'J':10,'K':11,\n                     'L':12,'M':13,'N':14,'O':15,'P':16,\n                     'Q':17,'R':18,'S':19,'T':20,'U':21,\n                     'V':22,'W':23,'X':24,'Y':25,'Z':26}\ntrain['ord_4'] = train['ord_4'].map(uppercase_letters)\ntest['ord_4'] = test['ord_4'].map(uppercase_letters)")


# ## Dummy coding
# 
# For `nom_0` to `nom_4` I will use dummy coding.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nnoms_0_4 = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']\ntrain = pd.get_dummies(train, \n                       columns = noms_0_4, \n                       prefix = noms_0_4,\n                       drop_first=True,\n                       sparse=True, \n                       dtype=np.int8)\n\ntest = pd.get_dummies(test, \n                      columns = noms_0_4, \n                      prefix = noms_0_4,\n                      drop_first=True,\n                      sparse=True, \n                      dtype=np.int8)")


# ## Target encoding
# 
# `nom_5` to `nom_9` and `ord_5` are high-cardinality features. One way to handle these features is to use target encoding. However, there are different ways of doing target encoding and some are better at avoiding leakage / overfitting. Here I use the CatBoostEncoder which implements a leave-one-out strategy of target encoding.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom category_encoders.cat_boost import CatBoostEncoder\n\n# noms 5-9\nfor i in [5,6,7,8,9]:\n    cbe = CatBoostEncoder()\n    train[f'nom_{i}'] = cbe.fit_transform(train[f'nom_{i}'], train_targets)\n    test[f'nom_{i}'] = cbe.transform(test[f'nom_{i}'])\n\n# ord 5\ncbe = CatBoostEncoder()\ntrain['ord_5'] = cbe.fit_transform(train['ord_5'], train_targets)\ntest['ord_5'] = cbe.transform(test['ord_5'])")


# In[ ]:


print(train.shape)
print(test.shape)


# # CatBoost Classifier

# I used `grid_search()` to identify the best parameters for the `CatBoostClassifier`.

# In[ ]:


#grid = {'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3],
#        'depth': [3, 4, 5, 6],
#        'l2_leaf_reg': [1, 2, 3, 4, 5, 6, 7]}

#grid_search_result = cb.grid_search(grid, 
#                                    X=train_cbe, 
#                                    y=train_targets, 
#                                    plot=True)


# The model below uses the best parameters that I found.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom catboost import CatBoostClassifier\ncb = CatBoostClassifier(eval_metric='AUC',\n                        learning_rate=0.1,\n                        depth=3,\n                        l2_leaf_reg=5)\n\ncb.fit(train, train_targets, verbose=False)")


# # Prepare submission

# In[ ]:


preds = cb.predict_proba(test)[:, 1]
preds_df = pd.DataFrame(list(zip(test_ids, preds)), 
                        columns = ['id', 'target'])
preds_df.sort_values(by=['id'], inplace = True)

preds_df.to_csv("./submission.csv", index=False)


# In[ ]:


preds_df.head()

