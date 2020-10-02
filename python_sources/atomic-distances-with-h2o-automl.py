#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GroupKFold

import os
print(os.listdir("../input"))


# In[20]:


train = pd.read_csv('../input/train.csv', index_col='id')
test = pd.read_csv('../input/test.csv', index_col='id')


# In[21]:


train.head()


# In[32]:


train.describe()


# In[22]:


train.shape


# In[23]:


test.shape


# In[24]:


structures = pd.read_csv('../input/structures.csv')
display(structures.head())


# In[25]:


# molecule level EDA + stats
print("unique molecules",structures["atom"].nunique())


# In[28]:


val = 1
print(f"{val+1}")


# In[29]:


# Map the atom structure data into train and test files

def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)


# In[30]:


train.head()


# In[31]:


# %%time
# # Engineer a single feature: distance vector between atoms
# #  (there's ways to speed this up!)

# def dist(row):
#     return ( (row['x_1'] - row['x_0'])**2 +
#              (row['y_1'] - row['y_0'])**2 +
#              (row['z_1'] - row['z_0'])**2 ) ** 0.5

# train['dist'] = train.apply(lambda x: dist(x), axis=1)
# test['dist'] = test.apply(lambda x: dist(x), axis=1)


# In[33]:


get_ipython().run_cell_magic('time', '', "# This block is SPPED UP\n\ntrain_p_0 = train[['x_0', 'y_0', 'z_0']].values\ntrain_p_1 = train[['x_1', 'y_1', 'z_1']].values\ntest_p_0 = test[['x_0', 'y_0', 'z_0']].values\ntest_p_1 = test[['x_1', 'y_1', 'z_1']].values\n\ntrain['dist_speedup'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)\ntest['dist_speedup'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)")


# ### export before model running

# In[36]:


train["index_diff"] =( (train["atom_index_0"]- train["atom_index_1"]).abs()-1)
test["index_diff"] =( (test["atom_index_0"]- test["atom_index_1"]).abs()-1)


# In[34]:


train.head()


# In[ ]:


train.to_csv("MolecularProperties_train_v1.csv.gz",compression="gzip")
test.to_csv("MolecularProperties_test_v1.csv.gz",compression="gzip")


# ## prep data for model running

# In[ ]:


molecules = train.pop('molecule_name')
test = test.drop('molecule_name', axis=1)


# In[ ]:


train.head()


# In[ ]:


train['fold'] = 0


# In[ ]:


n_splits = 3
gkf = GroupKFold(n_splits=n_splits) # we're going to split folds by molecules


for fold, (in_index, oof_index) in enumerate(gkf.split(train, groups=molecules)):
    train.loc[oof_index, 'fold'] = fold


# In[ ]:


import h2o
print(h2o.__version__)
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='14G')


# In[ ]:


train = h2o.H2OFrame(train)


# In[ ]:


test = h2o.H2OFrame(test)


# In[ ]:


train['type'] = train['type'].asfactor()
train['atom_0'] = train['atom_0'].asfactor()
train['atom_1'] = train['atom_1'].asfactor()

test['type'] = test['type'].asfactor()
test['atom_0'] = test['atom_0'].asfactor()
test['atom_1'] = test['atom_1'].asfactor()


# In[ ]:


x = test.columns
y = 'scalar_coupling_constant'


# In[ ]:


aml = H2OAutoML(max_models=10, seed=47, max_runtime_secs=100) # 50 models,  max_runtime_secs=26000
aml.train(x=x, y=y, training_frame=train)


# In[ ]:


# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head()  # Print all rows instead of default (10 rows)


# In[ ]:


# The leader model is stored here
aml.leader


# In[ ]:


preds = aml.predict(test)
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['scalar_coupling_constant'] = preds.as_data_frame().values.flatten()
sample_submission.to_csv('h2o_submission_2.csv', index=False)


# In[ ]:




