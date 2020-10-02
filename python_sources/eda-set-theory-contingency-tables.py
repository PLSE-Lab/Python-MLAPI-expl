#!/usr/bin/env python
# coding: utf-8

# ![](https://deelyee.files.wordpress.com/2011/01/i_hate_organic_chemistry_by_laughingwarlock.jpg?w=880&h=312&crop=1)

# This is a **Work in progress** I will write the narrative and develop it further soon. Stay tuned....

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


train_ = pd.read_csv('../input/train.csv', index_col = "id")


# In[ ]:


test_ = pd.read_csv('../input/test.csv', index_col = "id")


# In[ ]:


train_.head()


# In[ ]:


train_.dtypes


# In[ ]:


train_.shape


# In[ ]:


train_['atom_index_0'] = train_.atom_index_0.astype('category')
train_['atom_index_1'] = train_.atom_index_1.astype('category')


# In[ ]:


test_['atom_index_0'] = test_.atom_index_0.astype('category')
test_['atom_index_1'] = test_.atom_index_1.astype('category')


# In[ ]:


train_.describe(include = 'all')


# In[ ]:


dipole_ = pd.read_csv('../input/dipole_moments.csv')
# magnetic_ = pd.read_csv('../input/mag')
potential_ = pd.read_csv('../input/potential_energy.csv')
scalar_ = pd.read_csv('../input/scalar_coupling_contributions.csv')


# In[ ]:


scalar_['atom_index_0'] = scalar_.atom_index_0.astype('category')
scalar_['atom_index_1'] = scalar_.atom_index_1.astype('category')


# In[ ]:


scalar_.dtypes


# In[ ]:


potential_.head()


# In[ ]:


train_dm_ = pd.merge(train_, dipole_, how = 'inner', on = 'molecule_name')
train_dm_pe = pd.merge(train_dm_, potential_, how = 'inner', on = 'molecule_name')
train_dm_pe_s = pd.merge(train_dm_pe, scalar_, how = 'inner', on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])


# In[ ]:


train_dm_pe_s.head()


# In[ ]:


train_dm_pe_s.shape


# In[ ]:


import seaborn as sns
sns.set(rc={'figure.figsize':(15,12)})


# In[ ]:


sns.heatmap(train_dm_pe_s.corr(), annot=True, fmt="f")


# In[ ]:


sns.regplot(x = "fc", y = "scalar_coupling_constant", data = train_dm_pe_s)


# In[ ]:


sns.lmplot(x = "fc", y = "scalar_coupling_constant", data = train_dm_pe_s)


# In[ ]:


sns.lmplot(x = "dso", y = "scalar_coupling_constant", data = train_dm_pe_s)


# In[ ]:


sns.lmplot(x = "fc", y = "scalar_coupling_constant", col = "type", col_wrap = 3, data = train_dm_pe_s)


# In[ ]:


sns.lmplot(x = "fc", y = "scalar_coupling_constant", col = "atom_index_0", col_wrap = 2, data = train_dm_pe_s)


# In[ ]:


sns.lmplot(x = "fc", y = "scalar_coupling_constant", col = "atom_index_1", col_wrap = 2, data = train_dm_pe_s)


# In[ ]:


test_.describe(include = 'all')


# In[ ]:


train_['scalar_coupling_constant'].describe().apply(lambda x: format(x, 'f'))


# In[ ]:


train_mol = train_.loc[train_['molecule_name'] == 'dsgdb9nsd_042139']


# In[ ]:


train_mol.shape


# In[ ]:


ax = sns.heatmap(pd.crosstab(train_.atom_index_0, train_.type), annot = True, fmt = "d")


# In[ ]:


ax = sns.heatmap(pd.crosstab(test_.atom_index_0, test_.type), annot = True, fmt = "d")


# In[ ]:


ay = sns.heatmap(pd.crosstab(train_.atom_index_1, train_.type), annot = True, fmt = "d")


# In[ ]:


ay = sns.heatmap(pd.crosstab(test_.atom_index_1, test_.type), annot = True, fmt = "d")


# In[ ]:


sns.set(rc={'figure.figsize':(25,15)})
az = sns.heatmap(pd.crosstab(train_.atom_index_1, train_.atom_index_0), annot = True, fmt = "d")


# In[ ]:


az = sns.heatmap(pd.crosstab(test_.atom_index_1, test_.atom_index_0), annot = True, fmt = "d")


# **This is a disjoint set.**

# In[ ]:


train_.tail()


# In[ ]:


test_.head()


# In[ ]:


# library
import matplotlib.pyplot as plt
from matplotlib_venn import venn2


# In[ ]:


venn2([set(train_.atom_index_0), set(test_.atom_index_0)])


# In[ ]:


set(train_.atom_index_0).symmetric_difference(set(test_.atom_index_0))
# set(train_.atom_index_0).intersection(set(test_.atom_index_0))


# In[ ]:


venn2([set(train_.atom_index_1), set(test_.atom_index_1)])
# set(train_.atom_index_1).intersection(set(test_.atom_index_1))


# In[ ]:


venn2([set(train_.type), set(test_.type)])


# In[ ]:


train_grp_all = pd.DataFrame(train_.groupby(['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])['scalar_coupling_constant'].mean())
train_grp_mn = pd.DataFrame(train_.groupby(['molecule_name'])['scalar_coupling_constant'].mean())
train_grp_ai0 = pd.DataFrame(train_.groupby(['atom_index_0'])['scalar_coupling_constant'].mean())
train_grp_ai1 = pd.DataFrame(train_.groupby(['atom_index_1'])['scalar_coupling_constant'].mean())
train_grp_t = pd.DataFrame(train_.groupby(['type'])['scalar_coupling_constant'].mean())


# In[ ]:


train_grp_ai0.reset_index(level = 0, inplace = True)
train_grp_ai0.head()
train_grp_t.reset_index(level = 0, inplace = True)
train_grp_t.head()


# In[ ]:


ai0_0 = train_.loc[train_['atom_index_0'] == 0]


# In[ ]:


ai0_0


# In[ ]:


# sns.set_style("whitegrid")
sns.distplot(train_grp_t.scalar_coupling_constant, rug = True)
plt.style.use("dark_background")


# In[ ]:


sns.set_palette('colorblind')
sns.kdeplot(train_grp_t.scalar_coupling_constant, shade=True, color = "yellow", alpha = 0.9)


# In[ ]:


sns.kdeplot(train_grp_ai0.scalar_coupling_constant)


# In[ ]:


sns.kdeplot(train_grp_ai1.scalar_coupling_constant)


# In[ ]:


sns.boxplot(x = train_.type, y = train_.scalar_coupling_constant)


# In[ ]:


sns.boxplot(x = train_.atom_index_0, y = train_.scalar_coupling_constant)


# In[ ]:


sns.boxplot(x = train_.atom_index_1, y = train_.scalar_coupling_constant)


# In[ ]:


h = sns.jointplot(x = train_grp_ai0.scalar_coupling_constant, y = train_grp_ai1.scalar_coupling_constant, kind = "kde")
h.set_axis_labels('train_grp_ai0.scalar_coupling_constant', 'train_grp_ai1.scalar_coupling_constant', fontsize=16)


# In[ ]:


train_.loc[train_['molecule_name'] == 'dsgdb9nsd_042139']


# In[ ]:


structure_ = pd.read_csv('../input/structures.csv')


# In[ ]:


structure_.loc[structure_['molecule_name'] == 'dsgdb9nsd_042139']


# Refer here
# https://www.kaggle.com/joydeb28/ml-models

# In[ ]:


import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing

import graphviz
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

try:
    # To enable interactive mode you should install ipywidgets
    # https://github.com/jupyter-widgets/ipywidgets
    from ipywidgets import interact, SelectMultiple
    INTERACTIVE = True
except ImportError:
    INTERACTIVE = False


# In[ ]:


def get_train_data():
    # load dataset
    dataset_train = pandas.read_csv("../input/train.csv")
    dataset_test = pandas.read_csv("../input/test.csv")
    #['id' 'molecule_name' 'atom_index_0' 'atom_index_1' 'type' 'scalar_coupling_constant']
    cat_columns =['molecule_name', 'type']
    label_encoders = {}
    for col in cat_columns:
        new_le = LabelEncoder()
        dataset_train[col] = new_le.fit_transform(dataset_train[col])
        dataset_test[col] = new_le.fit_transform(dataset_test[col])
    X_train = pandas.DataFrame(dataset_train, columns=['molecule_name','atom_index_0','atom_index_1','type'])
    X_test = pandas.DataFrame(dataset_test, columns=['id','molecule_name','atom_index_0','atom_index_1','type'])
    Y_train = dataset_train['scalar_coupling_constant']
    return X_train,X_test,Y_train


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train,X_test_with_id,Y_train = get_train_data()


# In[ ]:


X_train.head()


# In[ ]:


X_test_with_id.head()


# In[ ]:


Y_train.head()


# In[ ]:


X_train = min_max_scaler.fit_transform(X_train)
# 0	1	0	0 the first row in the pandas dataframe becomes scaled numpy array 
# [0.        , 0.03571429, 0.        , 0.        ],
X_test = pandas.DataFrame(X_test_with_id, columns=['molecule_name','atom_index_0','atom_index_1','type'])
X_test = min_max_scaler.fit_transform(X_test)


# In[ ]:


X_train.shape


# In[ ]:


X_train[:10]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
print(x_train[0:5])
print(x_test[0:5])


# In[ ]:


evals_result = {}
#Parameters for LightGBM Model
params_lgb = {'num_leaves': 5,
          'min_child_samples': 79,
          'objective': 'regression',
          'max_depth': 9,
          'learning_rate': 0.1,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.9,
          "bagging_seed": 47,
          "metric": ['mae'],
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501,
          'colsample_bytree': 1.0,
          'n_estimators':1500}


import lightgbm as lgb
lgtrain = lgb.Dataset(x_train, label=y_train)
lgval = lgb.Dataset(x_test, label=y_test)
model_lgb = lgb.train(params_lgb, 
                      lgtrain, 10000, 
                      valid_sets=[lgtrain, lgval], 
                      verbose_eval=500,
                      evals_result=evals_result)
y_out = model_lgb.predict(X_test)


# In[ ]:


def render_metric(metric_name):
    ax = lgb.plot_metric(evals_result, metric=metric_name, figsize=(15, 10))
    plt.show()


# In[ ]:


if INTERACTIVE:
    # create widget to switch between metrics
    interact(render_metric, metric_name=params_lgb['metric'])
else:
    render_metric(params['metric'][0])


# In[ ]:


def render_plot_importance(importance_type, max_features=10,
                           ignore_zero=True, precision=4):
    ax = lgb.plot_importance(model_lgb, importance_type=importance_type,
                             max_num_features=max_features,
                             ignore_zero=ignore_zero, figsize=(12, 8),
                             precision=precision)
    plt.show()


# In[ ]:


if INTERACTIVE:
    # create widget for interactive feature importance plot
    interact(render_plot_importance,
             importance_type=['split', 'gain'],
             max_features=(1, X_train.shape[-1]),
             precision=(0, 10))
else:
    render_plot_importance(importance_type='split')


# In[ ]:


def render_tree(tree_index, show_info, precision=4):
    show_info = None if 'None' in show_info else show_info
    return lgb.create_tree_digraph(model_lgb, tree_index=tree_index,
                                   show_info=show_info, precision=precision)


# In[ ]:


if INTERACTIVE:
    # create widget to switch between trees and control info in nodes
    interact(render_tree,
             tree_index=(0, model_lgb.num_trees() - 1),
             show_info=SelectMultiple(  # allow multiple values to be selected
                 options=['None',
                          'split_gain',
                          'internal_value',
                          'internal_count',
                          'leaf_count'],
                 value=['None']),
             precision=(0, 10))
    tree = None
else:
    tree = render_tree(84, ['None'])
tree


# In[ ]:


my_submission = pandas.DataFrame({'id': X_test_with_id.id, 'scalar_coupling_constant': y_out})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




