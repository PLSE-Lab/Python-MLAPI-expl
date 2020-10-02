#!/usr/bin/env python
# coding: utf-8

# # EDA and training models
# Data preparation techniques used:
# * onehot encoding for categorical features
# * sorting and numerical encoding for ordinal features
# * sin/cos encoding for cyclic data
# * onehot+SVD for categorical features with too many categories
# 
# Models tested:
# * Linear regression
# * Two-layer perceptron
# * Simple voting
# 
# References:
# * https://www.kaggle.com/shahules/an-overview-of-encoding-techniques
# * https://www.kaggle.com/peterhurford/why-not-logistic-regression

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook as tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load data
df_train = pd.read_csv('../input/cat-in-the-dat/train.csv')
df_test = pd.read_csv('../input/cat-in-the-dat/test.csv')

print(df_train.shape)
print(df_test.shape)

df_train.info()


# ## EDA and Feature Engineering

# ### Binary

# In[ ]:


# Binary data

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(2, 5, figsize=(20, 8))

for i in range(5):
    col = 'bin_{}'.format(i)
    ax = axs[0, i]
    sns.countplot(df_train[col], hue=df_train['target'], ax=ax)
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.legend(title="target", loc='upper center')
    
    ax = axs[1, i]
    sns.barplot(x=col, y='target', data=df_train, ax=ax)


# ### Categorical and ordinal

# In[ ]:


# Categorical count
for i in range(10):
    col = 'nom_{}'.format(i)
    print(col, df_train[col].nunique())

# Ordinal count
for i in range(6):
    col = 'ord_{}'.format(i)
    print(col, df_train[col].nunique())


# In[ ]:


# Categorical with few uniques

fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    col = 'nom_{}'.format(i)
    ax = axs[i]
    #sns.countplot(train[col], hue=target, ax=ax)
    sns.barplot(x=col, y='target', data=df_train.iloc[:10000], ax=ax)
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.legend(title="target", loc='upper center')


# In[ ]:


# Ordinal

fig, axs = plt.subplots(1, 6, figsize=(20, 4))

for i in range(6):
    col = 'ord_{}'.format(i)
    ax = axs[i]
    sns.barplot(x=col, y='target', data=df_train, ax=ax)
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.legend(title="target", loc='upper center')


# In[ ]:


# Ordinal sorted by label
fig, axs = plt.subplots(1, 6, figsize=(20, 4))

for i in range(6):
    col = 'ord_{}'.format(i)
    
    order = sorted(df_train[col].unique(), key=lambda val: df_train[df_train[col] == val]['target'].mean())
    df_train[col + '_sort'] = df_train[col].map({val: i for (i, val) in enumerate(order)})
    df_test[col + '_sort'] = df_test[col].map({val: i for (i, val) in enumerate(order)})
    
    ax = axs[i]
    sns.barplot(x=col + '_sort', y='target', data=df_train, ax=ax)
    ax.set_title(col + '_sort', fontsize=14, fontweight='bold')
    ax.legend(title="target", loc='upper center')


# ### Cyclic

# In[ ]:


cyclic_cols = ['day','month']

fig, axs = plt.subplots(1, len(cyclic_cols), figsize=(8, 4))

for i in range(len(cyclic_cols)):
    col = cyclic_cols[i]
    ax = axs[i]
    sns.barplot(x=col, y='target', data=df_train, ax=ax)
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.legend(title="target", loc='upper center')


# In[ ]:


for df in [df_train, df_test]:
    for col in cyclic_cols:
        df[col+'_sin'] = np.sin((2 * np.pi * df[col]) / max(df[col]))
        df[col+'_cos'] = np.cos((2 * np.pi * df[col]) / max(df[col]))


# In[ ]:


sns.pairplot(df_train.loc[:1000], vars=['day_sin', 'day_cos', 'day'], hue='target')


# In[ ]:


sns.pairplot(df_train.loc[:1000], vars=['month_sin', 'month_cos', 'month'], hue='target')


# ## Data preparation

# In[ ]:


# Subset
target = df_train['target']
train_id = df_train['id']
test_id = df_test['id']
train0 = df_train.copy().drop(['target', 'id'], axis=1)
test0 = df_test.copy().drop('id', axis=1)

print(train0.shape)
print(test0.shape)

for df in [train0, test0]:
    # binary str to num
    df['bin_3'] = df['bin_3'].map({'T': 1, 'F': 0})
    df['bin_4'] = df['bin_4'].map({'Y': 1, 'N': 0})
    
    # drop cyclic features, leaving their sin and cos
    #df.drop(cyclic_cols, axis=1, inplace=True)
    df[cyclic_cols] /= df[cyclic_cols].max(axis=0)
    
    # drop unsorted ordinal features
    df.drop(['ord_{}'.format(i) for i in range(6)], axis=1, inplace=True)
    # scale sorted ordinal features
    df[['ord_{}_sort'.format(i) for i in range(6)]] /= df[['ord_{}_sort'.format(i) for i in range(6)]].max(axis=0)

train0.head()


# ### Onehot encoding + SVD
# Encode categorical features into sparse matrix, then lower the dimension  
# This will be feature set #1

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.decomposition import TruncatedSVD, PCA\n\n# One hot for features with few categories\ncol_names = ['nom_{}'.format(i) for i in range(5)] + ['ord_{}'.format(i) for i in range(5)] + ['day', 'month']\ntrain_ohe = df_train[col_names]\ntest_ohe = df_test[col_names]\n\ntraintest = pd.concat([train_ohe, test_ohe])\ndummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)\n\ntrain_ohe1 = dummies.iloc[:df_train.shape[0], :]\ntest_ohe1 = dummies.iloc[df_train.shape[0]:, :]\nprint('train_ohe1.shape', train_ohe1.shape)\n\n# One hot + SVD  for features with many categories\ncol_names = ['nom_{}'.format(i) for i in range(5, 10)] + ['ord_{}'.format(i) for i in range(5, 6)]\ntrain_ohe = df_train[col_names]\ntest_ohe = df_test[col_names]\n\ntraintest = pd.concat([train_ohe, test_ohe])\ndummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)\n\ntrain_ohe2 = dummies.iloc[:df_train.shape[0], :].sparse.to_coo().tocsr()\ntest_ohe2 = dummies.iloc[df_train.shape[0]:, :].sparse.to_coo().tocsr()\nprint('train_ohe2.shape', train_ohe2.shape)\n\n# Lower dimensionality\nn_components = 200\nsvd = TruncatedSVD(n_components=n_components)\ntrain_svd = svd.fit_transform(train_ohe2)\ntest_svd = svd.transform(test_ohe2)\nprint('train_svd.shape', train_svd.shape)\n\n# Join features\ntrain = train0.drop(['nom_{}'.format(i) for i in range(10)], axis=1)\ntest = test0.drop(['nom_{}'.format(i) for i in range(10)], axis=1)\ncol_names = ['pca_{}'.format(i) for i in range(n_components)]\ntrain = pd.concat([train, train_ohe1, pd.DataFrame(train_svd, columns=col_names)], axis=1)\ntest = pd.concat([test, test_ohe1, pd.DataFrame(test_svd, columns=col_names)], axis=1)\n\nprint('train.shape', train.shape) # feature set #1")


# ### Onehot encoding to sparse matrix
# Encode everything to onehot  
# This is feature set #2

# In[ ]:


# One hot
col_names = ['bin_{}'.format(i) for i in range(5)]         + ['nom_{}'.format(i) for i in range(10)]         + ['ord_{}'.format(i) for i in range(6)]         + ['day', 'month']
train_ohe = df_train[col_names]
test_ohe = df_test[col_names]

traintest = pd.concat([train_ohe, test_ohe])
dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)

train_ohe = dummies.iloc[:df_train.shape[0], :].sparse.to_coo().tocsr()
test_ohe = dummies.iloc[df_train.shape[0]:, :].sparse.to_coo().tocsr()

print('train_ohe.shape', train_ohe.shape)  # feature set #2


# ## Model training

# ### Logistic regression

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.testing import ignore_warnings

lr1 = LogisticRegression(solver='lbfgs', C=0.1)
lr2 = LogisticRegression(solver='lbfgs', C=0.1)

with ignore_warnings(category=FutureWarning):
    get_ipython().run_line_magic('time', "print('Dense CV:', cross_val_score(lr1, train, target, cv=2, scoring='roc_auc', n_jobs=-1).mean())")
    get_ipython().run_line_magic('time', "print('OHE CV:', cross_val_score(lr2, train_ohe, target, cv=2, scoring='roc_auc', n_jobs=-1).mean())")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lr1.fit(train, target)\nlr2.fit(train_ohe, target)')


# In[ ]:


from sklearn.utils.testing import ignore_warnings

with ignore_warnings(category=FutureWarning):
    lr1_predictions = lr1.predict_proba(test)
lr1_submission = pd.DataFrame({'id': test_id, 'target': lr1_predictions[:, 1]})
lr1_submission.to_csv('lr1.csv', index=False)
lr1_submission.head()


# In[ ]:


with ignore_warnings(category=FutureWarning):
    lr2_predictions = lr2.predict_proba(test_ohe)
lr2_submission = pd.DataFrame({'id': test_id, 'target': lr2_predictions[:, 1]})
lr2_submission.to_csv('lr2.csv', index=False)
lr2_submission.head()


# ### SVM
# It takes too long to train, so I commented it out

# In[ ]:


# %%time

# from sklearn.svm import SVC

# svm = SVC(kernel='rbf', probability=True, gamma='scale', C=0.1)
# print('CV:', cross_val_score(svm, train.iloc[:10000], target.iloc[:10000], cv=2, n_jobs=-1, scoring='roc_auc').mean())


# In[ ]:


# svm.fit(train.iloc[:10000], target.iloc[:10000])


# In[ ]:


# %%time
# svm_predictions = svm.predict_proba(test)
# svm_submission = pd.DataFrame({'id': test_id, 'target': svm_predictions[:, 0]})
# svm_submission.to_csv('svm.csv', index=False)
# svm_submission.head()


# ### 2 Layer Perceptron
# Neural network is usually worse than SVM on this kind of data.  
# But it trains faster when data is big, like this.  
# I used skorch (https://github.com/skorch-dev/skorch) for easy compatibility with torch and sklearn.

# In[ ]:


get_ipython().run_cell_magic('time', '', "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nfrom skorch import NeuralNetClassifier\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n\nclass MyModule(nn.Module):\n    def __init__(self, num_units=20):\n        super(MyModule, self).__init__()\n\n        self.dense1 = nn.Linear(train.shape[1], num_units)\n        self.dropout = nn.Dropout(0.5)\n        self.dense2 = nn.Linear(num_units, 2)\n\n    def forward(self, X, **kwargs):\n        X = self.dense1(X)\n        X = torch.tanh(X)\n        X = self.dropout(X)\n        X = self.dense2(X)\n        X = F.softmax(X, dim=-1)\n        return X\n\nnet = NeuralNetClassifier(\n    MyModule,\n    max_epochs=20,\n    lr=0.01,\n    iterator_train__shuffle=True,\n)\n\npipe = Pipeline([\n    ('scale', StandardScaler()),\n    ('net', net),\n])\n\nprint('CV:', cross_val_score(pipe,\n                             train.values.astype('float32'),\n                             target.values,\n                             cv=2,\n                             n_jobs=-1,\n                             scoring='roc_auc').mean())")


# In[ ]:


pipe.fit(train.values.astype('float32'), target.values)


# In[ ]:


net_predictions = pipe.predict_proba(test.values.astype('float32'))
net_submission = pd.DataFrame({'id': test_id, 'target': net_predictions[:, 1]})
net_submission.to_csv('net.csv', index=False)
net_submission.head()


# ## Voting

# In[ ]:


predictions = pd.DataFrame({'lr1': lr1_predictions[:, 1],
                            'lr2': lr2_predictions[:, 1],
                            'net': net_predictions[:, 1]})
corr = predictions.corr()
corr


# In[ ]:


sns.heatmap(corr, annot=True, square=True);


# Logregression and neural network output almost the same results when trained on the same data  
# So, we can conclude that there might be no need for nonlinearity

# In[ ]:


vote_predictions = (lr1_predictions + lr2_predictions) / 2
vote_submission = pd.DataFrame({'id': test_id, 'target': vote_predictions[:, 1]})
vote_submission.to_csv('vote1.csv', index=False)
vote_submission.head()


# In[ ]:


vote_predictions = (lr1_predictions * 0.1 + lr2_predictions) / 1.1
vote_submission = pd.DataFrame({'id': test_id, 'target': vote_predictions[:, 1]})
vote_submission.to_csv('vote2.csv', index=False)
vote_submission.head()


# In[ ]:


vote_predictions = (lr1_predictions * 0.1 + net_predictions * 0.1 + lr2_predictions) / 1.2
vote_submission = pd.DataFrame({'id': test_id, 'target': vote_predictions[:, 1]})
vote_submission.to_csv('vote3.csv', index=False)
vote_submission.head()

