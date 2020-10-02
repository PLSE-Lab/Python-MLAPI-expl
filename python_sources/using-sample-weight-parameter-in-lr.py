#!/usr/bin/env python
# coding: utf-8

# LogisticRegression's fit() method has parameter sample_weights, which could be used to assign individual weights to the samples. According to documantation:
# > sample_weight : array-like, shape (n_samples,) optional
# 
# > Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight.
# 
# In this kernel we'll compare results with no sample weights and setting weights according to class distribution in train datatset. As our two classes are not evenly distributed in train dataset(but **they ARE in test dataset!**), I expect that model will benefit from setting "balancing" weights to the samples of minority class.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#First load train dataset
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')


# Since the purpose of this kernel is just to demonstrate use of sample_weight parameter, we'll just use One Hot Encoding for all columns, and no fancy data preprocessing.

# In[ ]:


dummy_cols=['bin_0','bin_1','bin_2','bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9','ord_0','ord_1','ord_2','ord_3','ord_4','ord_5','month', 'day']
train = pd.get_dummies(train, columns=dummy_cols, sparse=True)


# Because we want to test results with balanced dataset (as is test.csv), we'll pick first 3000 rows from each class as validation dataset.

# In[ ]:


#validation dataset
validate = pd.concat([train[train['target']==0].head(3000),train[train['target']==1].head(3000)]).reset_index(drop=True)

#drop validation rows from train
train=train[~train.id.isin(validate.id)].reset_index(drop=True)

#get target column, then drop 'id' and  'target' from the two dataframes
target = train['target']
train = train.drop(['id','target'], axis=1)
target_val = validate['target']
validate = validate.drop(['id','target'], axis=1)

#convert to sparse 
train = train.sparse.to_coo().tocsr()
validate = validate.sparse.to_coo().tocsr()


# Let's define a helper function - Logistic regression classifier and normalized confussion matrix plot(no need to write code twice).

# In[ ]:


def lr_classifier(x, y, x_val, y_val, sample_weight):
    lr = LogisticRegression(solver = 'lbfgs', C = 0.1, max_iter=1000)
    lr.fit(x,y,sample_weight) 
    y_pred = lr.predict(x_val)
    
    cm = confusion_matrix(y_val, y_pred )
    cm = cm.astype('float') / cm.sum(axis=1)

    plt.matshow(cm)
    plt.title('Confusion matrix')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.show()


# First we'll try LogisticRegression withouth sample_weights.
# 

# In[ ]:


lr_classifier(train,target,validate,target_val,sample_weight = None)


# Now let's set weights. Since we have 208 236 samples for class 0 and 91 764 samples for class 1, I'll set weight 2.27 for all samples of minority class (we have 2.27 times less samples for class 1).

# In[ ]:


sample_weight= target.apply( lambda x: 2.27 if x==1 else 1)
lr_classifier(train,target,validate,target_val,sample_weight)


# We see clearly, that "weighted" example performs much better on balanced dataset. You can try it on the real data (test.csv, which, as I already mentioned, is balanced) and find the difference for yourself. 
# 
# Happy kaggling :))))
