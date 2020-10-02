#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col=0)
train.head()


# It looks like we mainly have sparse numeric features here.

# In[ ]:


train.info()


# The data has floating point and integer features. We may be able to reduce the in-memory size here.
# 
# We are scored on root-mean-square log error, so we should transform the target by a logarithm so that we can make use of methods based on the RMS error.

# In[ ]:


train['log_target'] = np.log1p(train['target'])


# Now, let's see what the raw target looks like.

# In[ ]:


plt.hist(train.target,range=(0,4e7),bins=100)
plt.xlabel('Target')
plt.ylabel('Number of Samples')
plt.show()


# It covers several orders of magnitude and also has some quantization. Let's start zooming in.

# In[ ]:


plt.hist(train.target,range=(0,1e7),bins=100)
plt.xlabel('Target')
plt.ylabel('Number of Samples')
plt.show()


# The quantization looks somewhat regular but at several different scales.

# In[ ]:


plt.hist(train.target,range=(0,2e6),bins=100)
plt.xlabel('Target')
plt.ylabel('Number of Samples')
plt.show()


# We should remove all features that have no variation.

# In[ ]:


std = train.std().sort_values()
bad_fields = std[std==0].index
train = train.drop(bad_fields, axis=1)
train.head()


# This removed a couple hundred features.
# 
# We can also check that none of the features have minimum values less than zero. We can exploit this to reduce the size of the integer data.

# In[ ]:


changed_type = []
for col, dtype in train.dtypes.iteritems():
    if dtype==np.int64:
        max_val = np.max(train[col])
        bits = np.log(max_val)/np.log(2)
        if bits < 8:
            new_dtype = np.uint8
        elif bits < 16:
            new_dtype = np.uint16
        elif bits < 32:
            new_dtype = np.uint32
        else:
            new_dtype = None
        if new_dtype:
            changed_type.append(col)
            train[col] = train[col].astype(new_dtype)
print('Changed types on {} columns'.format(len(changed_type)))
print(train.info())


# This was a significant reduction in memory.
# 
# We also want to remove features that are too sparse. We can first calculate the sparsity of each feature.

# In[ ]:


sparsity = {
    col: (train[col] == 0).mean()
    for idx, col in enumerate(train)
}
sparsity = pd.Series(sparsity)
    


# In[ ]:


fig = plt.figure(figsize=[7,12])
ax = fig.add_subplot(211)
ax.hist(sparsity, range=(0,1), bins=100)
ax.set_xlabel('Sparsity of Features')
ax.set_ylabel('Number of Features')
ax = fig.add_subplot(212)
ax.hist(sparsity, range=(0.8,1), bins=100)
ax.set_xlabel('Sparsity of Features')
ax.set_ylabel('Number of Features')
plt.show()


# Most of the features are very sparse. We'll set a minimum number of non-zero values of 10 and remove all the other features.

# In[ ]:


min_non0 = 10
too_sparse = sparsity[(((1-sparsity) * train.shape[0]) < min_non0)].index
train = train.drop(too_sparse, axis=1)


# In[ ]:


train.info()


# In[ ]:


train.head()


# Now that we've prepared some data, we can train a model. Decision trees typically don't care about the scale of the features, so we can plug everything into a decision tree model to get a very simple result.
# 
# For other models such as linear regression, SVMs, neural nets, etc. we will want o figure out how to properly scale the various features.
# 
# I haven't done much tuning at all, but I will run a 10-fold cross validation using the built-in LightGBM cv() function.

# In[ ]:


import lightgbm as lgb
features = train.drop(['target','log_target'], axis=1).values
targets = train['log_target'].values.reshape([-1])
feature_names = list(train.drop(['target','log_target'], axis=1).columns.values)
train_dataset = lgb.Dataset(
        features,
        targets,
        feature_name=feature_names 
)

    
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'metric': {'rmse'},
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8    
}

cv_output = lgb.cv(
    params,
    train_dataset,
    num_boost_round=500,
    nfold=10,
    stratified=False
)


# The cross validation will also show us the optimal number of iterations to use.

# In[ ]:


n_iterations = np.argmin(cv_output['rmse-mean'])
print('Optimal # of iterations: {}'.format(n_iterations))
print('Score: {:0.5}, Std. Dev.: {:0.5}'.format(
    cv_output['rmse-mean'][n_iterations],
    cv_output['rmse-stdv'][n_iterations]
))


# Now we can train the model. This is very fast on this dataset, so I won't bother saving the model.

# In[ ]:


model = lgb.train(
    params,
    train_dataset,
    num_boost_round=n_iterations
)


# Now I can prepare the test data.

# In[ ]:


test = pd.read_csv('../input/test.csv', index_col=0)
test = test.drop(bad_fields, axis=1)
test = test.drop(too_sparse, axis=1)
test.info()


# It's interesting that now all the features are floating-point on the test data. This means that there is some difference between how the training and test data was prepared. I won't investigate this here.

# In[ ]:


preds = model.predict(test.values)
preds = np.exp(preds) - 1


# In[ ]:


test['target'] = preds
test[['target']].to_csv('lightgbm_basic.csv')


# In[ ]:




