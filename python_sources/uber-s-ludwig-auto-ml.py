#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# <a><img src="https://raw.githubusercontent.com/uber/ludwig/master/docs/images/og-image.jpg" alt="luldwig" border="0" height="500" width="500" ></a>
# 
# Here I want to try out Uber's auto-ML framework called Ludwig. As I do this exercise, it turned out to be quite simple and fast.
# 
# v1 (commit v1, v2): In this notebook, I don't do any data cleaning or feature engineering. So the result is actually quite poor. Especially there are a few columns that has 0 for what is supposed to be null value. But nonetheless, the intent is to show what a Ludwig pipeline would look like.
# 
# v2 (commit v3-v5): Since the performance is so bad, I will do some basic clean-up and data preprocessing first. The preprocessing steps will mainly be sourced from other kernels: [this one](https://www.kaggle.com/apapiu/regularized-linear-models) and [this one](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard). The result on train/val becomes very good, but the output on dftest doesn't really makes sense, simply just looking by the distribution. Perhaps Ludwig is extremely overfitting for small sample?
# 
# <a href="https://ibb.co/ZLcCx4z"><img src="https://i.ibb.co/p04pdVj/Screen-Hunter-3152.jpg" alt="Screen-Hunter-3152" border="0"></a>
# <a href="https://imgbb.com/"><img src="https://i.ibb.co/CwjzwQP/Screen-Hunter-3153.jpg" alt="Screen-Hunter-3153" border="0"></a><br />
# 
# 
# 

# # Import Ludwig
# 
# We need to pip install first. I'm installing from the repo, since there was an error when doing regular pip install (according to [this](https://www.mikulskibartosz.name/ludwig-machine-learing-model-in-kaggle/) article).

# In[ ]:


get_ipython().system('pip install https://github.com/uber/ludwig/archive/master.zip')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import ludwig
from ludwig.api import LudwigModel
import scipy as scipy

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


traindf = pd.read_csv("../input/train.csv")
testdf = pd.read_csv("../input/test.csv")


# In[ ]:


print(traindf.shape)
print(testdf.shape)


# In[ ]:


traindf.head()


# In[ ]:


testdf.head()


# In[ ]:


# Save the original for later
traindf_old = traindf 
testdf_old = testdf


# # Data Preprocessing

# In[ ]:


all_data = pd.concat((traindf.loc[:,'MSSubClass':'SaleCondition'], testdf.loc[:,'MSSubClass':'SaleCondition']))


# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index


# In[ ]:


skewed_feats = traindf[numeric_feats].apply(lambda x: scipy.stats.skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data = all_data.fillna(all_data.mean())


# In[ ]:


traindf = all_data.iloc[:len(traindf),:]
testdf = all_data.iloc[len(testdf)+1:,:]
traindf["SalePrice"] = np.log1p(traindf_old['SalePrice'])
print(traindf.shape)
print(testdf.shape)
print(traindf_old.shape)
print(testdf_old.shape)
# the ID columns is removed in the new traindf and testdf


# In[ ]:


traindf["SalePrice"].plot.hist()


# In[ ]:


traindf.head()


# # Input Features Dictionary
# 
# We need to create list of input feature dictionary as input into Ludwig. Here, I simply loop through all available columns, except for the target variable.

# In[ ]:


columns = list(traindf.columns)


# In[ ]:


dtypes = []
for c in columns:
    dtypes.append(traindf[c].dtype)


# In[ ]:


dtypes2 = []
for d in dtypes:
    if d in ('int64', 'float64'):
        dtypes2.append('numerical')
    if d == object:
        dtypes2.append('category')


# In[ ]:


print(dtypes2)


# In[ ]:


input_features = []
for col, dtype in zip(columns[:-1],dtypes2[:-1]):
    input_features.append(dict(name=col,type=dtype))
print(input_features)


# # Create Ludwig Model and train the model

# In[ ]:


model_definition = {
    'input_features':input_features,
    'output_features':[
        {'name': 'SalePrice', 'type':'numerical'}
    ]
}


# In[ ]:


model = LudwigModel(model_definition)
trainstats = model.train(traindf)


# # Model training statistics

# Note: I'm hiding the output since it's very long. The following section will explain the structure of the dictionary.

# In[ ]:


print(trainstats)


# ## Check the structure of the trainstats dictionary

# In[ ]:


for i in trainstats.keys():
    print(i)
    for j in trainstats[i]:
        print(' ',j)
        for k in trainstats[i][j]:
            print('  ',k)
    print('--')


# In[ ]:


print(trainstats['train'].keys())


# ## Check some performance metrics
# 
# v1: It turned out to be not so great
# <a href="https://ibb.co/zSv9yQx"><img src="https://i.ibb.co/br9wSQm/Screen-Hunter-3150.jpg" alt="Screen-Hunter-3150" border="0"></a>
# 
# v2: Much improved performance after doing some basic feature engineering. (Need to review again what feature engineering is being done vs. not being done by Ludwig)
# <a href="https://ibb.co/qBzCCwW"><img src="https://i.ibb.co/sjhCCr6/Screen-Hunter-3151.jpg" alt="Screen-Hunter-3151" border="0"></a>

# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(15,6))
axes[0].plot(trainstats['train']['SalePrice']['loss'],label='train')
axes[0].plot(trainstats['validation']['SalePrice']['loss'],label='validation')
axes[0].plot(trainstats['test']['SalePrice']['loss'],label='test')
axes[0].legend(loc='upper right')
axes[0].set_title('Loss')
axes[1].plot(trainstats['train']['SalePrice']['mean_absolute_error'],label='train')
axes[1].plot(trainstats['validation']['SalePrice']['mean_absolute_error'],label='validation')
axes[1].plot(trainstats['test']['SalePrice']['mean_absolute_error'],label='test')
axes[1].legend(loc='upper right')
axes[1].set_title('mean_absolute_error')
plt.show()


# # Do the predictions

# In[ ]:


pd.set_option('display.max_columns', 10)
print(traindf.head())
print(testdf.head())
print(traindf_old.head())
print(testdf_old.head())


# In[ ]:


testdf.head()


# In[ ]:


predictions = model.predict(testdf)


# In[ ]:


predictions.plot.hist()


# In[ ]:


predictions.head(10)


# In[ ]:


predictions = np.expm1(predictions)


# In[ ]:


pd.options.display.float_format = '{:,.0f}'.format
predictions.head(10)


# In[ ]:


submission = pd.DataFrame(testdf_old['Id'])
submission['SalePrice'] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)

