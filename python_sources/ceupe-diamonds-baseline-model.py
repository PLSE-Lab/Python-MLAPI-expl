#!/usr/bin/env python
# coding: utf-8

# ## python imports & setup

# In[ ]:


# imports
import numpy as np
import pandas as pd


# In[ ]:


# dataviz
import matplotlib as mpl
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[ ]:


# mpl.rcParams['figure.figsize'] = (12.0, 8.0)


# ## data loading

# In[ ]:


train = pd.read_csv('../input/ceupe-big-data-analytics/diamonds_train.csv')
test = pd.read_csv('../input/ceupe-big-data-analytics/diamonds_test.csv')
sample_sub = pd.read_csv('../input/ceupe-big-data-analytics/sample_submission.csv')


# ## exploratory data analysis (eda)

# In[ ]:


# ejemplo
train['carat'].plot(kind='hist', bins=20, title='histogram', figsize=(10, 7));


# ## modeling (linear regression baseline)

# In[ ]:


target = 'price'
cat_features = ['cut', 'color', 'clarity']
num_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

for cat_feat in cat_features:
    train[cat_feat] = train[cat_feat].astype('category')
    test[cat_feat] = test[cat_feat].astype('category')
    
cat_df = pd.get_dummies(train[cat_features])
num_df = train.loc[:,num_features]
train_df = pd.concat([cat_df, num_df], axis=1)

cat_df = pd.get_dummies(test[cat_features])
num_df = test.loc[:,num_features]
test_df = pd.concat([cat_df, num_df], axis=1)


features = list(cat_df.columns) + list(num_df.columns)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(train_df.loc[:,features].values)
y = train[target]


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X=X, y=y)


# ## make submission

# In[ ]:


X_test = scaler.transform(test_df.loc[:,features].values)
y_hat = model.predict(X_test).clip(0, 30000)
submission = pd.DataFrame({'id': test['id'], 'price': y_hat})
submission.to_csv('submission.csv', index=False)

