#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')


# In[ ]:


train.columns


# In[ ]:


y = train['SalePrice']

features = ['LotArea', 'YearBuilt', 'GarageYrBlt', 'PoolArea', 'GarageCars']
X = train[features]


# In[ ]:


X['GarageYrBlt'] = X['GarageYrBlt'].fillna(1978)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X, y)


# In[ ]:


X_test = test[features]


# In[ ]:


X_test['GarageYrBlt'] = X_test['GarageYrBlt'].fillna(X_test['GarageYrBlt'].mean())
X_test['GarageCars'] = X_test['GarageCars'].fillna(X_test['GarageCars'].mean())


# In[ ]:


preds = model.predict(X_test)


# In[ ]:


submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')
submission.head()


# In[ ]:


submission['SalePrice'] = preds


# In[ ]:


submission.to_csv('submission_1.csv', index=False)

