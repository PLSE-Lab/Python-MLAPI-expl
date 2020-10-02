#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


X = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')


# In[ ]:


X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# In[ ]:


low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)


# In[ ]:


imp=SimpleImputer(strategy="mean")

imputed_X_train = pd.DataFrame(imp.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imp.transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns


# In[ ]:


model = XGBRegressor(n_estimators=1000,learning_rate=0.09,max_depth=3)


model.fit(imputed_X_train,y_train,
              early_stopping_rounds=50,
              eval_set=[(imputed_X_valid, y_valid)],
              verbose=0)

pred = model.predict(X_valid)


mae = mean_absolute_error(pred,y_valid)


print("Mean Absolute Error:" , mae)


# In[ ]:


preds_test = model.predict(X_test)


# In[ ]:


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

