#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

X_train_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

X_train_full.dropna(subset=['SalePrice'], axis=0, inplace=True)
y = X_train_full.SalePrice
X_train_full.drop(['SalePrice'], axis=1, inplace=True)

num_cols = [col for col in X_train_full.columns 
            if X_train_full[col].dtype in ['int64', 'float64']]
cat_cols = [col for col in X_train_full.columns 
            if X_train_full[col].dtype == 'object' and X_train_full[col].nunique() < 10]

X_train_num = X_train_full[num_cols].copy()
X_train_cat = X_train_full[cat_cols].copy()
X_test_num = X_test_full[num_cols].copy()
X_test_cat = X_test_full[cat_cols].copy()

num_imputer = SimpleImputer(strategy='mean')
X_train_num_imputed = pd.DataFrame(num_imputer.fit_transform(X_train_num))
X_test_num_imputed = pd.DataFrame(num_imputer.transform(X_test_num))
X_train_num_imputed.columns = X_train_num.columns
X_test_num_imputed.columns = X_test_num.columns

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_train_cat))
X_test_cat_imputed = pd.DataFrame(cat_imputer.transform(X_test_cat))
X_train_cat_imputed.columns = X_train_cat.columns
X_test_cat_imputed.columns = X_test_cat.columns

cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_cat_encoded = pd.DataFrame(cat_encoder.fit_transform(X_train_cat_imputed))
X_test_cat_encoded = pd.DataFrame(cat_encoder.transform(X_test_cat_imputed))
X_train_cat_encoded.index = X_train_cat_imputed.index
X_test_cat_encoded.index= X_test_cat_imputed.index

X_train_cleaned = pd.concat([X_train_num_imputed, X_train_cat_encoded], axis=1)
X_test_cleaned = pd.concat([X_test_num_imputed, X_test_cat_encoded], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_cleaned, y, 
                                                      train_size=0.8, test_size=0.2,random_state=0)


# In[ ]:


xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05)
xgb_model.fit(X_train, y_train, 
         early_stopping_rounds=5, 
         eval_set=[(X_valid, y_valid)], 
         verbose=False)
y_predict = xgb_model.predict(X_valid)
print(int(mean_absolute_error(y_valid, y_predict)))

