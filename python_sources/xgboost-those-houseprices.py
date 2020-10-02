#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

from os.path import join


# In[ ]:


dir_path = '../input/home-data-for-ml-course/'
train_df = pd.read_csv(join(dir_path, 'train.csv'))
test_df = pd.read_csv(join(dir_path, 'test.csv'))

y = train_df.SalePrice
X = train_df.drop(['SalePrice'], axis=1)


# In[ ]:


categorical_cols = [ 
    cname
    for cname in X.columns
    if X[cname].dtype == 'object'
    ## normally would keep hotencoding to small categories
    ## but including neighborhoods seems important
    # and X[cname].nunique() < 10
]

numerical_cols = [
    cname
    for cname in X.columns
    if X[cname].dtype in ['int64', 'float64']    
]


# In[ ]:


# split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, random_state=0)


# # Preprocessing Pipelines
# Handle:
# - missing data
# - categorical data (One-Hot Encoding)

# In[ ]:


num_transformer = SimpleImputer(strategy='median')
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer([
    ('numerical',   num_transformer, numerical_cols),
    ('categorical', cat_transformer, categorical_cols),
])


# # XGBoost Model
# Setting `n_estimators` really high (normally 100-1000) and `learning_rate` lower than the default (0.10), although it takes longer to train, is recommended by the [xgboost tutorial](https://www.kaggle.com/alexisbcook/xgboost#learning_rate).

# In[ ]:


model = XGBRegressor(
    random_state=0,
    # high n_estimators and low learning_rate as 
    n_estimators=2500,
    learning_rate=0.04
)


# In[ ]:


final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

final_pipe.fit(train_X, train_y)
predictions = final_pipe.predict(val_X)

# Score it!
mean_absolute_error(val_y, predictions)


# # Submission

# In[ ]:


pred_price = final_pipe.predict(test_df)

pd.DataFrame({
    'Id': test_df.Id,
    'SalePrice': pred_price
}).to_csv('submission.csv', index=False)


# In[ ]:




