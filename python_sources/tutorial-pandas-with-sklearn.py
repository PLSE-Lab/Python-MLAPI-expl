#!/usr/bin/env python
# coding: utf-8

# In[ ]:


link = "https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62"


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # read training data

# In[ ]:


df = pd.read_csv("../input/train.csv")


# In[ ]:





# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.dtypes['LotFrontage'] == float


# 
# Remove the target variable from the training set

# In[ ]:


data = df.drop(columns='SalePrice')
label = df['SalePrice'].values


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.33, random_state=42)


# Encoding a single string column

# In[ ]:


hs_train = X_train[['HouseStyle']].copy() 
hs_val = X_val[['HouseStyle']].copy()
hs_train.ndim


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
hs_train_transformed = ohe.fit_transform(hs_train)
hs_train_transformed


# In[ ]:


hs_train_transformed.shape


# 

# In[ ]:


# ohe = OneHotEncoder(sparse=False)
# hs_train_transformed = ohe.fit_transform(train)
# hs_train_transformed


# In[ ]:





# # use pipeline
# * SimpleImputer
# * OneHotEncoder

# In[ ]:


si_step = ('si', SimpleImputer(strategy='constant',fill_value='MISSING'))
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))


# In[ ]:


pipe = Pipeline([si_step, ohe_step])
pipe


# In[ ]:


hs_train_transform = pipe.fit_transform(hs_train)


# In[ ]:


hs_val_transform = pipe.transform(hs_val)


# ### use ColumnTransformer

# In[ ]:


# category col
cat_cols = ['RoofMatl', 'HouseStyle'] 
cat_si_step = ('si', SimpleImputer(strategy='constant',fill_value='MISSING'))
cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
cat_steps = [cat_si_step, cat_ohe_step]
cat_pipe = Pipeline(cat_steps)
cat_ct = ColumnTransformer(transformers=[("cat", cat_pipe, cat_cols)])


# In[ ]:


cat_pipe.fit_transform(X_train[["RoofMatl", "HouseStyle"]])


# Handle numerical data

# In[ ]:


num_si_step = ('si', SimpleImputer(strategy='median'))
num_ss_step = ('ss', StandardScaler())
num_cols = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual']
num_pipe = Pipeline([num_si_step, num_ss_step])
num_ct = ColumnTransformer(transformers=[("num", num_pipe, num_cols)])


# In[ ]:


X_train_transform_num = num_ct.fit_transform(X_train)
X_train_transform_num.shape


# In[ ]:


num_cols


# Use both cat pipe and num pipe to create training data

# In[ ]:


full_ct = ColumnTransformer(transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


# In[ ]:


X_train_transform_full = full_ct.fit_transform(X_train)
X_train_transform_full.shape


# ### build numerical and category col name

# In[ ]:


num_cols = []
cat_cols = []
for name in X_train.columns:
    if (name == "Id"):
        continue
    d_type = X_train.dtypes[name]
    if np.issubdtype(d_type, np.number):
        num_cols.append(name)
    else:
        cat_cols.append(name)


# ### transform full v2 with new col name

# In[ ]:


full_ct_v2 = ColumnTransformer(transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
X_train_transform_full_v2 = full_ct_v2.fit_transform(X_train)
X_train_transform_full_v2.shape


# ### Machine learning pipeline

# In[ ]:


from sklearn.linear_model import Ridge
ml_pipe = Pipeline([('transform', full_ct_v2), ('ridge', Ridge())])


# In[ ]:


ml_pipe.fit(X_train, y_train)


# In[ ]:


ml_pipe.score(X_val, y_val)


# ### Run model to predict test dataset and submit 

# In[ ]:


evaluate_df = pd.read_csv("../input/test.csv")
evaluate_df = evaluate_df.set_index('Id')
evaluate_id = evaluate_df.index


# In[ ]:


pred = ml_pipe.predict(evaluate_df)


# In[ ]:


submission = pd.DataFrame({
    'Id': evaluate_id,
    'SalePrice': pred
})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




