#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction
# Just a try with Kaggle Machine Learning Course.
# 
# I used many others code for my reference.
# 
# This is about predicting the house price based on the details like size of the house, number of rooms, number of bed rooms, number of Garages, location of the house, age of the house and etc. Normally In realestate, this value is fixed based on these parameters manually. we are trying to teach an algorithm to predict the price for us.
# 
# Let go into the data. :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#https://towardsdatascience.com/machine-learning-kaggle-competition-part-three-optimization-db04ea415507


# In[ ]:


#Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 11100)


# ## 1. Data Loading

# ### We have two dataset for training and testing.
# > Our Target varibale is **SalePrice**

# In[ ]:


# Read the data
X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
print("Train Size ",X_full.shape)
X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
print("Test Size ",X_test_full.shape)


# # EDA

# In[ ]:


X_full.head()


# In[ ]:


X_full.columns


# In[ ]:


yrblt = X_full.groupby(['YearBuilt'])['Street'].count().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.YearBuilt, y=yrblt.Street))
fig.show()


# ## After 2000 number of house construction gone peak.

# In[ ]:


yrblt = X_full.groupby(['HouseStyle'])['Street'].count().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.HouseStyle, y=yrblt.Street))
fig.show()


# ## Mostly 1 Story houses followed by 2 Story houses.

# In[ ]:


yrblt = X_full.groupby(['SaleCondition'])['Street'].count().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.SaleCondition, y=yrblt.Street))
fig.show()


# In[ ]:


yrblt = X_full.groupby(['LotArea'])['Street'].count().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.LotArea, y=yrblt.Street))
fig.show()


# ## Most of the house built area is less than 20K Sq.Ft

# In[ ]:


yrblt = X_full.groupby(['RoofStyle'])['Street'].count().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.RoofStyle, y=yrblt.Street))
fig.show()


# ## Gable -  supports for cold or temperate climates

# In[ ]:


yrblt = X_full.groupby(['BldgType'])['Street'].count().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.BldgType, y=yrblt.Street))
fig.show()


# ## 1 Family houses are too high

# In[ ]:


yrblt = X_full.groupby(['GarageCars'])['Street'].count().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.GarageCars, y=yrblt.Street))
fig.show()


# ## Even most of the houses are 1 family, but it contains 2 car parking garages.

# In[ ]:


yrblt = X_full.groupby(['GarageType'])['Street'].count().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.GarageType, y=yrblt.Street))
fig.show()


# ## Attached Car parking is high followed by Separate Garage

# In[ ]:


yrblt = X_full.groupby(['BldgType'])['SalePrice'].max().reset_index()
#print(yrblt)
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=yrblt.BldgType, y=yrblt.SalePrice))
fig.show()


# ## 1 Family houses selling price is too high

# In[ ]:


# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'])
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


# Columns with Null Values and its count
# this is helpful to identify columns to drop or impute
[{x: X_full[x].isnull().sum()} for x in X_full.columns[X_full.isna().any()]]


# Generally dropping the columns is not adviceable, but in our case the following columns are having too many null values. if the null values is less than 10% for a particular feature, we can try to impute to get better result. *(Imputing a large null based on our assumption may lead us in the wrong direction.)*

# In[ ]:


# Dropping Columns which is having high number of Nulls
high_null_columns = [x for x in X_full.columns if X_full[x].isna().sum()>100]
print([high_null_columns])
X_full.drop(axis = 1, columns = high_null_columns, inplace = True)
X_test_full.drop(axis = 1, columns = high_null_columns, inplace = True)
print("Train Size ",X_full.shape)
print("Test Size ",X_test_full.shape)


# > ## 2. Feature Engineering

# In[ ]:


cate_columns = X_full.select_dtypes('object').columns
print("Categorical columns ")
print(len(cate_columns),cate_columns)
num_columns = X_full.select_dtypes(exclude = ['object']).columns
print("Numerical columns ")
print(len(num_columns), num_columns)


# In[ ]:


# Cardinality checks
# Need to remove high cardinality columns since It will make more computing time after transformation.
sorted({x:X_full[x].nunique() for x in cate_columns}.items(), key=lambda x: x[1],reverse=True)

categorical_small_variety_cols = [cname for cname in cate_columns if
                    X_full[cname].nunique() < 15]
categorical_large_variety_cols = [cname for cname in cate_columns if
                    X_full[cname].nunique() >= 15]
print(categorical_small_variety_cols)
print(categorical_large_variety_cols)
categorical_label_cols =[]
skew_cols = []


# ## 3. Imputation and Pipeline building

# Imputation is done based on Numerical and Catecorical features. Mostly for numerical features filled by 'Median' values. Categorcial values filled by most frequent values of a particular feature.
# 
# Pipeline is an amazing tool to integrate the imputation and model. It will reduce a lot of manual typed codes.

# In[ ]:


# Imputation

# Preprocessing for numerical data
numerical_transformer = Pipeline(verbose=False,steps=[
    ('imputer_num', SimpleImputer(strategy='mean'))])

# Preprocessing for categorical data
categorical_onehot_transformer = Pipeline(verbose=False,steps=[
    ('imputer_onehot', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

categorical_label_transformer = Pipeline(verbose=False,steps=[
    ('imputer_label', SimpleImputer(strategy='most_frequent')),
    ('label', ce.OrdinalEncoder())
    
])

categorical_count_transformer = Pipeline(verbose=False,steps=[
    ('imputer_count', SimpleImputer(strategy='most_frequent')),
    ('count', ce.TargetEncoder(handle_missing='count'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(verbose=False,
    transformers=[
        ('num', numerical_transformer, num_columns),
        ('cox_box', PowerTransformer(method='yeo-johnson', standardize=False),skew_cols),
        ('cat_label', categorical_label_transformer, categorical_label_cols),
        ('cat_onehot', categorical_onehot_transformer, categorical_small_variety_cols),
        ('cat_count', categorical_onehot_transformer, categorical_large_variety_cols),
    ])

train_pipeline = Pipeline(verbose=False,steps=[
                    ('preprocessor', preprocessor),   
                    ('scale', StandardScaler(with_mean=False,with_std=True)),
                    ('model', XGBRegressor(random_state=42))
                    ])


# ## 4. Model Buidling

# Training a model is like teaching a kid. need a lot of attention. especially in Learning_Rate and Estimators. You can change these paramters and see the prediction. 
# 
# Initially, I have fixed the estimator as 500, the MAE(Mean Absolute Error) was fine. While changing the value little bit, MAE started reducing. finally I fixed this value.

# In[ ]:


param_grid = {'model__nthread':[2], #when use hyperthread, xgboost may become slower:2
              'model__learning_rate': [0.04, 0.05], #so called `eta` value
              'model__max_depth': range(3,5,1), # 3,5,1
              "model__colsample_bytree" : [ 0.2 ],
              'model__silent': [1],
              'model__n_estimators': [800], #number of trees:800
             }
searched_model = GridSearchCV(estimator=train_pipeline,param_grid = param_grid, scoring="neg_mean_absolute_error", cv=5, error_score='raise', verbose = 1)
searched_model.fit(X_full,y)
print(searched_model.best_score_)
# -14495.6467
# -14443.2930 : 900
# -14446.1679 : 1000
# -14438.5401 : 950,4,0.04,0.05 or 950,2, [0.04, 0,05]
# -14435.5775 : 950, 2, [0.04, 0,05], 
# -14435.4461 : 860


# ## 5. Prediction

# In[ ]:


preds_test = searched_model.predict(X_test_full)


# ## 6. Submission

# In[ ]:


output = pd.DataFrame({'Id': X_test_full.index,'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
output


# you like it plz **upvote**, 
# 
# you have suggestions to improve, please comment :) 

# # Happy Learning :)
