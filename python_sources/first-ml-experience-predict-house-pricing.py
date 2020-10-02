#!/usr/bin/env python
# coding: utf-8

# # Predict House Pricing
# work in progres
# <br><br><br>
# Future additions:
# * Cross-validation
# * Method to calculate best parameters for the model
# * better data cleaning (Experimenting with SimpleImputer(), one hot encoding for more columns)
# * more visualizations for categorical data
# 

# In[ ]:


# Import all modules

import numpy as np # linear algebra
import pandas as pd # data processing

# ML model: RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

#Cross Validation
from sklearn.model_selection import cross_val_score

# XGBRegressor
from xgboost import XGBRegressor

# Filling missing values
from sklearn.impute import SimpleImputer

# Visualization
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

print('Import completed')


# In[ ]:


# Load csv files
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')


# # 1. Exploratory data analysis 

# In[ ]:


# First look at the data
print(train_data.shape)
train_data.head()


# As you can see we have a dataset containing 80 columns (with both numerical and categorical values) and 1460 rows (houses).
# <br>The first step for now is to split the dataset in numerical and categorical subsets.
# <br>We will also set our target to be 'SalePrice' since it is the attribute we will want to predict.

# In[ ]:


# Set target which will be predicted later on
target = train_data['SalePrice']

# Splitting data in numerical and categorical subsets
num_attr = train_data.select_dtypes(exclude='object').drop('SalePrice', axis=1).copy()
cat_attr = train_data.select_dtypes(include='object').copy()


# # 1.1 Analyzing numerical attributes

# In[ ]:


# Finding outliers by graphing numerical attributes to SalePrice
plots = plt.figure(figsize=(12,20))

print('Loading 35 plots ...')
for i in range(len(num_attr.columns)-1):
    plots.add_subplot(9, 4, i+1)
    sns.regplot(num_attr.iloc[:,i], target)
    
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# **Outliers (don't follow regression line):**
# * LotFrontage > 200
# * LotArea > 100,000
# * BsmtFinSF1 > 4000
# * TotalBsmtSF > 6000
# * GrLivArea > 4000 + SalePrics < 300,000
# * LowQualFinSF > 550

# # 1.2 Analyzing categorical attributes

# In[ ]:


cat_attr.columns


# In[ ]:


sns.countplot(x='SaleCondition', data=cat_attr)


# # 2. Data Cleaning

# In[ ]:


# Missing values for numerical attributes
num_attr.isna().sum().sort_values(ascending=False).head()


# 259 missing values for LotFrontage --> we will use SimpleImputer() to fill them with averaged values.<br>

# In[ ]:


# Missing values for categorical attributes
cat_attr.isna().sum().sort_values(ascending=False).head(16)


# There are a lot of missing values here.<br>
# We can drop nearly all of them as one hot encoding them won't be useful with this high amount of missing values. (It would dramatically increase the amount of columns in the dataset)
# <br><br>
# * MasVnrType and MasVnrArea both have 8 missing values
# * Electrical --> One hot encoding (pd.get_dummies())
# 

# In[ ]:


# Copy the data to prevent changes to original data
data_copy = train_data.copy()

data_copy.MasVnrArea = data_copy.MasVnrArea.fillna(0)

# Columns which can be filled with 'None'
cat_cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                     'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',
                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',
                     'MasVnrType']
for cat in cat_cols_fill_none:
    data_copy[cat] = data_copy[cat].fillna("None")
    
data_copy.isna().sum().sort_values(ascending=False).head()


# Missing values left in the dataset: LotFrontage (259), GarageYrBuilt (81), Electrical (1)

# In[ ]:


# Dropping outliers found when visualizing the numerical subset of our dataset
data_copy = data_copy.drop(data_copy['LotFrontage'][data_copy['LotFrontage']>200].index)
data_copy = data_copy.drop(data_copy['LotArea'][data_copy['LotArea']>100000].index)
data_copy = data_copy.drop(data_copy['BsmtFinSF1'][data_copy['BsmtFinSF1']>4000].index)
data_copy = data_copy.drop(data_copy['TotalBsmtSF'][data_copy['TotalBsmtSF']>6000].index)
data_copy = data_copy.drop(data_copy['1stFlrSF'][data_copy['1stFlrSF']>4000].index)
data_copy = data_copy.drop(data_copy.GrLivArea[(data_copy['GrLivArea']>4000) & (target<300000)].index)
data_copy = data_copy.drop(data_copy.LowQualFinSF[data_copy['LowQualFinSF']>550].index)

X = data_copy.drop('SalePrice', axis=1)

y = data_copy.SalePrice


numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_attr.columns),
        ('cat', categorical_transformer, cat_attr.columns)
    ])


# # 3. Building two models (RandomForestRegressor and XGBRegressor)

# # 3.1 XGBRegressor

# Parameters for XGBRegressor seem to yield good results. (Future addition: tweaking parameters to fine-tune algorithm)<br>
# Same applys to RandomForestRegressor.

# In[ ]:


xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05)

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', xgb_model)
                             ])
xgb_pipeline.fit(X, y)

scores = -1 * cross_val_score(xgb_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print(scores.mean())


# # 3.2 RandomForestRegressor

# In[ ]:


# Create RandomForestRegressor model, fitting it with train_data and create validation predictions to calculate MAE
rf_model = RandomForestRegressor(n_estimators=50, random_state=1)

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', rf_model)
                             ])
rf_pipeline.fit(X, y)

scores_rf = -1 * cross_val_score(rf_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print(scores_rf.mean())


# **Comparing the MAE of both models we can say, that the XGBRegressor model works better. So we will use this model for our final predictions.**

# In[ ]:


# Applying the same data cleaning we used for the training data to the test data

test_X = test_data.copy()
test_X.MasVnrArea = test_X.MasVnrArea.fillna(0)
test_X = test_X.drop('Id', axis=1)


# # 4. Final predictions and submission

# In[ ]:


test_preds = xgb_pipeline.predict(test_X)
test_preds


# This array contains all predictions. To create the submission.csv file, we will run the next code cell:

# In[ ]:


output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
print('Submitted')


# In[ ]:




