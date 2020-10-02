#!/usr/bin/env python
# coding: utf-8

# # My score: 11.1444% (hmm... cool number :D)
# # Final | Zhandos Ainabek CSSE-192M [ID=24506]

# ## The objectives of this final
# In this competition we have to build a regression model that predicts the selling price of a house depending on the parameters of X. We have to make feature selection, select those parameters that should be left in the model and which we should get rid of.

# ## Contents
# - Data preparation
# - Feature selection and data cleaning
# - Data preprocessing
# - LineaerRegression algorithm
# - SVR algorithm
# - RandromForestRegressor algorithm
# - DecisionTreeRegressor algorithm
# - XGBRegressor algorithm
# - Submitting results

# ## Data preparation
# Let us import basic libraries in order to start making EDA on our dataset:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Then we have to import our dataset to train our future model:

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# Let us see information about the dataset:

# In[ ]:


train.info()


# We see that our dataset has `81` columns where amount of object columns is half of whole amount of the columns (43). If there were no categorical values we could start building our model without doing dataset preprocessing steps such as one-hot encoding. However in our case we have to perform above operations. Therefore let us start from data cleaning.

# ## Feature selection and data cleaning
# Half amount of columns has categorical values, that is why, it seems that we may drop some of them if they are useless. First we will drop ID column because intuitively this column does not have influence on target column:

# In[ ]:


train.drop(columns=['Id'], inplace=True)


# Then let us look at correlation situation of our dataset:

# In[ ]:


import seaborn as sns
plt.figure(figsize=(20,20))
g = sns.heatmap(train.corr(), annot=True, cmap="RdYlGn")


# We see that our target column has some high correlation values with several columns. Let us keep in mind it. Now we gonna find null values:

# In[ ]:


train.isnull().sum().sort_values(ascending=False)[:30]


# We see that we have about 4 columns that has too much null values. We gonna drop them:

# In[ ]:


has_too_much_null_columns = [
    'PoolQC',
    'MiscFeature',
    'Alley',
    'Fence'
]

train.drop(columns=has_too_much_null_columns, inplace=True)


# For other columns those have less null values, we will fill `median` value for float columns (because median is much better than mean if there are outliers) and fill `0` for object columns.

# In[ ]:


has_null_columns = [
    'FireplaceQu',
    'LotFrontage',
    'GarageCond',
    'GarageType',
    'GarageYrBlt',
    'GarageFinish',
    'GarageQual',
    'BsmtExposure',
    'BsmtFinType2',
    'BsmtFinType1',
    'BsmtCond',
    'BsmtQual',
    'MasVnrArea',
    'MasVnrType',
    'Electrical'
];

for col in has_null_columns:
    if (train[col].dtype == np.object):
        train[col].fillna(0, inplace=True)
    else:
        train[col].fillna(train[col].median(), inplace=True)


# ## Data preprocessing
# Now let us identidy which columns are object columns:

# In[ ]:


object_columns = list(train.select_dtypes(include=['object']).columns)


# And start encoding those object columns (one-hot encoding):

# In[ ]:


train_encoded = pd.get_dummies(train.iloc[:, :-1], columns=object_columns)


# Let us separate our encoded set as independent variables and target variables:

# In[ ]:


X = train_encoded.iloc[:, :-1]
y = train.iloc[:, -1]


# And preparing local train and test variables to evaluate final result:

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)


# ## Linear Regression algorithm
# Let us initialize Linear Regression object, fit our model and predict local values:

# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds_lr = lr.predict(X_test)


# And calculate `R^2` score:

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_preds_lr)


# We also can calculate so called `Adjusted R^2`:

# In[ ]:


def adj_r2(r2score, train):
    return (1 - (1 - r2score) * ((train.shape[0] - 1) / (train.shape[0] - train.shape[1] - 1)))

adj_r2(r2_score(y_test, y_preds_lr), X_train)


# And we can also look at on `Ordinary least squares` summary:

# In[ ]:


import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = y, exog = X).fit()
regressor_OLS.summary()


# In order to compare our actual test values and prediction values we can create method that will help us visualize our results:

# In[ ]:


def comparing_preds_and_test(y_test, y_preds):
    plt.scatter(y_test, y_preds)
    plt.xlabel('y_test')                       
    plt.ylabel('y_preds_lr')
    plt.show()

comparing_preds_and_test(y_test, y_preds_lr)


# Our prediction values are closer to our actual values. This good result so let us move to next algorith.

# ## SVR algorithm
# We gonna perform the same operation as in `Logisitin Regression` section:

# In[ ]:


from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)
y_preds_svr = svr.predict(X_test)


# In[ ]:


r2_score(y_test, y_preds_svr), adj_r2(r2_score(y_test, y_preds_svr), X_train)


# In[ ]:


comparing_preds_and_test(y_test, y_preds_svr)


# The result is not so good as we expect. Maybe our default parameters for SVR is not so good. So maybe we had to use `Grid Search` to find best parameters in order to improve our results.

# ## RandomForestRegressor algorithm

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=2020)
rf.fit(X_train, y_train)
y_preds_rf = rf.predict(X_test)


# In[ ]:


r2_score(y_test, y_preds_rf), adj_r2(r2_score(y_test, y_preds_rf), X_train)


# In[ ]:


comparing_preds_and_test(y_test, y_preds_rf)


# Our prediction values are closer to our actual values. This good result so let us move to next algorithm.

# ## DecisionTreeRegressor algorithm

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=2020)
dt.fit(X_train,y_train)
y_preds_dt = dt.predict(X_test)


# In[ ]:


r2_score(y_test, y_preds_dt), adj_r2(r2_score(y_test, y_preds_dt), X_train)


# In[ ]:


comparing_preds_and_test(y_test, y_preds_dt)


# The results are not so good enough as in `RegressionTreeClassifier`. It is not so surprisingly.

# ## XGBRegressor algorithm

# In[ ]:


from xgboost.sklearn import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_preds_xgb = xgb.predict(X_test)


# In[ ]:


r2_score(y_test, y_preds_xgb), adj_r2(r2_score(y_test, y_preds_xgb), X_train)


# In[ ]:


comparing_preds_and_test(y_test, y_preds_xgb)


# `XGBRegressor` shows good resutls. We could use `GridSearch` in order to make better results for this type of algorithm.

# ## Submitting results
# In order to submit our results by predicting using test variables we will perform all necessary operation on test dataset.

# In[ ]:


test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


test.head()


# In[ ]:


test.drop(columns=has_too_much_null_columns, inplace=True)


# In[ ]:


test_has_null_columns = test.isnull().sum().sort_values(ascending=False)[:30]
test_has_null_columns


# In[ ]:


test_has_null_columns = [
    'FireplaceQu',
    'LotFrontage',
    'GarageCond',
    'GarageQual',
    'GarageYrBlt',
    'GarageFinish',
    'GarageType',
    'BsmtCond',
    'BsmtQual',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'MasVnrType',
    'MasVnrArea',
    'MSZoning',
    'BsmtHalfBath',
    'Utilities',
    'Functional',
    'BsmtFullBath',
    'BsmtUnfSF',
    'SaleType',
    'BsmtFinSF2',
    'BsmtFinSF1',
    'Exterior2nd',
    'Exterior1st',
    'TotalBsmtSF',
    'GarageCars',
    'KitchenQual',
    'GarageArea'
]

for col in test_has_null_columns:
    if (test[col].dtype == np.object):
        test[col].fillna(0, inplace=True)
    else:
        test[col].fillna(test[col].median(), inplace=True)


# In[ ]:


test_object_columns = list(test.select_dtypes(include=['object']).columns)
test_encoded = pd.get_dummies(test, columns=test_object_columns)
test_encoded


# In[ ]:


for col in test_encoded.columns:
    if (col not in X.columns):
        test_encoded.drop(columns=[col], inplace=True)

X.shape, test_encoded.shape


# In[ ]:


for col in X.columns:
    if (col not in test_encoded.columns):
        test_encoded[col] = 0
        
X.shape, test_encoded.shape


# In[ ]:


y_preds_lr_res = lr.predict(test_encoded)
y_preds_svr_res = svr.predict(test_encoded)
y_preds_rf_res = rf.predict(test_encoded)
y_preds_dt_res = dt.predict(test_encoded)


# In[ ]:


i = 0
rows_list = []
for pred in y_preds_lr_res:
    row = {'Id': test["Id"][i], 'SalePrice': pred}
    i += 1
    rows_list.append(row)
df = pd.DataFrame(rows_list) 
df.to_csv("y_preds_lr_res.csv", index=False)

i = 0
rows_list = []
for pred in y_preds_svr_res:
    row = {'Id': test["Id"][i], 'SalePrice': pred}
    i += 1
    rows_list.append(row)
df = pd.DataFrame(rows_list) 
df.to_csv("y_preds_svr_res.csv", index=False)

i = 0
rows_list = []
for pred in y_preds_rf_res:
    row = {'Id': test["Id"][i], 'SalePrice': pred}
    i += 1
    rows_list.append(row)
df = pd.DataFrame(rows_list) 
df.to_csv("y_preds_rf_res.csv", index=False)

i = 0
rows_list = []
for pred in y_preds_dt_res:
    row = {'Id': test["Id"][i], 'SalePrice': pred}
    i += 1
    rows_list.append(row)
df = pd.DataFrame(rows_list) 
df.to_csv("y_preds_dt_res.csv", index=False)


# In[ ]:




