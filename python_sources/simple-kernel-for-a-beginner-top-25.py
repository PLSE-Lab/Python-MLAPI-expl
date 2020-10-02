#!/usr/bin/env python
# coding: utf-8

# This is a very simple solution just to get started with House Regression problem.
# 
# Following are the steps:
# 
# 1. Import data. Combine training data and test data.
# 2. Handle missing data
# 3. Apply log transformation on numerical features
# 4. Get dummies for the categorical features
# 5. Separate training data and test data before fitting the model
# 6. Apply Lasso Regression
# 7. Apply XGB Regression
# 8. Ensemble both models and predict
# 9. Submit data
# 
# Detailed explanation of the steps are given at the respective places. 
# 

# In[ ]:


#import section
import numpy as np # linear algebra
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV


# **Import data. Combine training data and test data**
# 
# Training data and given test data are combined so that 
# * Missing values are handled together
# * Getting dummy values for categorical features (Features with object-string datatype) can be easily handled [To know more about dummy values read, https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding]
# 
# Later, we separate both of them while training.

# In[ ]:


#import and understand data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


#concatenate both train and test data
all_data = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)
#"SalePrice" is the target value. We don't include it in data. We don't want "id" affecting our model. Hence remove it.
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data = all_data.drop(["Id"], axis=1)


# **Handle missing data**
# 
# Now we find columns with missing values and handle each of those. We look into the data description and apply some sensible way to handle the missing values

# In[ ]:


cols_with_missing = [col for col in all_data.columns 
                                 if all_data[col].isnull().any()]
#You can print the cols_with_missing to get better understanding of the columns with missing values
cols_with_missing


# In[ ]:


#Handle missing values one by one
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])
all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna(all_data["Functional"].mode()[0])
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):
    all_data[col] = all_data[col].fillna("None")
for col in ('GarageYrBlt','GarageCars','GarageArea'):
    all_data[col] = all_data[col].fillna(0)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])

#Now check again if there are anymore columns with missing values.
cols_with_missing = [col for col in all_data.columns 
                                 if all_data[col].isnull().any()]
len(cols_with_missing)


# In[ ]:


#"SalePrice" is skewed. This isn't good. It is better to apply log transformation.
sns.distplot(df_train['SalePrice']);


# In[ ]:


df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
sns.distplot(df_train['SalePrice']);


# **Apply log transformation on numerical features**
# 
# Skewed numerical features isn't good while training. 
# You can read about it further here: https://medium.com/@TheDataGyan/day-8-data-transformation-skewness-normalization-and-much-more-4c144d370e55

# In[ ]:


numerical_features = all_data.select_dtypes(exclude = ["object"]).columns
print("Number of numerical features:" + str(len(numerical_features)))

#log transform numerical features
skewness = all_data[numerical_features].apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.7]
skewed_features = skewness.index
all_data[skewed_features] = np.log1p(all_data[skewed_features])


# **Get dummies for the categorical features**
# 
# You will get an error if you try to plug these object values into most machine learning models in Python without "encoding" them first. 
# You can read about it further here: https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding
# 
# And, then we concatenate numerical features in original data and categorical features with dummies

# In[ ]:


categorical_features = all_data.select_dtypes(include = ["object"]).columns
print("Number of categorical features:" + str(len(categorical_features)))

#getdummies for categorical features
#Create a dataFrame with dummy categorical values
dummy_all_data = pd.get_dummies(all_data[categorical_features])
#Remove categorical features from original data, which leaves original data with only numerical featues
all_data.drop(categorical_features, axis=1, inplace=True)
#Concatenate the numerical features in original data and categorical features with dummies
all_data = pd.concat([all_data, dummy_all_data], axis=1)
#print(all_data.shape)


# **Separate training data and test data before fitting the model**
# 
# We have to fit the model only to training data. Hence now we separate them.

# In[ ]:


#Separate training and given test data
X = all_data[:df_train.shape[0]]
test_data = all_data[df_train.shape[0]:]
y = df_train["SalePrice"]


# In[ ]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))    #squared mean error
    return(rmse)


# **Apply Lasso Regression**

# In[ ]:


m_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X, y)
rmse_cv(m_lasso).mean()   


# **Apply XGB Regression**

# In[ ]:


m_xgb = xgb.XGBRegressor(n_estimators=10000, max_depth=5,learning_rate=0.07)
m_xgb.fit(X, y)


# **Ensemble both models and predict**

# In[ ]:


p_xgb = np.expm1(m_xgb.predict(test_data))
p_lasso = np.expm1(m_lasso.predict(test_data))
predicted_prices = 0.75*p_lasso + 0.25*p_xgb
print(predicted_prices)


# **Submit data**

# In[ ]:


my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# **References**:
# 
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 
# https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
# 
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# 
# P.S: I'm a beginner myself. Kindly let me know if there are any mistakes.

# Also for novices like me, you can browse through these simple notes I prepared for scikit learn and ML algos.
# - [Practical scikit Basics](https://www.kaggle.com/nee2shaji/for-novices-practical-scikit-basics-part-1)
# - [ML Algos and tips Part 1](https://www.kaggle.com/nee2shaji/for-novices-ml-algos-how-to-use-them-part-1)
# - [ML Algos and tips Part 2](https://www.kaggle.com/nee2shaji/for-novices-ml-algos-how-to-use-them-part-2)
# - [ML Algos and tips Part 3](https://www.kaggle.com/nee2shaji/for-novices-ml-algos-how-to-use-them-part-3)
