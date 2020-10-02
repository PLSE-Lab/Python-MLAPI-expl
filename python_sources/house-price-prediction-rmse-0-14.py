#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Limit numeric output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 


# In[ ]:


ls ../*


# In[ ]:


# Load data
train_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_raw.shape, test_raw.shape


# ## Some simple EDA

# In[ ]:


train_raw.head()


# In[ ]:


train_raw.info()


# ### Explore missing values

# In[ ]:


# Check missing values in columns
num_missing = train_raw.isnull().mean()
num_missing = num_missing[num_missing != 0].sort_values(ascending = False)
num_missing


# In[ ]:


# Display the distribution of missing value percentages for the variable that have missing values
plt.subplots(figsize=(12, 5))
sns.barplot(num_missing.index, num_missing.values)
plt.xticks(rotation = '45')
plt.show()


# In[ ]:


num_missing = train_raw.isnull().sum()
num_missing[num_missing != 0]


# ### Explore target variable: SalePrice

# In[ ]:


price = train_raw.SalePrice


# In[ ]:


sns.distplot(price)


# The distribution of the sale price is right skewed by some extremely large numbers, apparently deviates from normal distribution. 
# 
# I will take the log of sales price.

# In[ ]:


price.describe()


# The minimum value is greater than 0, thus no need to plus 1 when take the log of sales price.

# In[ ]:


log_price = np.log(price)
sns.distplot(log_price)


# The distribution of the log sale price looks much more like a normal distribution.

# ### Explore numeric features

# In[ ]:


# Look at numeric features
numeric_train = train_raw.select_dtypes(exclude=['object']).drop('Id', axis = 1)
numeric_train.describe()


# Numeric features are varied in scales.

# In[ ]:


# Use heat map to show the correlations between different numeric variables and sales price, 
# as well as the correlations between each pair of themselves.
plt.subplots(figsize = (10,10))
sns.heatmap(numeric_train.corr(), cmap=sns.color_palette('coolwarm'), square=True)
plt.title('Correlations of numeric features')
plt.show()


# * Most of the numeric features are positevely corelated with sale price. 
# * `OverallQual` has the strongest correlation with sale price.
# * Some of the numeric features are correlated with each other.
# 

# ### Explore categorical features

# In[ ]:


# Look at categorical features
cate_train = train_raw.select_dtypes(include=['object'])
cate_train.head()


# In[ ]:


# Count unique values in each categorical feature
unique_count_cate = cate_train.apply(lambda x: len(x.unique())).sort_values(ascending=False)

# Display them in descending order
plt.subplots(figsize = (15, 6))
sns.barplot(unique_count_cate.index, unique_count_cate.values)
plt.title('Number of unique values for each categorical features')
plt.xticks(rotation = '90')
plt.show()


# ## Feature engineering

# In[ ]:


class Engineer:
    def fit(self, X, y = None):
        df = X.copy()
        self.LotFrontage_mean = df.LotFrontage.mean()
        self.ords = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 
                     'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 
                    'Fin': 3, 'RFn': 2, 'Unf': 1, 
                    'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 
                    'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 2.5, 
                    'Shed': 1, 'Gar2': 1, 'TenC': 1, 'Othr': 0.5}
        
       
    def transform(self, X, y = None):
        df = X.copy()
        
        df['LotFrontage'] = df['LotFrontage'].fillna(self.LotFrontage_mean)
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)       
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        
        ordinal_cols = ['PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 
                       'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'Electrical']
        dummy_cols = ['Alley', 'GarageType', 'MasVnrType']
        df[ordinal_cols] = df[ordinal_cols].fillna('None')
        df[dummy_cols] = df[dummy_cols].fillna('None')
        
        for l in ordinal_cols:
            df[l] = df[l].map(lambda x: self.ords.get(x, 0))
            
        df = pd.get_dummies(df, drop_first=True)  
        
        return df
            
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
    


# In[ ]:


Engineer().fit_transform(train_raw).head()


# ### Transform the training data and test data using the Engineer class above

# In[ ]:


# Split the traning data into X (feature matrix) and Y (target variable)
X = train_raw.drop('SalePrice', axis=1)
y = log_price

X.shape, y.shape


# In[ ]:


# Create engineer object to transform data
en = Engineer()
X = en.fit_transform(X)
test = en.transform(test_raw)

# To keep the training and test data have the same columns
X, test = X.align(test, join='outer', axis=1, fill_value=0)

X.shape, test.shape


# ## Model building

# ### Split training data into training set and validation set

# In[ ]:


# Use a 80 vs. 20 split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 2020)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# ### Random Forest

# In[ ]:


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[ ]:


rf = RandomForestRegressor(random_state=2020)
rf.fit(X_train, y_train)


# In[ ]:


preds_train = rf.predict(X_train)
preds_valid = rf.predict(X_valid)


# In[ ]:


print('Training RMSE: {:.3f}'.format(rmse(y_train, preds_train)))
print('Validation RMSE: {:.3f}'.format(rmse(y_valid, preds_valid)))


# ### XGBoost

# In[ ]:


xgb = XGBRegressor(random_state=2020)
xgb.fit(X_train, y_train)

preds_train = xgb.predict(X_train)
preds_valid = xgb.predict(X_valid)


# In[ ]:


print('Training RMSE: {:.3f}'.format(rmse(y_train, preds_train)))
print('Validation RMSE: {:.3f}'.format(rmse(y_valid, preds_valid)))


# XGBoost has a less RMSE on the validation set and less overfitting phenomenon compared with random forest.

# ## Use XGBoost to predict test data

# In[ ]:


# Predict test data
test_results = xgb.predict(test)
test_results = np.exp(test_results)


# In[ ]:


# Save the results to csv file
predict_submission = pd.DataFrame()
predict_submission['Id'] = [1461 + i for i in range(test.shape[0])]
predict_submission['SalePrice'] = test_results


# In[ ]:


predict_submission.to_csv('house_submission.csv', header=True, index = False)


# In[ ]:




