#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing,


# In[ ]:


train_orig = pd.read_csv("../input/train.csv")
test_orig = pd.read_csv("../input/test.csv")


# Combining train and test into a single dataset for further analysis

# In[ ]:


housing = train_orig.append(test_orig, sort = False)
housing = housing.reset_index(drop=True)


# In[ ]:


train_orig.shape


# In[ ]:


test_orig.shape


# Check if there are null values in the data

# In[ ]:


nullval = housing.isnull().sum()
nullval[nullval > 10].sort_values(ascending = False)


# A quick look at the missing values clearly says that missing data follows "MAR" (Missing At Random). 
# "Missing at Random means the propensity for a data point to be missing is not related to the missing data, but it's related to some of the observed data"
# Fireplacequality is missing for the data points that has 0 fire places.
# All the basement related variables are missing the same number of data points close to 80 which indicates the house has no basement. Which in case might be a service apartment.
# Similarly all the garage realted features are missing 159 values except GarageType which has 157 missing data points.
# Veneer type and veneer area also have the same behavior.
# 

# > Handling the missing values:
# * poolQua & Pool area - Remove the variables.(>90% data missing)
# 
# > Imputing the missing values :
# * MiscFeature - NoMisc 
# * Alley - NoAlley
# * Fence - NoFence
# * FirePlaceQu - NoFP 
# * LotFrontage - 0
# * GarageFinish - NoGrg
# * GarageYrBlt - 0
# * GarageCond - NoGrg
# * GarageType - NoGrg
# * BsmtExposure - NoBsmnt
# * BsmtCond     - NoBsmnt
# * BsmtQual     - NoBsmnt
# * BsmtFinType2 - NoBsmnt
# * BsmtFinType1 - NoBsmnt
# * MasVnrType  - Other
# * MasVnrArea  - 0

# In[ ]:


housing_imputed = housing.copy()


# In[ ]:


ColToDrop = ['PoolQC', 'PoolArea', 'SalePrice']
housing_imputed = housing_imputed.drop(ColToDrop, axis = 1)


# In[ ]:


housing_imputed.MiscFeature = housing_imputed.MiscFeature.fillna('NoMisc')
housing_imputed.Alley = housing_imputed.Alley.fillna('NoAlley')
housing_imputed.Fence = housing_imputed.Fence.fillna('Nofnc')
housing_imputed.FireplaceQu = housing_imputed.FireplaceQu.fillna('NoFP')
housing_imputed.LotFrontage = housing_imputed.LotFrontage.fillna(0)
housing_imputed.GarageFinish = housing_imputed.GarageFinish.fillna('NoGrg')
housing_imputed.GarageYrBlt=housing_imputed.GarageYrBlt.fillna(0)
housing_imputed.GarageCond=housing_imputed.GarageCond.fillna('NoGrg')
housing_imputed.GarageType=housing_imputed.GarageType.fillna('NoGrg')
housing_imputed.GarageQual = housing_imputed.GarageQual.fillna(0)
housing_imputed.BsmtExposure=housing_imputed.BsmtExposure.fillna('NoBsmnt')
housing_imputed.BsmtQual=housing_imputed.BsmtQual.fillna('NoBsmnt')
housing_imputed.BsmtCond=housing_imputed.BsmtCond.fillna('NoBsmnt')
housing_imputed.BsmtFinType2=housing_imputed.BsmtFinType2.fillna('NoBsmnt')
housing_imputed.BsmtFinType1=housing_imputed.BsmtFinType1.fillna('NoBsmnt')
housing_imputed.MasVnrType=housing_imputed.MasVnrType.fillna('Other')
housing_imputed.MasVnrArea=housing_imputed.MasVnrArea.fillna(0)


# for all other features which are missing values less than 10, replace those with the mode

# In[ ]:


for col in housing_imputed.columns:
    if(housing_imputed[col].isnull().sum() > 0):
        housing_imputed[col].fillna(housing_imputed[col].mode()[0], inplace=True)


# In[ ]:


housing_imputed = housing_imputed.drop('Id', axis = 1)
housing_imputed = pd.get_dummies(housing_imputed)
train_final = housing_imputed[0:1460]
test_final =  (housing_imputed[1460: 2919]).reset_index(drop= True)
Y = housing.iloc[0:1460,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_final, Y, test_size = 0.3, random_state = 0)


# In[ ]:


from sklearn.linear_model import LinearRegression
lmodel = LinearRegression()
lmodel.fit(X_train, y_train)


# In[ ]:


y_pred = lmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# Trying random Forest Regressor and obtaining feature importance plot

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RFmodel = RandomForestRegressor(random_state=1)
RFmodel.fit(X_train, y_train)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


features = X_train.columns
len(features)


# In[ ]:


features = X_train.columns
importances = RFmodel.feature_importances_
indices = np.argsort(importances)[-20:]  # top 20 features
plt.title('Feature Importance plot')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


y_pred_RF = RFmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

sqrt(mean_squared_error(y_test, y_pred_RF))


# In[ ]:


sqrt(mean_squared_error(y_test, y_pred))


# RMSE for randomforest regressor is less, hence training the entire set using RF regressor and getting the predictions for the test data

# In[ ]:


model = RandomForestRegressor(random_state=1)
model.fit(train_final, Y)
y_out = model.predict(test_final)


# In[ ]:


output = pd.DataFrame(test_orig['Id'])


# In[ ]:


output['SalePrice'] = y_out


# In[ ]:


output.head()

