#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)

from sklearn.preprocessing import LabelEncoder,StandardScaler,RobustScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV,train_test_split,KFold
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import statsmodels.regression.linear_model as sm
import xgboost as xgb
from sklearn.pipeline import make_pipeline


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index,inplace=True)


# In[ ]:


train.dtypes.value_counts()


# In[ ]:


test.dtypes.value_counts()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape,test.shape


# In[ ]:


train.drop('Id',axis=1,inplace=True)
Submission = test[['Id']]
test.drop('Id',axis=1,inplace=True)


# In[ ]:


train_missing = pd.DataFrame(train.isna().sum()[train.isna().sum() !=0].sort_values())
train_missing.columns = ['#Missing']
train_missing['Percent_Missing'] = train.isna().sum()[train.isna().sum() !=0]/len(train)
train_missing


# In[ ]:


test_missing = pd.DataFrame(test.isna().sum()[test.isna().sum() !=0].sort_values())
test_missing.columns = ['#Missing']
test_missing['Percent_Missing'] = test.isna().sum()[test.isna().sum() !=0]/len(test)
test_missing


# In[ ]:


train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)


# In[ ]:





# Some of the variables have different number of Unique values in the train data as compared to the test data and vice versa, this should be sorted out because when we create dummy variables and then train the model on the train data, due to differences in the unique values, the model throws an error message when we run it on the test data.

# In[ ]:


for col in train.columns:
    if train[col].dtype == "object":
        print ("Cardinality of %s variable in Train Data:"%col,train[col].nunique())
        print ("Cardinality of %s variable in Test Data:"%col,test[col].nunique())
        print ("\n")


# From the above results, **Utilities, Condition2, HouseStyle, RoofMatl, Exterior1st, Exterior2nd, Heating, Electrical, GarageQual** variables have different unique values in Train and Test data. So either some of the values need to be combined or some values have to be dropped depending on the needs.

# **EDA: Univariate, Multivariate Analysis, Correlation Analysis**

# In[ ]:


train['SalePrice'].describe()


# There are no missing values in the dependent variable. Minimum price is 34900 and maximum is 755000, the 75% percentile value or the third quartile value is 214000. Considering this, the maximum value seems to be an outlier.

# In[ ]:


sns.distplot(train['SalePrice'])
plt.title("Distribution of Sale Price variable")
plt.xlabel("Price")


# The distribution does not look normal, it is positively skewed, some outliers can also be seen. Simple log transformation might change the distribution to normal.

# In[ ]:


sns.boxplot(train['SalePrice'],orient='vert')


# There are some values that are quite different from the rest. It makes sense to delete these variables as the Linear regression methods are very sensitive to outliers. 

# In[ ]:


train.drop(train.index[[691,1182]],inplace=True)
train['SalePrice'] = np.log(train['SalePrice'])


# In[ ]:


sns.distplot(train['SalePrice'])


# Now the distribution looks normal. 

# In[ ]:


train['MSSubClass'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


test['MSSubClass'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses are single storied 1946 type followed by 2 storied 1946 type.

# In[ ]:


plt.figure(figsize=(8,8))
sns.boxplot(x=train['MSSubClass'],y=train['SalePrice'])


# No particular anomalies can be seen here.

# In[ ]:


train.groupby('MSSubClass')['SalePrice'].mean().sort_values().plot(kind='bar')


# MSSubClass variable can be converted to str data type as it is not a continuous numerical variables and it represents a category.

# In[ ]:


train['MSSubClass'] = train['MSSubClass'].astype(str)
test['MSSubClass'] = test['MSSubClass'].astype(str)


# In[ ]:





# In[ ]:


train['MSZoning'].value_counts(dropna=False).sort_values().plot(kind='bar')


# 85% of the houses belong to Residential Low Density type of neighborhood.

# In[ ]:


sns.boxplot(x=train['MSZoning'],y=train['SalePrice'])


# In[ ]:


train.groupby('MSZoning')['SalePrice'].mean().sort_values().plot(kind='bar')


# In[ ]:


test['MSZoning'].fillna(test['MSZoning'].value_counts(dropna=False).index[0],inplace=True)


# In[ ]:





# In[ ]:


train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:





# In[ ]:


train['LotArea'].describe()


# In[ ]:


sns.scatterplot(x=train['LotArea'],y=train['SalePrice'])


# One can notice many values that are way above the group. It can also be noted that these two variables are not linearly related.

# In[ ]:





# In[ ]:


train['Street'].value_counts(dropna=False)


# In[ ]:


test['Street'].value_counts(dropna=False)


# Since the variable is dominated by a single value in both Train and Test data, it doesn't impart much information about the dependent variable. Hence it makes sense to delete the variable.

# In[ ]:


train.drop('Street',axis=1,inplace=True)
test.drop('Street',axis=1,inplace=True)


# In[ ]:





# In[ ]:


train['LotShape'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


test['LotShape'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


sns.boxplot(x=train['LotShape'],y=train['SalePrice'])


# In[ ]:





# In[ ]:


train['LandContour'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the Houses are flat.

# In[ ]:


test['LandContour'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


sns.boxplot(train['LandContour'],train['SalePrice'])


# In[ ]:





# In[ ]:


train['Utilities'].value_counts(dropna=False)


# In[ ]:


test['Utilities'].value_counts(dropna=False)


# In the Utilities variable again, just one value "AllPub" is dominant.After replacing the missing value in test data, the variable will contain just one value that is AllPub which will amount to no information towards the dependent variable. Hence deleting it makes more sense.

# In[ ]:


train.drop('Utilities',axis=1,inplace=True)
test.drop('Utilities',axis=1,inplace=True)


# In[ ]:





# In[ ]:


train['LotConfig'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses are are Inside Lot type followed by Corner plot.

# In[ ]:


sns.boxplot(x=train['LotConfig'],y=train['SalePrice'])


# In[ ]:





# Since LandContour and LandSlope talk about the Flatness of the property or the Slope of the property, it doesn't make sense to keep both the variables, hence dropping LandSlope variable.

# In[ ]:


train.drop('LandSlope',axis=1,inplace=True)
test.drop('LandSlope',axis=1,inplace=True)


# In[ ]:





# In[ ]:


train['Neighborhood'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


plt.figure(figsize=(12,12))
sns.boxplot(x=train['Neighborhood'],y=train['SalePrice'])


# No particular relation evident from the boxplot.

# In[ ]:


train.groupby('Neighborhood')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:





# In[ ]:


train['Condition1'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses are near to various facilities like parks, railroads, feeder streets etc.No Missing values here.

# In[ ]:


test['Condition1'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


train.groupby('Condition1')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# If we analyse the meaning of these two variables Condition1 and Condition2, they basically talk about proximity of the properties to various facilities, so it doesn't matter if there is one additional facility in the near proximity. So instead of keeping both Condition1 and Condition2, we can just drop Condition2 and keep Condition1. 

# In[ ]:


train.drop('Condition2',axis=1,inplace=True)
test.drop('Condition2',axis=1,inplace=True)


# In[ ]:





# In[ ]:


train['BldgType'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses are Single-Family Detached type 

# In[ ]:


test['BldgType'].value_counts(dropna=False)


# In[ ]:


sns.boxplot(x=train['BldgType'],y=train['SalePrice'])


# In[ ]:





# In[ ]:


train['HouseStyle'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses are 1Story/2Story. Some of the houses have One and Half story with 2nd level unfinished etc. Some of them can be combined.

# In[ ]:


test['HouseStyle'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


train['HouseStyle'].value_counts(dropna=False)


# In[ ]:


test['HouseStyle'].value_counts(dropna=False)


# In[ ]:





# In[ ]:


train['OverallQual'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses have a medium rating for overall material and finish of the house followed by 6,7,8 and 9 rating.

# In[ ]:


sns.boxplot(train['OverallQual'],train['SalePrice'])


# It can be seen from the Boxplot that as the rating increases, the Sale Price of the house increases. Some outliers can be seen here.

# In[ ]:


sns.scatterplot(train['OverallQual'],train['SalePrice'])


# In[ ]:


train.groupby('OverallQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:


#train['OverallQual'] = train['OverallQual'].astype(str)
#test['OverallQual'] = test['OverallQual'].astype(str)


# In[ ]:





# In[ ]:


train['OverallCond'].value_counts(dropna=False).sort_values().plot(kind='bar')


# 1. More than 50% of the houses have average (rating of 5) overall condition of the house.
# 2. There is no house with a rating of 10.
# 3. Very few have a rating of 9.
# 4. OverallCond does not have the same affect as that of OverallQual towards SalePrice variable.

# In[ ]:


sns.boxplot(train['OverallCond'],train['SalePrice'])


# Sale price also increases with the rating of Overall Condition of the house.

# In[ ]:


train.groupby('OverallCond')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# Although very few houses have a rating of 9, their average Sale price is very high.

# In[ ]:





# In[ ]:


plt.figure(figsize=(14,14))
sns.boxplot(train['YearBuilt'],train['SalePrice'])


# Houses that were built recently have higher prices as compared to others.

# In[ ]:


train['YearBuilt'].value_counts(dropna=False).sort_values(ascending=False).head()


# Many houses were built in 2005 and 2006 as compared to other years.

# In[ ]:





# In[ ]:


plt.figure(figsize=(14,14))
sns.boxplot(train['YearRemodAdd'],train['SalePrice'])


# In[ ]:


train['YearRemodAdd'].value_counts(dropna=False).sort_values(ascending=False).head()


# More than 10% of the houses were modified /rebuilt in the year 1950 as compared to others.

# In[ ]:





# In[ ]:


train['RoofStyle'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses have Gable type of roof followed by Hip type.

# In[ ]:


sns.boxplot(train['RoofStyle'],train['SalePrice'])


# In[ ]:





# In[ ]:


train['RoofMatl'].value_counts(dropna=False)


# In[ ]:


test['RoofMatl'].value_counts(dropna=False)


# In[ ]:





# In[ ]:


train['Exterior1st'].value_counts(dropna=False)


# In[ ]:


test['Exterior1st'].value_counts(dropna=False)


# In[ ]:


test['Exterior1st'].fillna(test['Exterior1st'].value_counts(dropna=False).index[0],inplace=True)


# In[ ]:


train['Exterior1st'].nunique(),test['Exterior1st'].nunique()


# In[ ]:





# In[ ]:


train['Exterior2nd'].value_counts(dropna=False)


# In[ ]:


test['Exterior2nd'].value_counts(dropna=False)


# In[ ]:


test['Exterior2nd'].fillna(test['Exterior2nd'].value_counts(dropna=False).index[0],inplace=True)


# In[ ]:


train['Exterior2nd'].nunique(),test['Exterior2nd'].nunique()


# In[ ]:





# In[ ]:





# In[ ]:


train['MasVnrType'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses do not have Masonry Veneer(like a facade). Some missing values.

# In[ ]:


train['MasVnrType'].fillna("None",inplace=True)
test['MasVnrType'].fillna("None",inplace=True)


# Some of the values of MasVnrType are None suggesting there is no Masonry Veneer, now the missing values are imputed with this None values, similarly, the missing values in MasVnrArea should be filled with 0.

# In[ ]:





# In[ ]:


train['MasVnrArea'].fillna(0,inplace=True)
test['MasVnrArea'].fillna(0,inplace=True)


# In[ ]:


sns.scatterplot(train['MasVnrArea'],train['SalePrice'])


# There is no relationship evident here.

# In[ ]:





# In[ ]:


train['ExterQual'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses have Average / Typical external quality of the house, this is followed by houses that are rated Good. No Missing values here.

# In[ ]:


sns.boxplot(train['ExterQual'],train['SalePrice'])


# The price of houses that have Excellent external quality rating are much higher as compared to others. 

# In[ ]:


train.groupby('ExterQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:





# In[ ]:


train['ExterCond'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses have either Typical or Good condition of the material on the exterior.

# In[ ]:


sns.boxplot(train['ExterCond'],train['SalePrice'])


# In[ ]:


train.groupby('ExterCond')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# Here too, the houses that have Excellent rating have higher average although the count is way too low.

# In[ ]:





# In[ ]:


train['Foundation'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most houses have Poured Concrete or Concrete Block as their Foundation.

# In[ ]:


sns.boxplot(train['Foundation'],train['SalePrice'])


# Poured Concrete has higher values of Sale Prices.

# In[ ]:


train.groupby('Foundation')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:





# In[ ]:


train['BsmtQual'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Missing values in the BsmtQual variable indicate there is no basement in the house. So it is imputed with "No_Basement".Also most of the houses have Typical or Good height of the basement.

# In[ ]:


train['BsmtQual'].fillna("No_Basement",inplace=True)
test['BsmtQual'].fillna("No_Basement",inplace=True)


# In[ ]:


train.groupby('BsmtQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# Higher prices for Excellent height of the basement.

# In[ ]:


train['BsmtCond'].value_counts(dropna=False)


# Most of the houses have Typical basement conditions with slight dampness allowed. Missing values are imputed with "No_Basement"

# In[ ]:


train['BsmtCond'].fillna("No_Basement",inplace=True)
test['BsmtCond'].fillna("No_Basement",inplace=True)


# In[ ]:


train['BsmtExposure'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


train['BsmtExposure'].fillna("No_Basement",inplace=True)
test['BsmtExposure'].fillna("No_Basement",inplace=True)


# In[ ]:


train['BsmtFinType1'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses have unfinished Basement, followed by Good Living Quarters. Some missing values are imputed accordingly.

# In[ ]:


train['BsmtFinType1'].fillna("No_Basement",inplace=True)
test['BsmtFinType1'].fillna("No_Basement",inplace=True)


# In[ ]:





# In[ ]:


test['BsmtFinSF1'].fillna(0,inplace=True)


# In[ ]:


sns.scatterplot(train['BsmtFinSF1'],train['SalePrice'])


# The relationship is not linear, there is one very high value of Basement Finished area, and 0 values indicate that these houses do not have a basement.

# In[ ]:


train['BsmtFinType2'].fillna("No_Basement",inplace=True)
test['BsmtFinType2'].fillna("No_Basement",inplace=True)


# In[ ]:


sns.scatterplot(train['BsmtFinSF2'],train['SalePrice'])


# Values are much more scattered as compared to the scatterplot of BsmtFinSF1 and SalePrice, again 0 values indicate No Basement.

# In[ ]:


test['BsmtFinSF2'].fillna(0,inplace=True)


# In[ ]:


test['BsmtUnfSF'].fillna(0,inplace=True)


# In[ ]:


sns.scatterplot(train['BsmtUnfSF'],train['SalePrice'])


# No Missing values here.

# In[ ]:


train['TotalBsmtSF'].isna().sum(),test['TotalBsmtSF'].isna().sum()


# In[ ]:


test['TotalBsmtSF'].fillna(0,inplace=True)


# In[ ]:


sns.scatterplot(train['TotalBsmtSF'],train['SalePrice'])


# In[ ]:





# In[ ]:


train['Heating'].value_counts(dropna=False)


# Most of the houses have Gas forced warm air furnace type of Heating. 

# In[ ]:


test['Heating'].value_counts(dropna=False)


# Since there are unequal number of unique values in train and test data, we have to combine some categories.

# In[ ]:


train.groupby('Heating')['SalePrice'].agg(['count','min','max','mean'])


# In[ ]:





# In[ ]:


train['HeatingQC'].isna().sum(),test['HeatingQC'].isna().sum()


# In[ ]:


train['HeatingQC'].value_counts().sort_values().plot(kind='bar')


# In[ ]:


sns.boxplot(train['HeatingQC'],train['SalePrice'])


# In[ ]:





# In[ ]:


train['CentralAir'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses have Central Air conditioning.

# In[ ]:


train.groupby('CentralAir')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:





# In[ ]:


train['Electrical'].value_counts(dropna=False)


# Most of the houses have Standard Circuit Breakers & Romex type of electrical system. There are some missing values. 

# In[ ]:


test['Electrical'].value_counts(dropna=False)


# In[ ]:


train['Electrical'].fillna(train['Electrical'].value_counts(dropna=False).index[0],inplace=True)


# In[ ]:





# In[ ]:


train['1stFlrSF'].describe()


# In[ ]:


sns.scatterplot(train['1stFlrSF'],train['SalePrice'])


# The relationship is slightly linear, there is one distinct value that stands out from the rest. No Missing values here.

# In[ ]:


train.drop(train.index[1300],inplace=True)


# In[ ]:


train['2ndFlrSF'].describe()


# In[ ]:


len(train[train['2ndFlrSF'] == 0])/len(train)


# 56% of the houses do not have 2nd floor. 

# In[ ]:


sns.scatterplot(train['2ndFlrSF'],train['SalePrice'])


# Now this is a different scatterplot. Lot of 0 values suggesting that there is no 2nd floor in the house. It consists of Ground plus 1. That apart, the points are relatively scattered and somewhat linear.

# In[ ]:





# In[ ]:


sns.scatterplot(train['LowQualFinSF'],train['SalePrice'])


# Since most of the values are 0 in this variable, it makes sense to drop the variable instead of increasing the complexity by adding one more variable.

# In[ ]:


train.drop('LowQualFinSF',axis=1,inplace=True)
test.drop('LowQualFinSF',axis=1,inplace=True)


# In[ ]:





# In[ ]:


sns.scatterplot(train['GrLivArea'],train['SalePrice'])


# It has a linear relationship with the dependent variable. There are two distinct values that stand out from the rest. 

# In[ ]:


train['GrLivArea'].isna().sum(),test['GrLivArea'].isna().sum()


# In[ ]:





# In[ ]:


train['BsmtFullBath'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses have 0 Full bathrooms in the basement and some have 1. No missing values here.

# In[ ]:


sns.boxplot(train['BsmtFullBath'],train['SalePrice'])


# In[ ]:


train.groupby('BsmtFullBath')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:


test['BsmtFullBath'].fillna(0,inplace=True)


# In[ ]:





# In[ ]:


train['BsmtHalfBath'].value_counts(dropna=False).sort_values().plot(kind='bar')


# 95% of the houses have 0 half bathrooms in the basement. Some have 1. 

# In[ ]:


test['BsmtHalfBath'].fillna(0,inplace=True)


# In[ ]:





# In[ ]:


train['FullBath'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Majority of the houses have either 1 or 2 full bathrooms above grade. 

# In[ ]:


sns.boxplot(train['FullBath'],train['SalePrice'])


# In[ ]:


train['HalfBath'].value_counts(dropna=False)


# In[ ]:


test['HalfBath'].value_counts(dropna=False)


# In[ ]:





# In[ ]:


train['BedroomAbvGr'].value_counts(dropna=False).sort_values().plot(kind='bar')


# In[ ]:


sns.boxplot(train['BedroomAbvGr'],train['SalePrice'])


# In[ ]:





# In[ ]:


train['KitchenAbvGr'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most houses have either 1 or 2 kitchens above ground.

# In[ ]:





# In[ ]:


train['KitchenQual'].value_counts(dropna=False).sort_values().plot(kind='bar')


# 90% of the houses have either Typical or Good quality Kitchens.

# In[ ]:


train.groupby('KitchenQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:


test['KitchenQual'].fillna(test['KitchenQual'].value_counts(dropna=False).index[0],inplace=True)


# In[ ]:





# In[ ]:


train['TotRmsAbvGrd'].value_counts(dropna=False).sort_index()


# In[ ]:


sns.boxplot(train['TotRmsAbvGrd'],train['SalePrice'])


# Prices increase till a certain threshold (11 rooms) and then decrease.

# In[ ]:





# In[ ]:


train['Functional'].value_counts(dropna=False).sort_values()


# In[ ]:


test['Functional'].fillna(test['Functional'].value_counts(dropna=False).index[0],inplace=True)


# In[ ]:


sns.boxplot(train['Functional'],train['SalePrice'])


# Prices are same across most of the categories.

# In[ ]:





# In[ ]:


train['Fireplaces'].value_counts(dropna=False).sort_index()


# 0 values indicate No Fireplace in the house. If not 0, most of the houses have a maximum of 1 fireplace.

# In[ ]:


sns.scatterplot(train['Fireplaces'],train['SalePrice'])


# In[ ]:


train.groupby('Fireplaces')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:





# In[ ]:


train['GarageType'].isna().sum(),test['GarageType'].isna().sum()


# In[ ]:


train['GarageType'].value_counts(dropna=False).sort_values().plot(kind='bar')


# Most of the houses have Attached type of Garage followed by Detached type. There are some missing values in both Train and Test data (Probably because these houses do not have a garage).

# In[ ]:


sns.boxplot(train['GarageType'],train['SalePrice'])


# In[ ]:


train['GarageType'].fillna("No_Garage",inplace=True)
test['GarageType'].fillna("No_Garage",inplace=True)


# In[ ]:


train.groupby('GarageType')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# Built in Garage Types have higher average SalePrice.

# In[ ]:





# In[ ]:


train['GarageYrBlt'].isna().sum(),test['GarageYrBlt'].isna().sum()


# Missing values indicate the absence of Garage in the house. Will replace it with an arbitary value of 0.

# In[ ]:


train['GarageYrBlt'].fillna(0,inplace=True)
test['GarageYrBlt'].fillna(0,inplace=True)


# In[ ]:


train['GarageYrBlt'].value_counts().sort_values(ascending=False).head()


# In[ ]:


plt.figure(figsize=(14,14))
sns.boxplot(train['GarageYrBlt'],train['SalePrice'])


# Garages buit recently have higher prices. 

# In[ ]:


test.loc[1132,'GarageYrBlt'] = 2007


# In[ ]:





# In[ ]:


train['GarageFinish'].isna().sum(),test['GarageFinish'].isna().sum()


# In[ ]:


train['GarageFinish'].fillna("No_Garage",inplace=True)
test['GarageFinish'].fillna("No_Garage",inplace=True)


# In[ ]:


train.groupby('GarageFinish')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:


sns.boxplot(train['GarageFinish'],train['SalePrice'])


# Finished Garages have higher prices as compared to Unfinished Garages.

# In[ ]:





# In[ ]:


train['GarageCars'].isna().sum(),test['GarageCars'].isna().sum()


# A value of 0 in GarageCars variable means there is no Garage in the House. The missing value in Testd ata is imputed.

# In[ ]:


test['GarageCars'].fillna(0,inplace=True)


# In[ ]:


train['GarageCars'].value_counts().sort_values().plot(kind='bar')


# Most of the houses can accomodate 2 cars followed by 1 and 3. Few can accomodate 4.

# In[ ]:


sns.boxplot(train['GarageCars'],train['SalePrice'])


# Price increases with increase in accomodating capacity of the cars in the 

# In[ ]:





# In[ ]:


train['GarageArea'].isna().sum(),test['GarageArea'].isna().sum()


# In[ ]:


train['GarageArea'].describe()


# In[ ]:


sns.scatterplot(train['GarageArea'],train['SalePrice'])


# 0 values indicate absence of Garage in the house. Another point to be noted here is that GarageCars and GarageArea are correlated with each other in a sense that, larger the area, larger is the number of cars that can be accomodated. This should be investigated in the correlation matrix.

# In[ ]:


test['GarageArea'].fillna(0,inplace=True)


# In[ ]:





# In[ ]:


train['GarageQual'].isna().sum(),test['GarageQual'].isna().sum()


# In[ ]:


train['GarageQual'].fillna("No_Garage",inplace=True)
test['GarageQual'].fillna("No_Garage",inplace=True)


# In[ ]:


train['GarageQual'].value_counts().sort_values(ascending=False)


# In[ ]:


test['GarageQual'].value_counts().sort_values(ascending=False)


# Most of the houses have Typical/Average Garage Quality rating.

# In[ ]:


train.groupby('GarageQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')


# In[ ]:





# In[ ]:


train['GarageCond'].isna().sum(),test['GarageCond'].isna().sum()


# In[ ]:


train['GarageCond'].fillna("No_Garage",inplace=True)
test['GarageCond'].fillna("No_Garage",inplace=True)


# In[ ]:


train['GarageCond'].value_counts().sort_values().plot(kind='bar')


# In[ ]:


sns.boxplot(train['GarageCond'],train['SalePrice'])


# In[ ]:





# In[ ]:


train['PavedDrive'].isna().sum(),test['PavedDrive'].isna().sum()


# In[ ]:


train['PavedDrive'].value_counts()


# Most of the houses have Paved Driveway.

# In[ ]:


test['PavedDrive'].value_counts()


# In[ ]:





# In[ ]:


train['WoodDeckSF'].isna().sum(),test['WoodDeckSF'].isna().sum()


# In[ ]:


sns.scatterplot(train['WoodDeckSF'],train['SalePrice'])


# 0 values indicate there is no wood deck area.

# In[ ]:





# In[ ]:


train['OpenPorchSF'].isna().sum(),test['OpenPorchSF'].isna().sum()


# In[ ]:


sns.scatterplot(train['OpenPorchSF'],train['SalePrice'])


# Some outliers are visible, again 0 values indicate the absence of Open Porch in the house.

# In[ ]:





# In[ ]:


train['EnclosedPorch'].isna().sum(),test['EnclosedPorch'].isna().sum()


# In[ ]:


sns.scatterplot(train['EnclosedPorch'],train['SalePrice'])


# The relationship is non linear with lot of 0 values and one outlier clearly visible.

# In[ ]:





# In[ ]:


train['3SsnPorch'].isna().sum(),test['3SsnPorch'].isna().sum()


# In[ ]:


sns.scatterplot(train['3SsnPorch'],train['SalePrice'])


# In[ ]:





# In[ ]:


train['ScreenPorch'].isna().sum(),test['ScreenPorch'].isna().sum()


# In[ ]:


sns.scatterplot(train['ScreenPorch'],train['SalePrice'])


# Lot of 0 values and no particular relationship with dependent variable is evident.

# In[ ]:





# In[ ]:


train['PoolArea'].isna().sum(),test['PoolArea'].isna().sum()


# In[ ]:


sns.scatterplot(train['PoolArea'],train['SalePrice'])


# Since the other variable related to Pool is already deleted, we will keep this variable. Although 99% of the houses do not have a pool.

# In[ ]:





# In[ ]:


sns.scatterplot(train['MiscVal'],train['SalePrice'])


# In[ ]:





# In[ ]:


train['MoSold'].isna().sum(),test['MoSold'].isna().sum()


# In[ ]:


train['MoSold'].value_counts().sort_values().plot(kind='bar')


# Lot of houses are sold in the Months of May, June and July

# In[ ]:


sns.boxplot(train['MoSold'],train['SalePrice'])


# Prices are similar irrespective of the month sold.

# In[ ]:


train.groupby('MoSold')['SalePrice'].agg(['count','min','max','mean']).sort_index()


# In[ ]:


train['MoSold'] = train['MoSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)


# In[ ]:





# In[ ]:


train['YrSold'].isna().sum(),test['YrSold'].isna().sum()


# In[ ]:


train['YrSold'].value_counts().sort_values(ascending=False).head(10)


# In[ ]:


sns.boxplot(train['YrSold'],train['SalePrice'])


# In[ ]:


train.groupby(['MoSold','YrSold'])['SalePrice'].count().sort_values(ascending=False)


# In[ ]:


train['YrSold'] = train['YrSold'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)


# In[ ]:





# In[ ]:


train['SaleType'].isna().sum(),test['SaleType'].isna().sum()


# In[ ]:


test['SaleType'].fillna(test['SaleType'].value_counts().index[0],inplace=True)


# In[ ]:


train['SaleType'].value_counts().sort_values().plot(kind='bar')


# In[ ]:


test['SaleType'].value_counts().sort_values().plot(kind='bar')


# In[ ]:





# In[ ]:


train['SaleCondition'].isna().sum(),test['SaleCondition'].isna().sum()


# In[ ]:


train['SaleCondition'].value_counts().sort_values().plot(kind='bar')


# In[ ]:


test['SaleCondition'].value_counts().sort_values().plot(kind='bar')


# In[ ]:





# Let's check if there are anymore missing values in Train and Test sets, and if there is difference in the number of unique values of variables in Train and Test sets.

# In[ ]:


#train['Total_sqr_footage'] = (train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF'])
train['Total_Bathrooms'] = (train['FullBath'] + (0.5*train['HalfBath']) +  train['BsmtFullBath'] + (0.5*train['BsmtHalfBath']))
train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])

#test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF'])
test['Total_Bathrooms'] = (test['FullBath'] + (0.5*test['HalfBath']) +  test['BsmtFullBath'] + (0.5*test['BsmtHalfBath']))
test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])


# In[ ]:


train.isna().sum()[train.isna().sum() !=0]


# In[ ]:


test.isna().sum()[test.isna().sum() !=0]


# In[ ]:


for col in train.columns:
    if train[col].dtype == "object":
        print ("Cardinality of %s variable in Training Data:"%col,train[col].nunique())
        print ("Cardinality of %s variable in Testing Data:"%col,test[col].nunique())
        print ("\n")


# In[ ]:


## Plot sizing. 
plt.subplots(figsize = (35,20))
## plotting heatmap.  
sns.heatmap(train.corr(), cmap="BrBG", annot=True, center = 0, );
## Set the title. 
plt.title("Heatmap of all the Features", fontsize = 30);


# In[ ]:


train.corr()['SalePrice'].sort_values(ascending=False)


# As we can see from the heatmap above, some of the Independent variables are correlated with each other. One of the basic assumptions of Linear regression is that the independent variables are not correlated with each other, if they are then we will not know which of the two variables is imparting the correct information about the Dependent variable. So if 2 variables have high correlation between them, we will have to drop one of the variable and keep the other. 

# As we can see from the heatmap above, some of the Independent variables are correlated with each other. One of the basic assumptions of Linear regression is that the independent variables are not correlated with each other, if they are then we will not know which of the two variables is imparting the correct information about the Dependent variable. So if 2 variables have high correlation between them, we will have to drop one of the variable and keep the other. 

# In[ ]:


train.drop(['GarageCars','TotRmsAbvGrd','1stFlrSF'],axis=1,inplace=True)
test.drop(['GarageCars','TotRmsAbvGrd','1stFlrSF'],axis=1,inplace=True)


# In[ ]:


train.corr()['SalePrice'].sort_values(ascending=False)


# In[ ]:


# saleprice correlation matrix
corr_num = 20 #number of variables for heatmap
cols_corr = train.corr().nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(train[cols_corr].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(15,15))
hm = sns.heatmap(corr_mat_sales, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",annot_kws = {'size':12}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()


# In[ ]:


train.shape, test.shape


# In[ ]:


trn = train.copy()
tst = test.copy()


# In[ ]:


y = train['SalePrice']
train.drop('SalePrice',inplace=True,axis=1)
len_train = len(train)
full_data = pd.concat([train,test],axis=0).reset_index(drop=True)
full_data = pd.get_dummies(full_data)
train = full_data[:len_train]
test = full_data[len_train:]


# In[ ]:


train.shape, test.shape


# In[ ]:


set(train.columns)-set(test.columns)


# In[ ]:


overfit = []
for col in train.columns:
    counts = train[col].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(train) * 100 >99:
        overfit.append(col)
print (len(overfit))


# In[ ]:


train.drop(overfit,axis=1,inplace=True)
test.drop(overfit,axis=1,inplace=True)


# **Model Building**

# In[ ]:


X = train


# In[ ]:


X_Train,X_Test,y_Train,y_Test = train_test_split(X,y,test_size=0.25)
print ("X_Train Shape:",X_Train.shape)
print ("X_Test Shape:",X_Test.shape)
print ("y_Train Shape:",y_Train.shape)
print ("y_Test Shape:",y_Test.shape)


# In[ ]:


kfolds = KFold(n_splits=5, shuffle=True)
def rmse(true,pred):
    return np.sqrt(mean_squared_error(true,pred))
def feature_importance(model,data):
    ser = pd.Series(model.coef_,data.columns).sort_values()
    plt.figure(figsize=(14,14))
    ser.plot(kind='bar')
def cv_score(model):
    return np.mean(np.sqrt(-(cross_val_score(model,X,y,cv=kfolds,scoring='neg_mean_squared_error'))))
def plot_importance(model,indep):
    Ser = pd.Series(model.coef_,indep.columns).sort_values()
    plt.figure(figsize=(30,20))
    Ser.plot(kind='bar')
def calc_r2(model,true,data):
    return r2_score(true,model.predict(data))


# In[ ]:





# **Linear Regression**

# In[ ]:


lr = LinearRegression()
lr.fit(X_Train,y_Train)
print ("Linear Regression, Training Set RMSE:",rmse(y_Train,lr.predict(X_Train)))
print ("Linear Regression,Training R Squared:",calc_r2(lr,y_Train,X_Train))
print ("\nLinear Regression,Testing Set RMSE:",rmse(y_Test,lr.predict(X_Test)))
print ("Linear Regression,Testing R Squared:",calc_r2(lr,y_Test,X_Test))
print ("\nLinear Regression,Cross Validation Score:",cv_score(lr))


# **Lasso Regression**

# In[ ]:


lasso = Lasso(alpha=0.001,max_iter=5000)
lasso.fit(X_Train,y_Train)
print ("Lasso Regression, Training Set RMSE:",rmse(y_Train,lasso.predict(X_Train)))
print ("Lasso Regression,Training R Squared:",calc_r2(lasso,y_Train,X_Train))
print ("\nLasso Regression,Testing Set RMSE:",rmse(y_Test,lasso.predict(X_Test)))
print ("Lasso Regression,Testing R Squared:",calc_r2(lasso,y_Test,X_Test))
print ("\nLasso Regression,Cross Validation Score:",cv_score(lasso))


# In[ ]:


#Submission['SalePrice'] = np.expm1(lasso.predict(test))
#Submission.to_csv("Lasso_Latest.csv",index=None)


# In[ ]:


coeffs = pd.DataFrame(list(zip(X.columns, lasso.coef_)), columns=['Predictors', 'Coefficients'])
coeffs.sort_values(by='Coefficients',ascending=False)


# In[ ]:


scores = []
alpha = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10]
for i in alpha:
    las = Lasso(alpha=i,max_iter=10000)
    las.fit(X_Train,y_Train)
    scores.append(rmse(y_Test,las.predict(X_Test)))
print ("Lasso Scores with Different Alpha Values \n",scores)


# In[ ]:


#Submission['SalePrice'] = np.expm1(lasso.predict(test))
#Submission.to_csv("Sub.csv",index=None)


# **Ridge Regression**

# In[ ]:


ridge = Ridge()
ridge.fit(X_Train,y_Train)
print ("Ridge Regression,Training Set RMSE:",rmse(y_Train,ridge.predict(X_Train)))
print ("Ridge Regression,Training R Squared:",calc_r2(ridge,y_Train,X_Train))
print ("\nRidge Regression,Testing Set RMSE:",rmse(y_Test,ridge.predict(X_Test)))
print ("Ridge Regression,Testing R Squared:",calc_r2(ridge,y_Test,X_Test))
print ("\nRidge Regression,Cross Validation Score:",cv_score(ridge))


# In[ ]:


scores = []
alpha = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10]
for i in alpha:
    ridge = Ridge(alpha=i)
    ridge.fit(X_Train,y_Train)
    scores.append(rmse(y_Test,ridge.predict(X_Test)))
print ("Ridge Scores with Different Alpha Values \n",scores)


# **Elastic Net**

# In[ ]:


en = ElasticNet(max_iter=5000)
params = {"alpha":[0.0001,0.0002,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10],
          "l1_ratio":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
grid = GridSearchCV(estimator=en,param_grid=params,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
grid.fit(X,y)


# In[ ]:


grid.best_estimator_,grid.best_params_,grid.best_score_


# In[ ]:


en = grid.best_estimator_
en.fit(X_Train,y_Train)
print ("Elastic Net Regression,Training Set RMSE:",rmse(y_Train,en.predict(X_Train)))
print ("Elastic Net Regression,Training R Squared:",calc_r2(en,y_Train,X_Train))
print ("\nElastic Net Regression,Testing Set RMSE:",rmse(y_Test,en.predict(X_Test)))
print ("Elastic Net Regression,Testing R Squared:",calc_r2(en,y_Test,X_Test))
print ("\nElastic Net Regression,Cross Validation Score:",cv_score(en))


# In[ ]:


en_submission = np.expm1(en.predict(test))
lasso_submission = np.expm1(lasso.predict(test))
ridge_submission = np.expm1(ridge.predict(test))
average = (en_submission+lasso_submission+ridge_submission)/3
Submission['SalePrice'] = average
Submission.to_csv("Stacked7.csv",index=None)

