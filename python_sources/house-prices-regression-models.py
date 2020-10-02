#!/usr/bin/env python
# coding: utf-8

# Feature engineering is an important part of machine learning process so I want to spend more time for this part. I'm gonna try I few models and tell you which work the best with train dataset from this competition. 

# **Import the Libraries**

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import math 
np.random.seed(2019)
from scipy.stats import skew
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

import statsmodels

#!pip install ml_metrics
from ml_metrics import rmsle

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("done")


# **Import Data**

# I'm adding here 'train' variable in order to check in the easiest way which observations are from train and test dataset because I'm gonna join train and test datasets.

# In[ ]:


def read_and_concat_dataset(training_path, test_path):
    train = pd.read_csv(training_path)
    train['train'] = 1
    test = pd.read_csv(test_path)
    test['train'] = 0
    data = train.append(test, ignore_index=True)
    return train, test, data

train, test, data = read_and_concat_dataset('../input/train.csv', '../input/test.csv')
data = data.set_index('Id')


# In[ ]:


data.columns[data.isnull().sum()>0]


# There are a few variables with NaN value but in these cases 'NaN' means something else than missing value. For example 'NaN' in 'GarageCond' means that this house hasn't a garage. I'm gonna change 'NaN' values to 'None' string. 

# ##**Fixing variables**

# In[ ]:


def filling_missing_values(data,variable, new_value):
    data[variable] = data[variable].fillna(new_value)


# In[ ]:


filling_missing_values(data,'GarageCond','None')
filling_missing_values(data,'GarageQual','None')
filling_missing_values(data,'FireplaceQu','None')
filling_missing_values(data,'BsmtCond','None')
filling_missing_values(data,'BsmtQual','None')
filling_missing_values(data,'PoolQC','None')
filling_missing_values(data,'MiscFeature','None')


# MSSubClass is not a numerical variables, so let's transform it to caterogical variable.

# In[ ]:


data['MSSubClass'][data['MSSubClass'] == 20] = '1-STORY 1946 & NEWER ALL STYLES'
data['MSSubClass'][data['MSSubClass'] == 30] = '1-STORY 1945 & OLDER'
data['MSSubClass'][data['MSSubClass'] == 40] = '1-STORY W/FINISHED ATTIC ALL AGES'
data['MSSubClass'][data['MSSubClass'] == 45] = '1-1/2 STORY - UNFINISHED ALL AGES'
data['MSSubClass'][data['MSSubClass'] == 50] = '1-1/2 STORY FINISHED ALL AGES'
data['MSSubClass'][data['MSSubClass'] == 60] = '2-STORY 1946 & NEWER'
data['MSSubClass'][data['MSSubClass'] == 70] = '2-STORY 1945 & OLDER'
data['MSSubClass'][data['MSSubClass'] == 75] = '2-1/2 STORY ALL AGES'
data['MSSubClass'][data['MSSubClass'] == 80] = 'SPLIT OR MULTI-LEVEL'
data['MSSubClass'][data['MSSubClass'] == 85] = 'SPLIT FOYER'
data['MSSubClass'][data['MSSubClass'] == 90] = 'DUPLEX - ALL STYLES AND AGES'
data['MSSubClass'][data['MSSubClass'] == 120] = '1-STORY PUD (Planned Unit Development) - 1946 & NEWER'
data['MSSubClass'][data['MSSubClass'] == 150] = '1-1/2 STORY PUD - ALL AGES'
data['MSSubClass'][data['MSSubClass'] == 160] = '2-STORY PUD - 1946 & NEWER'
data['MSSubClass'][data['MSSubClass'] == 180] = 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER'
data['MSSubClass'][data['MSSubClass'] == 190] = '2 FAMILY CONVERSION - ALL STYLES AND AGES'


# A few categorical variables are ordinal variables, so let's fix them. 

# In[ ]:


def fixing_ordinal_variables(data, variable):
    data[variable][data[variable] == 'Ex'] = 5
    data[variable][data[variable] == 'Gd'] = 4
    data[variable][data[variable] == 'TA'] = 3
    data[variable][data[variable] == 'Fa'] = 2
    data[variable][data[variable] == 'Po'] = 1
    data[variable][data[variable] == 'None'] = 0


# In[ ]:


fixing_ordinal_variables(data,'ExterQual')
fixing_ordinal_variables(data,'ExterCond')
fixing_ordinal_variables(data,'BsmtCond')
fixing_ordinal_variables(data,'BsmtQual')
fixing_ordinal_variables(data,'HeatingQC')
fixing_ordinal_variables(data,'KitchenQual')
fixing_ordinal_variables(data,'FireplaceQu')
fixing_ordinal_variables(data,'GarageQual')
fixing_ordinal_variables(data,'GarageCond')
fixing_ordinal_variables(data,'PoolQC')


# ..and one more but in different way.

# In[ ]:


data['PavedDrive'][data['PavedDrive'] == 'Y'] = 3
data['PavedDrive'][data['PavedDrive'] == 'P'] = 2
data['PavedDrive'][data['PavedDrive'] == 'N'] = 1


# ##**Missing values**

# First of all I'm gonna look how many variables have less than 50 missing values and fix it. Then I'll look how about variables with more than 50 missing values.

# In[ ]:


colu = data.columns[(data.isnull().sum()<50) & (data.isnull().sum()>0)]
for i in colu:
    print(data[colu].isnull().sum())


# In[ ]:


colu = data.columns[data.isnull().sum()>=50]
for i in colu:
    print(data[colu].isnull().sum())


# I'm putting 0 in GarageArea, GarageFinish, GarageType, GarageYrBlt and GarageCars where houses don't have garage. 

# In[ ]:


filling_missing_values(data, 'GarageArea',0)
filling_missing_values(data, 'GarageCars',0)
data['GarageFinish'][(data.GarageFinish.isnull()==True) & (data.GarageCond==0)] =0
data['GarageType'][(data.GarageType.isnull()==True) & (data.GarageCond==0)] =0
data['GarageYrBlt'][(data.GarageYrBlt.isnull()==True) & (data.GarageCond==0)] =0


# I'm gonna put 0 in MiscVal for house which don't have any MiscFeature and 'None' value for house with 0 in MiscValue and some value in MiscFeature.

# In[ ]:


print(data[['MiscFeature','MiscVal']][(data.MiscFeature=='None') & (data.MiscVal>0)])
data.MiscVal.loc[2550] = 0

print(data[['MiscFeature','MiscVal']][(data.MiscVal==0) & (data.MiscFeature!='None')])
c=data[['MiscFeature','MiscVal']][(data.MiscVal==0) & (data.MiscFeature!='None')].index
data.MiscFeature.loc[c] = 'None'


# Now I'm gonna write two functions to help me in imputing missing values in variables. I'm using here Random Forest Regressor and Classifier. 

# In[ ]:


def inputing(variab):
    y = data[variab]
    data2 = data.drop([variab],axis=1)
    col = data2.columns[data2.isnull().sum()==0]
    data2 = data2[col]
    data2 = pd.get_dummies(data2)
    c_train = y[y.notnull()==True].index
    y_train = y[c_train]
    columny = data2.columns
    X_train = data2[columny].loc[c_train]
    c_test = y[y.notnull()!=True].index
    y_test = y[c_test]
    X_test = data2[columny].loc[c_test]
    #Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #Filling missing data
    y_pred = pd.Series(y_pred, index=c_test)
    data[variab].loc[c_test] = y_pred.loc[c_test]
    
def inputingnum(variab):
    y = data[variab]
    data2 = data.drop([variab],axis=1)
    col = data2.columns[data2.isnull().sum()==0]
    data2 = data2[col]
    data2 = pd.get_dummies(data2)
    c_train = y[y.notnull()==True].index
    y_train = y[c_train]
    columny = data2.columns
    X_train = data2[columny].loc[c_train]
    c_test = y[y.notnull()!=True].index
    y_test = y[c_test]
    X_test = data2[columny].loc[c_test]
    #Model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #Filling missing data
    y_pred = pd.Series(y_pred, index=c_test)
    data[variab].loc[c_test] = y_pred.loc[c_test]


# Let's imput missing values using two functions which I wrote. In KitchenQual, BsmtFullBath and BsmtHalfBath cases I'm gonna use Regressor model and convert them to integer.

# In[ ]:


inputing(variab='Electrical')
inputing(variab='Exterior2nd')
inputing(variab='Exterior1st')
inputing(variab='MasVnrType')
inputing(variab='Functional')
inputing(variab='MSZoning')
inputing(variab='SaleType')
inputing(variab='Alley')
inputing(variab='BsmtExposure')
inputing(variab='BsmtFinType1')
inputing(variab='BsmtFinType2')
inputing(variab='Fence')

inputingnum(variab='KitchenQual')
data['KitchenQual'] = data.KitchenQual.astype(int)
inputingnum(variab='BsmtFullBath')
data['BsmtFullBath'] = data.BsmtFullBath.astype(int)
inputingnum(variab='BsmtHalfBath')
data['BsmtHalfBath'] = data.BsmtHalfBath.astype(int)

inputingnum(variab='TotalBsmtSF')
inputingnum(variab='BsmtFinSF1')
inputingnum(variab='BsmtFinSF2')
inputingnum(variab='MasVnrArea')
inputingnum(variab='BsmtUnfSF')
inputingnum(variab='LotFrontage')


# In[ ]:


print(data['Utilities'].value_counts())
data  = data.drop(['Utilities'],axis=1)


# In[ ]:


data.columns[data.isnull().sum()>0]


# It's everything about imputing missing values.

# Let's understand a data set variable after variable, check basic statistics and drop a few outliers. I'll also drop variables with little differentiation. 

# In[ ]:


data.describe()


# In[ ]:


from scipy.stats import norm
plt.figure(figsize=(15,8))
sns.distplot(data['SalePrice'][data.SalePrice.isnull()==False], fit= norm,kde=True)
plt.show()


# ##**Dropping outliers**

# On the scatter charts, I checked which observations could be considered outliers and I decided to delete them.
# I must be very careful because I don't want to remove observations from the test set.

# For example, let's look at scatter plot of SalePrice and Lot Frontage.

# In[ ]:


print(data.plot.scatter(x='LotFrontage',y='SalePrice'))


# In[ ]:


def dropping_outliers(data, condition):
    #put condition with with reference to the data table, use brackets and (& |) operators, remember about you can drop observation only from train dataset
    condition_to_drop = data[condition].index
    data = data.drop(condition_to_drop)


# In[ ]:


dropping_outliers(data, (data.SalePrice<100000) & (data.train==1) & (data.LotFrontage>150))
dropping_outliers(data, (data.LotFrontage>200) & (data.train==1))
dropping_outliers(data, (data.SalePrice>700000) & (data.train==1))
dropping_outliers(data, (data.SalePrice>700000) & (data.train==1))
dropping_outliers(data, (data.LotArea>60000) & (data.train==1))
dropping_outliers(data, (data.MasVnrArea>1450) & (data.train==1))
dropping_outliers(data, (data.BedroomAbvGr==8) & (data.train==1))
dropping_outliers(data, (data.KitchenAbvGr==3) & (data.train==1))
dropping_outliers(data, (data['3SsnPorch']>400) & (data.train==1))
dropping_outliers(data, (data.LotArea>100000) & (data.train==1))
dropping_outliers(data, (data.MasVnrArea>1300) & (data.train==1))
dropping_outliers(data, (data.BsmtFinSF1>2000) & (data.train==1) & (data.SalePrice<300000))
dropping_outliers(data, (data.BsmtFinSF2>200) & (data.SalePrice>350000)  & (data.train==1))
dropping_outliers(data, (data.BedroomAbvGr==8) & (data.train==1))
dropping_outliers(data, (data.KitchenAbvGr==3) & (data.train==1))
dropping_outliers(data, (data.TotRmsAbvGrd==2) & (data.train==1))


# In[ ]:


# c=data[(data['SalePrice']<100000) & (data.train==1) & (data['LotFrontage']>150)].index
# data = data.drop(c)
# c=data[(data['LotFrontage']>200) & (data.train==1)].index
# data = data.drop(c)
# c=data[(data['SalePrice']>700000) & (data.train==1)].index
# data = data.drop(c)
# c = data[(data['SalePrice']>700000) & (data.train==1)].index
# data = data.drop(c)
# c = data[(data['LotArea']>60000) & (data.train==1)].index
# data = data.drop(c)
# c = data[(data['MasVnrArea']>1450) & (data.train==1)].index
# data = data.drop(c)
# c = data[(data['BedroomAbvGr']==8) & (data.train==1)].index
# data = data.drop(c)
# c = data[(data['KitchenAbvGr']==3) & (data.train==1)].index
# data = data.drop(c)
# c = data[(data['3SsnPorch']>400) & (data.train==1)].index
# data = data.drop(c)
# c=data[(data.LotArea>100000) & (data.train==1)].index
# data = data.drop(c)
# c=data[(data.MasVnrArea>1300) & (data.train==1)].index
# data = data.drop(c)
# c=data[(data.BsmtFinSF1>2000) & (data.train==1) & (data.SalePrice<300000)].index
# data = data.drop(c)
# c=data[(data.BsmtFinSF2>200) & (data.SalePrice>350000)  & (data.train==1)].index
# data = data.drop(c)
# c=data[(data.BedroomAbvGr==8) & (data.train==1)].index
# data = data.drop(c)
# c=data[(data.KitchenAbvGr==3) & (data.train==1)].index
# data = data.drop(c)
# c=data[(data.TotRmsAbvGrd==2) & (data.train==1)].index
# data = data.drop(c)


# CentalAir variable needs transformation to binary variable.

# In[ ]:


#CentralAir
print(data['CentralAir'].value_counts())
data['CentralAir'] = pd.Series(np.where(data['CentralAir'].values == 'Y', 1, 0),
          data.index)


# ##**Feature engineering**

# * 2ndFloor - if the house has a second floor
# * Floors - total area of the first and second floor
# * TotBath - how many bathrooms house has
# * Porch - total area of the porch
# * TotalSF - total area of the house
# * Pool - if the house has a swimming pool
# * Bsmt - if the house has a basement
# * Garage - if the house has a garage
# * Fireplace - if the house has a fireplace
# * Remod - if the house was renovated
# * NewHouse - if the house is new
# * Age - ages of house
# 

# In[ ]:


data['2ndFloor'] = pd.Series(np.where(data['2ndFlrSF'].values == 0, 0, 1),data.index)
data['Floors'] = data['1stFlrSF'] + data['2ndFlrSF']
data = data.drop(['1stFlrSF'],axis=1)
data = data.drop(['2ndFlrSF'],axis=1)
data['TotBath'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])
data['Porch'] = data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch']
data['TotalSF'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['Floors'] 
data['Pool'] = pd.Series(np.where(data['PoolArea'].values == 0, 0, 1),data.index)
data['Bsmt'] = pd.Series(np.where(data['TotalBsmtSF'].values == 0, 0, 1),data.index)
data['Garage'] = pd.Series(np.where(data['GarageArea'].values == 0, 0, 1),data.index)
data['Fireplace'] = pd.Series(np.where(data['Fireplaces'].values == 0, 0, 1),data.index)
data['Remod'] = pd.Series(np.where(data['YearBuilt'].values == data['YearRemodAdd'].values, 0, 1),data.index)
data['NewHouse'] = pd.Series(np.where(data['YearBuilt'].values == data['YrSold'].values, 1, 0),data.index)
data['Age'] = data['YrSold'] - data['YearRemodAdd']


# I'm gonna drop more observations.

# In[ ]:


c = data[(data['Floors']>4000) & (data.train==1)].index
data = data.drop(c)
c = data[(data['SalePrice']>500000) & (data['TotalSF']<3500) & (data.train==1)].index
data = data.drop(c)


# **Droping a few variables**

# In[ ]:


data = data.drop(['PoolQC'],axis=1)
data = data.drop(['GrLivArea'],axis=1)
data = data.drop(['Street'],axis=1)
data = data.drop(['GarageYrBlt'],axis=1)
data = data.drop(['PoolArea'],axis=1)
data = data.drop(['MiscFeature'],axis=1)


# **Preparing to modeling:**
# - dummies variables
# - two data frames with independent variables for train and test set
# - vector y with Sale Price variable for train set

# ##**Modeling:**
# 
# - XGB Regressor
# - Decision Tree Regressor
# - Random Forest Regressor
# - LASSO Regression
# 
# 
# For each model I tuned the parameters using loops and each model contains SalePrice variable tranformed to logarithm.

# In[ ]:


Results = pd.DataFrame({'Model': [],'Accuracy Score': []})


# In[ ]:


data = pd.get_dummies(data)


# **XGBoost Regressor**

# In[ ]:


from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)
trainY = np.log(trainY)

model = XGBRegressor(learning_rate=0.001,n_estimators=4600,
                                max_depth=7, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=27,
                                reg_alpha=0.00006)
model.fit(trainX,trainY)
y_pred = model.predict(testX)
y_pred = np.exp(y_pred)

res = pd.DataFrame({"Model":['XGBoost'],
                    "Accuracy Score": [rmsle(testY, y_pred)]})
Results = Results.append(res)


# **Decision Tree Regressor**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)
trainY = np.log(trainY)

model = DecisionTreeRegressor(max_depth=6)
model.fit(trainX,trainY)
y_pred = model.predict(testX)
y_pred = np.exp(y_pred)

print(rmsle(testY, y_pred))

res = pd.DataFrame({"Model":['Decision Tree'],
                    "Accuracy Score": [rmsle(testY, y_pred)]})
Results = Results.append(res)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)
trainY = np.log(trainY)

model = RandomForestRegressor(n_estimators=1500,
                                max_depth=6)
model.fit(trainX,trainY)
y_pred = model.predict(testX)
y_pred = np.exp(y_pred)
print(rmsle(testY, y_pred))

res = pd.DataFrame({"Model":['Random Forest'],
                    "Accuracy Score": [rmsle(testY, y_pred)]})
Results = Results.append(res)


# **LASSO Regression**

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)
trainY = np.log(trainY)

model = Lasso(alpha=0.0005)

model.fit(trainX,trainY)
y_pred = model.predict(testX)
y_pred = np.exp(y_pred)
print(rmsle(testY, y_pred))

res = pd.DataFrame({"Model":['LASSO'],
                    "Accuracy Score": [rmsle(testY, y_pred)]})
Results = Results.append(res)


# **Stepwise Regression**

# In[ ]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)
trainY = np.log(trainY)

X2 = sm.add_constant(trainX)
o=0
for i in X2.columns:
    o+=1
    print(o)
    model = sm.OLS(trainY, X2.astype(float))
    model = model.fit()
    p_values = pd.DataFrame(model.pvalues)
    p_values = p_values.sort_values(by=0, ascending=False)
    if float(p_values.loc[p_values.index[0]])>=0.05:
        X2=X2.drop(p_values.index[0],axis=1)
    else:
        break

kolumny = X2.columns
testX = sm.add_constant(testX)
testX = testX[kolumny]

y_pred = model.predict(testX)
y_pred = np.exp(y_pred)


res = pd.DataFrame({"Model":['Stepwise Regression'],
                    "Accuracy Score": [rmsle(testY, y_pred)]})
Results = Results.append(res)


# **Ridge Regression**

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)
trainY = np.log(trainY)

model = Ridge(alpha=0.0005)

model.fit(trainX,trainY)
y_pred = model.predict(testX)
y_pred = np.exp(y_pred)
print(rmsle(testY, y_pred))

res = pd.DataFrame({"Model":['Ridge'],
                    "Accuracy Score": [rmsle(testY, y_pred)]})
Results = Results.append(res)


# **Linear Regression**
# 
# When you change alpha to 0 value in LASSO, you have simple Linear Regression model.

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)
trainY = np.log(trainY)

model = Lasso(alpha=0)

model.fit(trainX,trainY)
y_pred = model.predict(testX)
y_pred = np.exp(y_pred)
print(rmsle(testY, y_pred))

res = pd.DataFrame({"Model":['Linear Regression'],
                    "Accuracy Score": [rmsle(testY, y_pred)]})
Results = Results.append(res)


# ##**Results**

# In[ ]:


Results


# LASSO Regression model gives the best results. This model helps me to get 0.12903 (RMSLE) on competition test dataset and it gives me place in 17% best results on Leaderboard.

# In[ ]:


trainX = data[data.SalePrice.isnull()==False].drop(['SalePrice','train'],axis=1)
trainY = data.SalePrice[data.SalePrice.isnull()==False]
testX = data[data.SalePrice.isnull()==True].drop(['SalePrice','train'],axis=1)
trainY = np.log(trainY)
model = Lasso(alpha=0.0005)
model.fit(trainX, trainY)
test = data[data.train==0]
test['SalePrice'] = model.predict(testX)
test['SalePrice'] = np.exp(test['SalePrice'] )
test = test.reset_index()
test[['Id','SalePrice']].to_csv("submissionLASSO.csv",index=False)
print("done1")

