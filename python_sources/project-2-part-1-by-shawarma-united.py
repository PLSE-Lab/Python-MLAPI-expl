#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques 
# ### Project 2, part 1, by Shawarma United
# #### >- Wardah Alharbi
# #### >- Yara Garwan
# #### >- Abdulmohsin Ababtain

# #### by reviewing many kernals posted for the competition we have concluded that efficiency is key we will be mapping the features that correlate with more than 50% in respect to the SalePrice, afterwards we will analyze the relationships and pick out the outliers

# In[ ]:


# import libraries
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# import datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print("train dataset shape:", train.shape)
print("test dataset shape:", test.shape)


# #### after inspecting the shapes we notice that 'test' is missing a column...

# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


train.loc[train['SalePrice'] > 500000]


# #### ...the column is SalePrice, in this project we will build a model to predict it

# In[ ]:


train['SalePrice'].describe()


# ## EDA
# 
# #### SalePrice inspection

# In[ ]:


train = train[train.SalePrice < 500000] # only 9 rows above 500k, so it became out cut-off
plt.figure(figsize=[11,7])
plt.xticks(np.arange(0, 500000, step=100000)) # better presentation for xticks
plt.yticks([]) # removes yticks
plt.ylabel('frequency')
sns.distplot(train.SalePrice, fit=norm, color= 'green', bins=35);
plt.rcParams['font.size'] = 13


# #### positively skewed, highest frequency between 100k & 250K
# ##### we will aim to filter the outliers to better our predictions

# ## Heatmap
# ### showcasig correlation between features

# #### filtered by correlated features above 10% in respect to SalePrice

# In[ ]:


corr = train.corr()
mid = corr.index[abs(corr["SalePrice"])>0.15]
plt.figure(figsize=(11,11))
ax = sns.heatmap(train[mid].corr(),annot=False, cmap="RdYlGn", vmin=0.10, vmax=1, cbar=True)
plt.title('Correlated features above 15% in respect to SalePrice')
plt.rcParams['font.size'] = 5;


# ### features with correlation below 15% in respect to SalePrice
# #### will be dropped in both datasets to eliminate outliers

# In[ ]:


#low = corr.index[abs(corr["SalePrice"])<0.15]
#low 


# In[ ]:


#train.drop(low, axis=1, inplace=True)
#test.drop(low, axis=1, inplace=True)


# #### filtered by correlated features above 51% in respect to SalePrice

# In[ ]:


corr = train.corr()
high = corr.index[abs(corr["SalePrice"])>0.51]
plt.figure(figsize=(11,11))
plt.rcParams['font.size'] = 11
ax = sns.heatmap(train[high].corr(),annot=True, cmap="RdYlGn", fmt= '.2g', annot_kws={"size": 13}, vmin=0.3, vmax=1, cbar=False)
plt.title('Correlated features above 50% in respect to SalePrice');


# #### top related features with SalePrice:
# 
# 
# >1.  OverallQual      0.80
# >2.  GrLivArea        0.71
# 
# ####  Also..
# 
# >3.  GarageCars       0.66
# >4.  GarageArea       0.65
# >5.  TotalBsmtSF      0.63
# >6.  1stFlrSF         0.60
# >7.  YearBuilt        0.57
# >8.  FullBath         0.55
# >9.  YearRemodAdd     0.54
# >10. TotRmsAbvGrd     0.51

# #### we inspect relative quantitative feature similar to what we did previously with SalePrice, to filter outliers

# # Cleaning the datasets

# ### selected quantitative features before removing outliers

# In[ ]:


train[['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF']].describe()


# # GrLivArea

# In[ ]:


train = train[train.GrLivArea < 3300]
plt.figure(figsize=[11,7])
plt.xticks(np.arange(0, 3500, step=500)) # better presentation for xticks
plt.yticks([]) # removes yticks
plt.ylabel('frequency')
sns.distplot(train.GrLivArea, fit=norm, color= 'green', bins=35)
plt.rcParams['font.size'] = 13;


# In[ ]:


# removing outliers
plt.figure(figsize=[11,7])
plt.scatter(train.GrLivArea, train.SalePrice, c='green', alpha=0.3)
plt.title("Looking for outliers"+" in GrLivArea")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.xticks(np.arange(200, 3500, step=500)) # better presentation for ticks
plt.yticks(np.arange(0, 500000, step=100000)) 
plt.show()


# # GarageArea

# In[ ]:


train = train[train.GarageArea < 1100]
plt.figure(figsize=[11,7])
plt.xticks(np.arange(0, 1200, step=100)) # better presentation for xticks
plt.yticks([]) # removes yticks
plt.ylabel('frequency')
sns.distplot(train.GarageArea, fit=norm, color= 'green', bins=50)
plt.rcParams['font.size'] = 13;


# In[ ]:


# removing outliers
plt.figure(figsize=[11,7])
plt.scatter(train.GarageArea, train.SalePrice, c='green', alpha=0.3)
plt.title("Looking for outliers"+" in GarageArea")
plt.xlabel("GarageArea")
plt.ylabel("SalePrice")
plt.xticks(np.arange(0, 1100, step=100)) # better presentation for ticks
plt.yticks(np.arange(0, 500000, step=100000)) 
plt.show()


# # TotalBsmtSF

# In[ ]:


train = train[train.TotalBsmtSF < 2200]
plt.figure(figsize=[11,7])
plt.xticks(np.arange(0, 2400, step=600)) # better presentation for xticks
plt.yticks([]) # removes yticks
plt.ylabel('frequency')
sns.distplot(train.TotalBsmtSF, fit=norm, color= 'green', bins=35)
plt.rcParams['font.size'] = 13;


# In[ ]:


# removing outliers
plt.figure(figsize=[11,7])
plt.scatter(train.TotalBsmtSF, train.SalePrice, c='green', alpha=0.3)
plt.title("Looking for outliers"+" in TotalBsmtSF")
plt.xlabel("TotalBsmtSF")
plt.ylabel("SalePrice")
plt.xticks(np.arange(0, 2400, step=500)) # better presentation for ticks
plt.yticks(np.arange(0, 500000, step=100000)) 
plt.show()


# # 1stFlrSF

# In[ ]:


train = train[train['1stFlrSF'] < 2200]
plt.figure(figsize=[11,7])
plt.xticks(np.arange(0, 2400, step=200)) # better presentation for xticks
plt.yticks([]) # removes yticks
plt.ylabel('frequency')
sns.distplot(train['1stFlrSF'], fit=norm, color= 'green', bins=35)
plt.rcParams['font.size'] = 13;


# In[ ]:


# removing outliers
plt.figure(figsize=[11,7])
plt.scatter(train['1stFlrSF'], train.SalePrice, c='green', alpha=0.3)
plt.title("Looking for outliers"+" in 1stFlrSF")
plt.xlabel("1stFlrSF")
plt.ylabel("SalePrice")
plt.xticks(np.arange(200, 2400, step=200)) # better presentation for ticks
plt.yticks(np.arange(0, 500000, step=100000)) 
plt.show()


# ### selected quantitative features after removing outliers

# In[ ]:


train[['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF']].describe()


# ## Merge and correct the nullified

# In[ ]:


# replacing null categories with description
desc={ 'PoolQC':'No_Pool', 'MiscFeature':'No_Misc', 'Alley':'No_alley_access', 'Fence':'No_Fence', 'FireplaceQu':'No_Fireplace' }
train.fillna(desc,inplace=True)
test.fillna(desc,inplace=True)


# In[ ]:


n_train = train.shape[0]
n_test = test.shape[0]
y_train = train.SalePrice.values
#df = pd.concat([train.drop(['SalePrice'], axis=1), test])


# In[ ]:


# LotFrontage (median by neighborhood)
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


# GarageX qualitative
for columns in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[columns] = train[columns].fillna('None')
    test[columns] = test[columns].fillna('None')


# In[ ]:


# GarageX quantitative
for columns in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[columns] = train[columns].fillna(0)
    test[columns] = test[columns].fillna(0)


# In[ ]:


# BsmtX qualitative
for columns in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[columns] = train[columns].fillna('None')
    test[columns] = test[columns].fillna('None')


# In[ ]:


# BsmtX quantitative
for columns in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train[columns] = train[columns].fillna(0)
    test[columns] = test[columns].fillna(0)


# In[ ]:


# Veneer qual & quan
train['MasVnrType'] = train['MasVnrType'].fillna('None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
test['MasVnrType'] = test['MasVnrType'].fillna('None')
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)


# In[ ]:


# Zoning (mode, most common)
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])


# In[ ]:


#Functional: Home functionality (Assume typical unless deductions are warranted)
train['Functional'] = train['Functional'].fillna('Typ')
test['Functional'] = test['Functional'].fillna('Typ')


# In[ ]:


#KitchenQual: Kitchen quality (Assume Typical/Average)
train['KitchenQual'] = train['KitchenQual'].fillna('TA')
test['KitchenQual'] = test['KitchenQual'].fillna('TA')


# In[ ]:


# MSSubClass
train['MSSubClass'] = train['MSSubClass'].fillna('None')
test['MSSubClass'] = test['MSSubClass'].fillna('None')


# In[ ]:


# Msc (mode, most common)
mode_columns = ['Electrical', 'Exterior1st', 'Exterior2nd', 'SaleType']
for columns in mode_columns:
    train[columns] = train[columns].fillna(train[columns].mode()[0])
    test[columns] = test[columns].fillna(test[columns].mode()[0])


# In[ ]:


# Utilities (Assume NoSewr: Electricity, Gas, and Water)
train['Utilities'] = train['Utilities'].fillna('NoSewr')
test['Utilities'] = test['Utilities'].fillna('NoSewr')


# In[ ]:


nullified_train =train.isnull().sum().sort_values(ascending=False)
nullified_train.head(3)


# In[ ]:


nullified_test =test.isnull().sum().sort_values(ascending=False)
nullified_test.head(3)


# In[ ]:


train.shape, test.shape


# ## Transform Features

# In[ ]:


transform=['GrLivArea', 'TotalBsmtSF' ,'GarageArea', '1stFlrSF']

train[transform]= np.log(train[transform]+1)
test[transform]= np.log(test[transform]+1)


# #### categorical features values into dummies

# In[ ]:


X_train = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1),
                                          test.drop(["Id"], axis=1)), axis=0)).iloc[: train.shape[0]]

X_test = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1),
                                         test.drop(["Id"], axis=1)), axis=0)).iloc[train.shape[0]:]

X_train.shape, X_test.shape


# ### converting dataset values to arrays

# In[ ]:


y=np.asarray(train['SalePrice'])
y=np.log(y+1)  #Apply log transform on target  
print(y[:5])


# In[ ]:


x_train=np.asarray(X_train)
x_test=np.asarray(X_test)
x_train.shape,y.shape


# ### creating train_test_split variables

# In[ ]:


xtrain, xval, ytrain, yval= train_test_split(x_train, y, test_size=0.2, random_state=4)
xtrain.shape, xval.shape


# ### Ridge Regression

# In[ ]:


parameter=[{'alpha':[1,2,3,4,5,6,7,8,9,10]}]

RR= Ridge()

grid_RR=GridSearchCV(RR, parameter, cv=8)

grid_RR.fit(xtrain, ytrain)

print("Score of Ridge: ",np.round(grid_RR.best_score_,4) )


# ### Lasso

# In[ ]:


parameter=[{'alpha':[0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]}]

LSR = Lasso()

grid_LSR=GridSearchCV(LSR, parameter, cv=8)

grid_LSR.fit(xtrain, ytrain)

print("Score of Lasso: ",np.round(grid_LSR.best_score_,4) )


# ### Elastic Net

# In[ ]:


parameter=[{"alpha": [0.0001, 0.001, 0.01, 0.1],
            "l1_ratio": np.arange(0.0, 1.0, 0.1)}]

EN = ElasticNet(max_iter=3000,tol=0.1)

grid_EN=GridSearchCV(EN, parameter, cv=8)

grid_EN.fit(xtrain, ytrain)

print("Score of Elastic Net: ",np.round(grid_EN.best_score_,4) )


# In[ ]:


idd = test.Id
predict = grid_LSR.predict(X_test)


# In[ ]:


submission= pd.DataFrame({'Id':idd, 'SalePrice':predict})
submission.to_csv("submit_to_kernel.csv", index=False)


# In[ ]:





# In[ ]:


# convert dtypes to numerical values
#df['MSSubClass'] = df['MSSubClass'].apply(str)
#df['OverallCond'] = df['OverallCond'].astype(str)
#df['YrSold'] = df['YrSold'].astype(str)
#df['MoSold'] = df['MoSold'].astype(str)


# In[ ]:


# prepare for modeling; convert columns values to numbers using the Label Encoder:
#columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
#        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
#        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
#        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
#        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
#for c in columns:
#    lbl = LabelEncoder() 
#    lbl.fit(list(df[c].values)) 
#    df[c] = lbl.transform(list(df[c].values))


# In[ ]:


# convert categorical variables into dummies
#df = pd.get_dummies(df)
#df.shape


# In[ ]:


#Xtrain = df[:train]
#Xtest = df[train:]
#Xtrain.shape, Xtest.shape, y_train.shape


# In[ ]:


#n_train = train.shape[0]
#n_test = test.shape[0]
#y_train = train.SalePrice.values


# In[ ]:


#xtrain, xval, ytrain, yval= train_test_split(n_train, y_train, test_size=0.2, random_state=4)
#xtrain.shape, xval.shape


# In[ ]:




