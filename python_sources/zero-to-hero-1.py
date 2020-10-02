#!/usr/bin/env python
# coding: utf-8

# # **Zero To Hero**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Hey Folks!
# ## Welcome to my kernel

# This kernel is develpoed for beginners which will help to clear many doubts.
# This kernel will definately provide an idea about  "How to deal with data".
# > Before we move further let's understand the flow of the noteebook

# ## The flow of the notebook goes as
# 1.  **Importing the data ** : we will import the data and will try to understand it
# 2. **Treatment of sick data** : It includes processing of the data
# 3. **Visualization techniques** : We will try to analyze our data and convert our magic numbers into beautiful & meaningfull graphs
# 4. **EDA** : We will convert data into efficient trainable form
# 4. **Feature Addition** : Adding new features to our dat set
# 5. **Training Our Model**
# 6. **Submitting our predictiom**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train_data


# In[ ]:


test_data


# # ** ** Let's First try to understand the data

# In[ ]:


train_data.describe()


# In[ ]:


x=train_data.columns


# ## Checking the NaN values

# In[ ]:


#getting the colimns with high NaN values
nan=[]
for i in x:
    n=train_data[i].isnull().sum()
    if n>0:
        nan.append(i)
nan


# Hmmmm These are the troublesome 
# So we'll  treat these columns later

# ## Let's check the columns which have less unique values

# In[ ]:


less=[]
high=[]
for i in x:
    l=train_data[i].nunique()
    if l<50:
        less.append(i)
    else :
        high.append(i)
        


# We will convert these columns with dummies

# **Let's categorize the columns on basis of data types**

# In[ ]:


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in train_data.columns:
    if train_data[i].dtype in numeric_dtypes:
        numeric.append(i)
_=numeric.pop(0)
_=numeric.pop(-1)
# poped out Id,SalePrice  you will understand the reason later
numeric


# In[ ]:


category = []
for i in x:
    if train_data[i].dtypes == 'object':
        category.append(i)
    
category    


# ## now we will understand data with the help of visualization techniques

# In[ ]:


plt.figure(1,figsize=(15,7))
sns.distplot(train_data['SalePrice'])
plt.xlabel('Sales price')
plt.ylabel('Frequency')
plt.title('Sales Price Distribution')
plt.show()


# In[ ]:


# Skew and kurt
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())


# we need to correct the skewness of the price

# In[ ]:


# log(1+x) transform
corrected_price= np.log1p(train_data["SalePrice"])
#we will be substituting the value at last


# In[ ]:


plt.figure(1,figsize=(15,7))
sns.distplot(corrected_price)
plt.xlabel('Sales price')
plt.ylabel('Frequency')
plt.title('Sales Price Distribution')
plt.show()


# In[ ]:


zones=train_data[['MSZoning','SalePrice']]
sns.catplot(x='MSZoning',y='SalePrice',kind='swarm',data=zones)


# In[ ]:


z=zones.groupby(['MSZoning']).mean()
z.plot(kind='bar')
_=plt.xticks(rotation=0)


# In[ ]:


neighbor=train_data[['LandContour','Neighborhood','SalePrice']]
plt.figure(1,figsize=(25,10))
sns.swarmplot(x=neighbor['Neighborhood'],y=neighbor['SalePrice'])
_=plt.xticks(rotation=90)


# In[ ]:


plt.figure(1,figsize=(25,10))
sns.scatterplot(x=neighbor['Neighborhood'],y=neighbor['SalePrice'],hue=neighbor['LandContour'])
_=plt.xticks(rotation=90)


# In[ ]:


bldg = train_data[['BldgType','HouseStyle','OverallQual','OverallCond','SalePrice','MiscVal']]


# In[ ]:


bldg


# In[ ]:


plt.figure(1,figsize=(15,7))
sns.jointplot(x='OverallQual',y='SalePrice',data=bldg,kind='kde')
plt.figure(2,figsize=(15,7))
sns.jointplot(x='OverallCond',y='SalePrice',data=bldg,kind='kde')
plt.show()


# In[ ]:


bldg=bldg.groupby(['BldgType']).sum()


# In[ ]:


bldg


# In[ ]:


#bldg['SalePrice'].plot(kind='bar')
sns.regplot(x=bldg.OverallQual,y=bldg.SalePrice,units=bldg.OverallCond)


# In[ ]:


bldg.plot(kind='hist')


# In[ ]:


garage=train_data[['GarageType','GarageFinish','GarageCars','GarageQual','GarageCond','SalePrice']]
garage.head()


# In[ ]:


g=garage.groupby(['GarageType'])
ga=g.plot(x='GarageFinish',y='SalePrice',alpha=.6)


# In[ ]:


sns.pointplot(x=garage['SalePrice'],y=garage['GarageType'],hue=garage['GarageFinish'],join=False)


# In[ ]:


sns.pointplot(x=garage['GarageType'],y=garage['GarageCars'],hue=garage['GarageFinish'],color='blue',join=False)
plt.show()


# In[ ]:


g=g=garage.groupby(['GarageType']).mean()
g['GarageCars']=g['GarageCars'].astype(int)
g['SalePrice']=g['SalePrice'].astype(int)


# In[ ]:


sns.regplot(x=g['SalePrice'],y=g['GarageCars'],data=g,scatter=True,fit_reg=True)


# In[ ]:


sns.regplot(x=g['GarageCars'],y=g['SalePrice'],data=g)


# In[ ]:


misc=train_data[['MiscVal','SaleType','SalePrice']]
misc.groupby(['SaleType']).mean()


# In[ ]:


sns.lmplot(x='MiscVal',y='SalePrice',hue='SaleType',data=misc)


# In[ ]:


sns.catplot(x='MiscVal',y='SalePrice',hue='SaleType',data=misc)
plt.figure(1,figsize=(55,15))
_=plt.xticks(rotation=90)


# In[ ]:


date=train_data[['MoSold','YrSold','SalePrice','SaleCondition']]


# In[ ]:


sns.boxplot(x='YrSold',y='SalePrice',data=date)


# In[ ]:


sns.lmplot(x='MoSold',y='SalePrice',hue='YrSold',data=date)


# In[ ]:


sns.pointplot(x='MoSold',y='SalePrice',hue='YrSold',data=date,join=False)


# In[ ]:


corr = train_data.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Greens", square=True)


# Although these graphs might appear as images for you by now, but with practise these graph would be speaking out loud for you

# For missing values

# In[ ]:


missing = round(train_data.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot(kind='bar')
plt.figure(figsize=(15,7))


# # **EDA**

# Combining train and test data and separating the label

# In[ ]:


#separating the label
train = train_data.copy()
test=test_data.copy()
label=train_data['SalePrice'].reset_index(drop=True)
train=train.drop(['SalePrice'],axis=1)


# In[ ]:


#combining  the trainand test data
data=pd.concat([train,test]).reset_index(drop=True)


# In[ ]:


#removing the Id column
train_id=train['Id']
test_id=test['Id']
data=data.drop(['Id'],axis=1)
data.reset_index(drop=True)


# ## Filling the missing values

# In[ ]:


num_nan=[]
for i in numeric:
    for j in nan:
        if i==j:
            num_nan.append(j)
            print(j)
    


# so these are the columns which have missing value and have numeric data
# > if you don't know how to remove missing values take a look at this : https://www.kaggle.com/alexisbcook/missing-values

# In[ ]:


# imputing the numeric missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()


# In[ ]:


imp_data=pd.DataFrame(imputer.fit_transform(data[num_nan]))
imp_data.columns=num_nan
data[num_nan]=imp_data


# now we will add the categorial missing values manually

# In[ ]:


# these are the columns whose missing values are to be added
cat_nan=[set(nan)-set(num_nan)]
cat_nan


# Let's get started
# We wil fill them with our intution by analyzing the data thoroughly

# In[ ]:


data['Alley'] = data['Alley'].fillna(data["Alley"].mode()[0])
for i in ('BsmtCond','BsmtFinType1','BsmtFinType2','BsmtQual','BsmtExposure'):
    data[i] = data[i].fillna('None')
data['Electrical'] = data['Electrical'].fillna('SBrkr')
data['MasVnrType']=data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])
data['Fence'] = data['Fence'].fillna(data['Fence'].mode()[0])
for i in ('GarageFinish','GarageCond','GarageQual','GarageType'):
    data[i]=data[i].fillna('None')
data['FireplaceQu'] = data['FireplaceQu'].fillna('Fire')    
data['MiscFeature']=data['MiscFeature'].fillna('Misc')
data['PoolQC']=data['PoolQC'].fillna('Pool')


# In[ ]:


# doing the final check if any mising value left by mistake
data.update(data[numeric].fillna(0))
data.update(data[category].fillna('None'))


# # Adding New Features

# Our Features Are Based Over Our Intution.
# 
# 
# We will  try to create those features which would be directly related to the Sale Price and special features in it
# 
# And will remove columns which are not useful enough

# Before  doing it dig deeply into the data description :https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# In[ ]:


# adding columns of average/good features
data['hasallutilities']=1*(data['Utilities'] == 'AllPub')
data['average']=1*(data['OverallQual'] == 5 | 6)
data['ext_qual']=1*(data['ExterQual'] == 'Gd')
data['fencing']=1*(data['Fence'] == 'GdPrv')

#adding column on basis of features
data['total_qual'] = data['OverallQual'] + data['OverallCond']
data['total_floors_area'] =  data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['total_sqr'] =  data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['total_poarch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +data['EnclosedPorch'] + data['ScreenPorch'] +data['WoodDeckSF'])

#adding columns for special features
data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

# dropping some columns
data=data.drop(['PavedDrive','Street','PavedDrive'],axis=1)


# In[ ]:


data.head(3)


# # Encoding Our Data

# In[ ]:


data=pd.get_dummies(data).reset_index(drop=True)
data.head(3)


# Check out other encoding techniques :https://www.kaggle.com/alexisbcook/categorical-variables

# ## Spliting back the train and test data

# In[ ]:


train = data.iloc[:len(label),:]
test = data.iloc[len(label):,:]
train.shape,test.shape


# ## **Training the model**

# ## 1. Creating diffrent Models
# ## 2. Cross Validation of data
# ## 3. Selecting the best model

# In[ ]:


# for models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

#for cross validation
from sklearn.model_selection import KFold, cross_val_score

#for calculation of error
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline


# Creating models

# In[ ]:


# RAndom Forest Regressor
model1 = RandomForestRegressor(n_estimators=1000,max_depth=12,min_samples_split=5,min_samples_leaf=5,oob_score=True,random_state=50)

#XGBRegressor
model2 = XGBRegressor(n_estimators=5000,learning_rate=0.01,early_stopping_round=10,max_depth=4,min_child_weight=0,gamma=0.6,verbose=False,random_state=50)

#Light LGB Regressor
model3=  LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=5000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       verbose=-1,
                       random_state=50)


# Cross validation of these models

# In[ ]:


def cv(model,x=train):
    er = -1*cross_val_score(model,x,label,cv=10,scoring='neg_mean_absolute_error')
    # root mean square 
    rmse=np.sqrt(er)
    return rmse


# have a look if you don't get the above code :https://www.kaggle.com/alexisbcook/cross-validation

# Calculating score of each model

# In[ ]:


#for RandomForest
score1=cv(model1)
Rf=[score1.mean(),score1.std()]
Rf


# In[ ]:


score2=cv(model2)
Xg_mean=score2.mean()
Xg_std=score2.std()
Xg=[Xg_mean,Xg_std]
Xg


# In[ ]:


score3=cv(model3)
Lg_mean=score3.mean()
Lg_std=score3.std()
Lg=[Lg_mean,Lg_std]
Lg


# In[ ]:


# what do you understand with this data


# In[ ]:


model2.fit(train,label)


# In[ ]:


#predicting our data on test
pred= model2.predict(test)


# In[ ]:


#this is our predicted data
pred


# In[ ]:


plt.plot(pred,test.index,color='blue')
plt.title("This is the Predicted Price")


# In[ ]:


plt.plot(train_data['SalePrice'],train_data.index,color='green')
plt.title("This was the trained Price")


# In[ ]:


sample=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


sample


# In[ ]:


Id= np.array(range(1461,2920))


# In[ ]:


Id


# In[ ]:


out1 =pd.DataFrame({"Id":Id,"SalePrice": pred})
out1


# In[ ]:


out1.to_csv('my_submission.csv', index=False)


# our data is submitted successfully

# # **Note**

# ## We are data scientist (thiugh not now) and we deal with real facts . And one real fact is that no one can become an expert with one notebook**
# **In this notebook  I have tried to provide max. information one can gain from a notebbok additional  techniques would have  create confusion , the msin aim for this draft is to get an idea about hoew thing work, so thie output submiited is not much efficient and precise .I would keep adding new version with more advanced stuffs, which will generate a more efficient and precise output.**
# 
# ## Till then get through with it and accept it as the first draft
# 
# > 
# Tip : Never Submit your first draft in a competition , figure out more ways to improve results

# Please upvote if you found it informative in any way

# # Thank You For Giving Time To This Notebook

# In[ ]:




