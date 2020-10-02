#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as st 
import os
from tqdm import tqdm


# In[ ]:


df1=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
df2=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')


# # Data Cleaning
# 
# let us combine test and train to clean the data and make it machine usable and plit the data later on 

# In[ ]:


df=pd.concat([df1,df2],sort=False)


# In[ ]:


df.describe(include='object')


# In[ ]:


#df.info()


# Some of the missing values are supposed to be marked as 'N.A' if categorical and 0 if numerical. The rest can be imputed by mode or median. We must figure this out by checking the data description.
# 
# For example if a value is missing in garage condition it means there is no garage and the corresponding garage area will be 0. Hence it is important to notice these variables and impute the right values 

# In[ ]:


na=df.isnull().sum()


# In[ ]:


na=na[na>0]


# In[ ]:


na_cols=list(na.index)


# In[ ]:


na=df[na_cols].copy()


# In[ ]:


#na.info()


# In[ ]:


dummy_cols=['PoolQC','MiscFeature','Alley']
nan_na= ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageYrBlt','Fence',
 'GarageFinish', 'GarageQual', 'GarageCond','GarageCars','GarageArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','FireplaceQu']
na=na.drop(dummy_cols+['SalePrice'],axis=1)


# In[ ]:


na_mod={}
for i in tqdm(na.columns):
    if i not in nan_na: 
        if na[i].dtype=='O':
            mx=df1[i].value_counts()[df1[i].value_counts()==df1[i].value_counts().max()].index[0]
            na[i]=na[i].replace(np.nan,mx)
            na_mod[i]=mx

        elif na[i].dtype=='float64':
            mx=df1[i].median()
            na[i]=na[i].replace(np.nan,mx)
        


# In[ ]:


#na.info()
    


# In[ ]:


na_mod   


# In[ ]:


for i in tqdm(nan_na):
     
    if na[i].dtype=='O':
        mx='NA'
        na[i]=na[i].replace(np.nan,mx)
        na_mod[i]=mx

    elif na[i].dtype=='float64':
        mx=0.0
        na[i]=na[i].replace(np.nan,mx)
        


# In[ ]:


for i in na.columns:
    df[i]=na[i]


# Since columns such as 'PoolQC','MiscFeature','Alley' have very few values, the model will not have enough samples for each class to learn. Hence we will create a dummy column which has value 1 if any of these features are present and 0 if it is not present

# In[ ]:


for i in dummy_cols:
    df[i+'_present']=(~df[i].isnull()).astype(int)
    df=df.drop(i,axis=1)


# In[ ]:


df.info()


# All null values have been filled. Now we can focus on feature engineering, feature selection and encoding

# In[ ]:


df['MSSubClass']=df['MSSubClass'].astype(str)
objs=[]
for i in df.columns:
    if df[i].dtype=='O':
        objs.append(i)


# Because MSsubclass is actually categorical we convet it to object inspite of it being int. This variable should be one-hot encoded.
# We can drop variable street,utilities,LandSlope,LandContour,heating,electrical,PoolArea as it has very few variation.BldgType and HouseStyle are covered in the MSsubclass column.
# 

# In[ ]:


for i in objs:
    print('\n')
    print(df[i].value_counts())


# In[ ]:


drop_cols=['Street','LandContour','Utilities','LandSlope','BldgType',
          'Heating','Electrical',
           'GarageArea','PoolArea','LotShape','HouseStyle']
qual=['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
      'KitchenQual','GarageQual','GarageCond']
basement=['BsmtFinType1','BsmtFinType2']
len(drop_cols)


# In[ ]:


d=df
d['total_bathroom']=d['BsmtFullBath']+d['BsmtHalfBath']*0.5+                     d['FullBath']+d['HalfBath']*0.5
d=d.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1)


# In[ ]:


d[basement]=df[basement]
for i in basement:
    d[i]=d[i].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0})
d['BsmtFinType1'].value_counts()


# In[ ]:


#df2['Condition_isnorm']=df2['Condition1'].apply(lambda x:0 if x=='Norm' else 1)
#df2=df2.drop('Condition1',axis=1)


# In[ ]:


d[qual]=df[qual]
for i in qual:
    d[i]=d[i].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0})


# In[ ]:


d.info()


# In[ ]:


d=d.drop(drop_cols,axis=1)


# In[ ]:


d['total_porchSF']=0
d['total_porchSF']=d['OpenPorchSF']+d['EnclosedPorch']+d['3SsnPorch']+d['ScreenPorch']+d['WoodDeckSF']
d=d.drop(['EnclosedPorch','OpenPorchSF','3SsnPorch','WoodDeckSF','ScreenPorch'],axis=1)


# In[ ]:


d['BsmtExposure']=d['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0})


# In[ ]:


drop_num=['GarageYrBlt','MoSold','MiscFeature_present',
          'Alley_present']
d=d.drop(drop_num,axis=1)


# Month of sale has no correlation to the price.
# GarageYrBlt is highly correlated to YearBuilt

# In[ ]:


d.corr().loc['SalePrice',]


# Now we try to understand the variables and their distribution to draw insights which will help us in further feature engineering
# 

# In[ ]:


for i in d.columns:
    if d[i].dtype=='O':
        sns.boxplot(x=d[i],y=d['SalePrice'])
        plt.show()


# From the above visualization we can see how each categorical variable affects the sale price of the house
#     

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=d,x='YrSold',y='SalePrice')

Year ofsale doesnt seem to affect the sale price much. We can see that houses sold in 2007 is not more expensive than one sold in 2006. Hence we can treat this as a categorical variable. It also has an added benifit that if recession and other factors connect to year are affecting the prices it can easily be identified if we make it categorical and use one hot encoding.

# In[ ]:


plt.figure(figsize=(30,20))
sns.boxplot(data=d,x='YearBuilt',y='SalePrice')


# In[ ]:


def yr_blt(x):
    if x<1901:
        return '1800s'
    elif x<1950:
        return '1900-50s'
    elif x<1960:
        return '1950s'
    elif x<1970:
        return '1960s'
    elif x<1980:
        return '1970s'
    elif x<1990:
        return '1980s'
    elif x<2000:
        return '1990s'
    elif x<2020:
        return '2000s'
d['year_built']=d['YearBuilt'].apply(yr_blt)
d['year_built'].value_counts()


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=d,x='year_built',y='SalePrice')


# The YearBuilt variable also doesnt behave like a numerical variable as it doesnt exactly hold any numeric meaning. We can see that a house built in 2000 is not greater than 1800. It doesnt make sense in a numeric sense. Hence we convert it into categoricalvariable 

# In[ ]:


for i in d.columns:
    if d[i].dtype!='O':
        sns.regplot(x=d[i],y=d['SalePrice'])
        plt.show()


# In[ ]:


d['yr_remod']=d['YearRemodAdd'].apply(yr_blt)
plt.figure(figsize=(20,10))
sns.boxplot(data=d,x='yr_remod',y='SalePrice')
plt.xticks(rotation=90)
d=d.drop('YearRemodAdd',axis=1)


# In[ ]:


#sns.regplot(x=np.log(d['GrLivArea']),y=d['SalePrice'])
d=d.drop('KitchenAbvGr',axis=1)


# In[ ]:


d=d.drop('PoolQC_present',axis=1)


# # Model Building
# 
# Now we will prepare the data, fit and tune algorithms and choose one based on its performance

# In[ ]:


#d=d.drop(['ext1+ext2','cond1+cond2'],axis=1)
d['YrSold']=d['YrSold'].astype(str)
obj=[i for i in d.columns if d[i].dtype=='O']

d_final=pd.get_dummies(d,columns=obj,drop_first=True)
d_final.shape


# In[ ]:


train=d_final[d_final.SalePrice.isnull()==False].drop('Id',axis=1)
test=d_final[d_final.SalePrice.isnull()==True].drop(['Id','SalePrice'],axis=1)
test_id=d_final[d_final.SalePrice.isnull()==True]['Id']


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from statsmodels.api import OLS,add_constant


# In[ ]:


x=train.drop('SalePrice',axis=1)
y=train['SalePrice']
lr=LinearRegression()
lgb=LGBMRegressor()
lasso=Lasso(alpha=0.3,max_iter=10e6)
rf=RandomForestRegressor()


# We will check model performance once on the original target and then on target with log transformation as it is highly right skewed. Based on its performance on the test data we can choose the model

# In[ ]:


x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.3,random_state=6969)
models={'Linear Regression':lr,'Random Forest':rf,'Light GBM':lgb,'Lasso':lasso}
mod1={}
for i in models:
    print('\n',i)
    a=models[i].fit(x_tr,y_tr)
    pred_ts=a.predict(x_ts)
    pred_tr=a.predict(x_tr)
    print('\nTrain Scores')
    tr_r2=r2_score(y_tr,pred_tr)
    tr_mae=mean_absolute_error(y_tr,pred_tr)
    tr_mse=mean_squared_error(y_tr,pred_tr)
    print('R2:',tr_r2)
    print('MAE:',tr_mae)
    print('MSE:',tr_mse)
    print('\nTest Scores')
    ts_r2=r2_score(y_ts,pred_ts)
    ts_mae=mean_absolute_error(y_ts,pred_ts)
    ts_mse=mean_squared_error(y_ts,pred_ts)
    print('R2:',ts_r2)
    print('MAE:',ts_mae)
    print('MSE:',ts_mse)
    mod1[i]=a


# In[ ]:


x3_tr,x3_ts,y3_tr,y3_ts=train_test_split(x,np.log(y),test_size=0.3,random_state=6969)
models={'Linear Regression':lr,'Random Forest':rf,'Light GBM':lgb,'Lasso':lasso}
mod3={}
for i in models:
    print('\n',i)
    a=models[i].fit(x3_tr,y3_tr)
    pred_ts=a.predict(x3_ts)
    pred_tr=a.predict(x3_tr)
    print('\nTrain Scores')
    tr_r2=r2_score(y3_tr,pred_tr)
    tr_mae=mean_absolute_error(y3_tr,pred_tr)
    tr_mse=mean_squared_error(y3_tr,pred_tr)
    print('R2:',tr_r2)
    print('MAE:',tr_mae)
    print('MSE:',tr_mse)
    print('\nTest Scores')
    ts_r2=r2_score(y3_ts,pred_ts)
    ts_mae=mean_absolute_error(y3_ts,pred_ts)
    ts_mse=mean_squared_error(y3_ts,pred_ts)
    print('R2:',ts_r2)
    print('MAE:',ts_mae)
    print('MSE:',ts_mse)
    mod3[i]=a


# Since Light GBM with log transform gives the best result we will use that model and predict on test data and save it to a csv file

# In[ ]:


fm=lgb.fit(x,np.log(y))
pred=fm.predict(test)
p=np.exp(pred)
result=pd.DataFrame({'SalePrice':p},index=test_id)
result.to_csv('result2_lgb.csv')

