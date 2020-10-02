#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Feature Engineering**
# * Missing values
# * Temporal variables
# * Categorical variables
# * Standardize the values of variables to the same range

# In[ ]:


df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df.head()


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(df,df['SalePrice'],test_size=0.1,random_state=0)


# In[ ]:


print(xtrain.shape)
print(xtest.shape)


# # **Missing Values**
# handle the nan values in categorical features

# In[ ]:


catfeanan=[fea for fea in df.columns if df[fea].isnull().sum()>1 and df[fea].dtypes=='O']
print(catfeanan)


# **Replace NaN values**

# In[ ]:


def replace_cat_fea(df,catfeanan):
    dat=df.copy()
    dat[catfeanan]=dat[catfeanan].fillna('Missing')
    return dat

df=replace_cat_fea(df,catfeanan)
df[catfeanan].isnull().sum()


# **Thus we have zero missing values in all categorical variables**

# In[ ]:


df.head()


# # Missing values for numerical variables****

# In[ ]:


numfeanan=[fea for fea in df.columns if df[fea].isnull().sum()>1 and df[fea].dtypes!='O']
print(numfeanan)


# **Replace median values for numerical missing values**

# In[ ]:


for fea in numfeanan:
    med_val=df[fea].median()
    
    df[fea+'nan']=np.where(df[fea].isnull(),1,0)
    df[fea].fillna(med_val,inplace=True)
    
    
df[numfeanan].isnull().sum()


# In[ ]:


df.head()


# # Temporal variable****
# Date-time variable

# In[ ]:


for fea in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    df[fea]=df['YrSold']-df[fea]


# In[ ]:


df[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# # Gaussian Distribution****
# for skewed values

# In[ ]:


num_fea=['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']
for fea in num_fea:
    df[fea]=np.log(df[fea])
    plt.hist(df[fea])
    plt.show()


# **Now all the above features are not in skewed form which is of gaussian distribution**

# # Handling Rare Categorical Features****
# Removing categorical features with less than 1% of the total observation

# In[ ]:


catfea=[fea for fea in df.columns if df[fea].dtypes=='O']
catfea


# In[ ]:


for fea in catfea:
    temp=df.groupby(fea)['SalePrice'].count()/len(df)
    dftemp=temp[temp>0.01].index
    df[fea]=np.where(df[fea].isin(dftemp),df[fea],'RareValue')


# In[ ]:


df.head(50)


# # Converting categorical to numerical variables****

# In[ ]:


for feature in catfea:
    labels_ordered=df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)


# In[ ]:


df.head()


# # Feature Scaling****
# Apply the scaling so that all features will be in the same range

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


feascale=[fea for fea in df.columns if fea not in ['Id','SalePrice']]
scale=MinMaxScaler()
scale.fit(df[feascale])


# # Converting into a dataframe****

# In[ ]:


dat=pd.concat([df[['Id','SalePrice']].reset_index(drop=True),pd.DataFrame(scale.transform(df[feascale]),columns=feascale)],axis=1)


# In[ ]:


dat.head()


# In[ ]:


dat.to_csv('traindat.csv',index=False)


# # Feature Selection****

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[ ]:


df=pd.read_csv('traindat.csv')


# In[ ]:


df.head()


# In[ ]:


ytrain=df[['SalePrice']]


# In[ ]:


xtrain=df.drop(['Id','SalePrice'],axis=1)


# **Lasso: Penalizes the features with higher weights.**
# **SelectModel: It chooses the features of good weights**

# In[ ]:


fea_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
fea_model.fit(xtrain,ytrain)


# In[ ]:


fea_model.get_support()


# **True indicates the important features and False shows the feature of less importance**

# In[ ]:


sel_fea=xtrain.columns[(fea_model.get_support())]
print('Total No of Feature',xtrain.shape[1])
print('NO of selected feature',len(sel_fea))


# In[ ]:


print(sel_fea)


# In[ ]:


xtrain=xtrain[sel_fea]


# In[ ]:


xtrain.head()


# # **Train Test Split**

# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(xtrain,ytrain,test_size=0.2)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg=LinearRegression()
mod=reg.fit(xtrain,ytrain)


# In[ ]:


mod.score(xtest,ytest)


# In[ ]:




