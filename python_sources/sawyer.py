#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()


# In[ ]:


df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()


# In[ ]:


df_test.info()


# In[ ]:


df_test.shape


# In[ ]:


df_train.shape


# In[ ]:


num=df_train.select_dtypes(exclude='object')
numcorr=num.corr()
plt.subplots(figsize=(20,1))
numcorr.sort_values(by = ['SalePrice'], ascending=False)
sns.heatmap(numcorr.sort_values(by = ['SalePrice'], ascending=False).head(1),cmap='Blues')


# In[ ]:


numcorr['SalePrice'].sort_values(ascending = False).to_frame()


# In[ ]:


df_combined = pd.concat((df_test, df_train), sort = False).reset_index(drop = True)
df_combined.drop(['Id'], axis=1, inplace=True)


# In[ ]:


df_combined.shape


# In[ ]:


df_combined.isnull().sum().sort_values(ascending = False)


# In[ ]:


num = df_combined.select_dtypes(exclude = 'object')
num


# In[ ]:


num.isnull().sum().sort_values(ascending = False)


# In[ ]:


df_combined['LotFrontage'] = df_combined['LotFrontage'].fillna(df_combined.LotFrontage.median())
df_combined['GarageYrBlt'] = df_combined['GarageYrBlt'].fillna(df_combined.GarageYrBlt.median())
df_combined['MasVnrArea']  = df_combined['MasVnrArea'].fillna(0)  


# In[ ]:


df_combined.select_dtypes(exclude = 'object').isnull().sum().sort_values(ascending = False)


# In[ ]:


df_combined[['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','GarageArea','GarageCars','BsmtUnfSF','TotalBsmtSF','BsmtFinSF2']] = df_combined[['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','GarageArea','GarageCars','BsmtUnfSF','TotalBsmtSF','BsmtFinSF2']].fillna(0)


# In[ ]:


df_combined.select_dtypes(exclude = 'object').isnull().sum().sort_values(ascending = False)


# ## Categorical features

# In[ ]:


allna = df_combined.isnull().sum() / len(df_combined)*100
MV = df_combined[allna.index.to_list()]


# In[ ]:


allna


# In[ ]:


catmv = MV.select_dtypes(include='object')
catmv.isnull().sum()


# In[ ]:


catmv.isnull().sum().sort_values()


# In[ ]:


###for few missing values
df_combined['Electrical']=df_combined['Electrical'].fillna(method='ffill')
df_combined['SaleType']=df_combined['SaleType'].fillna(method='ffill')
df_combined['KitchenQual']=df_combined['KitchenQual'].fillna(method='ffill')
df_combined['Exterior1st']=df_combined['Exterior1st'].fillna(method='ffill')
df_combined['Exterior2nd']=df_combined['Exterior2nd'].fillna(method='ffill')
df_combined['Functional']=df_combined['Functional'].fillna(method='ffill')
df_combined['Utilities']=df_combined['Utilities'].fillna(method='ffill')
df_combined['MSZoning']=df_combined['MSZoning'].fillna(method='ffill')


# In[ ]:


df_combined.select_dtypes(include='object').isnull().sum().sort_values()


# In[ ]:


for col in df_combined.columns:
    if df_combined[col].dtype == 'object':
        df_combined[col] =  df_combined[col].fillna('None')        


# In[ ]:


df_combined.select_dtypes(include='object')


# In[ ]:


cat = df_combined.select_dtypes(include='object')
cat.head()


# In[ ]:


for col in cat.columns:
    df_combined[col] = df_combined[col].astype('category')
    df_combined[col] = df_combined[col].cat.codes


# In[ ]:


df_combined


# In[ ]:


df_combined.info()


# In[ ]:





# In[ ]:


df_combined['GarageFinish'].value_counts()


# In[ ]:


catmv['GarageFinish'].unique()


# In[ ]:


## test_dataset
df_test1 = df_combined.iloc[:1459]
df_test1


# In[ ]:


## training_dataset
df_train1 = df_combined.iloc[1459:]
df_train1


# In[ ]:


df_preprocessed = pd.concat([df_train1, df_test1], axis=0)
df_preprocessed = df_preprocessed.reset_index(drop=True)
df_preprocessed.index.name = "Id"

file = "/kaggle/working/features_Sawyer.csv"
df_preprocessed.to_csv(file)


# In[ ]:


df_preprocessed


# In[ ]:


file = "/kaggle/working/features_Sawyer.csv"
load_data = pd.read_csv(file, index_col = 'Id')
load_data


# In[ ]:




