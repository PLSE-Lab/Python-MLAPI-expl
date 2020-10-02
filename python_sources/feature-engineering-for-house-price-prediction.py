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


# # Feaure engineering for House price prediction Problem. 
# 1. Seperation of numerical and categorical columns
# 
# 1a. removing the outliers using the box plot
# 
# 1b. removing the positive skewness by applying log function
# 
# 2. Missing value imputation of numerical and categorical columns
# 3. onehot encoding
# 4. preparing the final data Frame

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


housetrain=pd.read_csv("../input/train.csv")


# In[ ]:


housetest=pd.read_csv("../input/test.csv")


# In[ ]:


house_train_num=housetrain.select_dtypes(include=[np.number])


# In[ ]:


house_train_cat=housetrain.select_dtypes(include=['object'])


# In[ ]:


house_train_num.columns


# In[ ]:


plt.boxplot(house_train_num["SalePrice"])


# In[ ]:


plt.hist(house_train_num["SalePrice"])


# # Removing outliers rows of label value
# This reduces the data dimension

# In[ ]:


h_train = housetrain[housetrain["SalePrice"]<300000]


# In[ ]:


plt.boxplot(h_train["SalePrice"])


# In[ ]:


h_train.shape


# # Removing the positive skewness by applying log function

# In[ ]:


plt.boxplot(np.log(housetrain["SalePrice"]))


# In[ ]:


house_train_num_corr=house_train_num.corr()


# # Correlationship of "Salesprice" label 

# In[ ]:


house_train_num_corr=house_train_num.corr()


# In[ ]:


house_train_num_corr["SalePrice"]


# # Considering only columns which are having correlationship value lesser than -0.3 and above 0.3. 
# Which are having high relationship on target variable "SalesPrice"

# In[ ]:


house_train_num_cols = []
house_train_num_cols.extend(house_train_num_corr[(house_train_num_corr["SalePrice"]>0.3) ].index.values)
house_train_num_cols.extend(house_train_num_corr[(house_train_num_corr["SalePrice"]<-0.3) ].index.values)


# In[ ]:



house_train_num_cols


# In[ ]:


h_train_num_col_filtered=house_train_num[house_train_num_cols]


# In[ ]:


(house_train_num.isnull().sum().sort_values(ascending=False))


# In[ ]:



for hc in ["LotFrontage","GarageYrBlt","MasVnrArea"]:
    print (hc)
    print(house_train_num[hc].mean())
    print(house_train_num[hc].median())


# # For this three variables there is mean distortion, hence replacing the null values with median

# In[ ]:


for col in ["LotFrontage","GarageYrBlt","MasVnrArea"]:
    h_train_num_col_filtered[col].fillna(h_train_num_col_filtered[col].median(),inplace=True)


# In[ ]:


(house_train_cat.isnull().sum().sort_values(ascending=False))


# # For the "PoolQC","MiscFeature","Alley","Fence","FireplaceQu" categorical columns more than 50% values are null values hence added a new value "New Value" in null value places

# In[ ]:


for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"]:
    house_train_cat[col].fillna('No Value',inplace=True)


# # For "GarageCond","GarageQual","GarageFinish","GarageType","BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtQual","BsmtCond","MasVnrType","Electrical" categorical columns adding value_counts().idxmax() in place of null values

# In[ ]:


for col in ["GarageCond","GarageQual","GarageFinish","GarageType","BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtQual","BsmtCond","MasVnrType","Electrical"]:
    house_train_cat[col].fillna(house_train_cat[col].value_counts().idxmax(),inplace=True)


# In[ ]:


house_train_cat["LotConfig"].unique()


# In[ ]:


for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageCond","GarageQual","GarageFinish","GarageType","BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtQual","BsmtCond","MasVnrType","Electrical"]:
     if len(house_train_cat[col].value_counts())< 10:
            print (col)


# # For the above catergorical columns max categories are lesser than 10, They are ready for applying  one hot ecoding

# In[ ]:


house_train_cat1 = house_train_cat
for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageCond","GarageQual","GarageFinish","GarageType","BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtQual","BsmtCond","MasVnrType","Electrical"]:
    n_df = pd.get_dummies(house_train_cat1[col])
    house_train_cat1 = pd.concat([house_train_cat1,n_df],axis=1)
    house_train_cat1.drop([col],axis=1,inplace = True)
    print(col)


# In[ ]:


house_train_cat1.columns


# In[ ]:


house_train_cat1.shape


# # Preparing final Dataframe with numerical and categorical columns

# In[ ]:


house_train_df=pd.concat([h_train_num_col_filtered,house_train_cat1],axis=1)


# In[ ]:


house_train_df.shape


# DataFrame ready for modelling

# In[ ]:




