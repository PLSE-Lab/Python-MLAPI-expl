#!/usr/bin/env python
# coding: utf-8

# >****Absolutely, this is the first kaggle project for my data science career. I need to be familiar with the working way of all friends in data science and kaggle world. Hope this will be a piece of unforgetable experience and a good start for my kaggle career.  
# BenJ - September 2018****

# **First of all**, before we start, we need to deeply understand what the problem is and what the variables means. 

# # **1. Data import and overview**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm
from scipy import stats
from sklearn import preprocessing


# In[ ]:


train_data = pd.read_csv('../input/train.csv') #train_data
test_data = pd.read_csv('../input/test.csv')  #test_data


# In[ ]:


train_data.head(5)


# In[ ]:


test_data.head(5)


# Through overview train_data and test_data, there are several more important things to do.
# 1. Some text type variables need transforming to dummies variable. Test data should keep same rules with train data.
# 2. Some features covered the NaN value need be filling or dropping.
# 3. In feature engineering, numerical variables need to tranform as normal distribution.

# # **2. Feature Engineering**  
# - ## Missing data filling  
#     1. Numerical features NaN value filling  
#     1. Text type features NaN value filling 

# ## About Train Set

# ### Numerical Features

# In[ ]:


#Numerical features NaN value filling
#train_data
num_feat_index = train_data.dtypes[train_data.dtypes != "object"].index
print('Train Data Numerical features are:' )
print(num_feat_index)
train_num_na = train_data[num_feat_index].isnull().sum()/len(train_data)
train_num_na=train_num_na.drop(train_num_na[train_num_na == 0].index).sort_values(ascending = False)
train_num_na = pd.DataFrame({'Train Data Missing Ratio': train_num_na})
train_num_na


# In[ ]:


plt.xticks(rotation = '0')
sns.barplot(x=train_num_na.index, y=train_num_na.iloc[:,0])


# Check each above feature's meaning based on data_description.txt file. If the data is surely missing, drop or filling.  
# **LotFrontage**:  linear feet of street connected to property.    
# Replace NaN with Mode.  
# **GarageYrBlt**: year garage was built.  
# Considering the Garage has seven related variables which covered three numerical type and four text type, filling any data needed to refer to all attributes. I'll do this later.  
# **MasVnrArea**: masonry veneer area in square feet.
# The NaN may have two meaning, none veneer or missing. Like Garage, do this later.

# In[ ]:


#for column in ['LotFrontage','GarageYrBlt']:
 #   train_data[column]=train_data[column].fillna(train_data[column].mode()[0])
train_data['LotFrontage']=train_data['LotFrontage'].fillna(train_data['LotFrontage'].mode()[0])


# ### Text Type Features

# In[ ]:


#Text type features NaN value filling
#train_data
text_feat_index = train_data.dtypes[train_data.dtypes == "object"].index
print('Train Data Text Type features are:' )
print(text_feat_index)
train_text_na = train_data[text_feat_index].isnull().sum()/len(train_data)
train_text_na=train_text_na.drop(train_text_na[train_text_na == 0].index).sort_values(ascending = False)
train_text_na = pd.DataFrame({'Train Data Missing Ratio': train_text_na})
plt.xticks(rotation = '90')
sns.barplot(x = train_text_na.index, y = train_text_na.iloc[:,0])


# In[ ]:


train_text_na


# Like numerical features, checking each feature's meaning, then decide to drop or fill.   
#   
# **PoolQC**: Pool quality. We find 'PoolArea' is only feature relevant to PoolQC. Back to PoolArea, numerical feature without NaN value. So, we need to make sure NaN in PoolQC means No Pool or missing evaluation value.  
# **MiscFeature**: Miscellaneous feature not covered in other categories. NaN means no other misc features.  
# **Alley**: Na means no alley access.  
# **Fence**: Na means no fence.  
# **FireplaceQu**: Na means no fireplace.    
# **GarageCond**: Na means no garage.  
# **GarageQual**: Na means no garage.  
# **GarageFinish**: Na means no garage.  
# **GarageType**: Na means no garage.  
# **BsmtFinType1**: Na means no basement.  
# **BsmtFinType2**: Na means no basement.  
# **BsmtExposure**: Na means no basement.  
# **BsmtCond**: Na means no basement.  
# **BsmtQual**: Na means no basement.  
# **MasVnrType**: Na means no masonry veneer.  
# **Electrical**: Na means missing.  
#   
#   Note: I just fill na with 'None' as dummies variable.
# 
# 
# 
# 

# **PoolQC**

# In[ ]:


train_data.loc[train_data['PoolArea']>0, ['PoolQC','PoolArea']]


# From above result, we could know when Pool exists in house the evaluation value isn't missing. So, that also means the NaN in PoolQC just have one possible that there is no Pool. Here, the NaN of PoolQC will be filled as 0.

# In[ ]:


train_data['PoolQC']=train_data['PoolQC'].fillna('None')


# **MiscFeature, Alley, Fence, FireplaceQu, Elecrical**  
# These features we just think the na value means no this feature item. After all we have no way to judge if it has other possibility.

# In[ ]:


columns = ['MiscFeature','Alley','Fence','FireplaceQu','Electrical']
for column in columns:
    train_data[column] = train_data[column].fillna('None')


# **Features about Garage**  
# The features about Garage covers t numerical type, like GarageArea and text type, like GarageType. The GarageArea is without Na, so, we could take it as a reference to judge other text type na value is 'None' or missing.

# In[ ]:


train_data.loc[train_data['GarageCars']==0, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType','GarageCars','GarageYrBlt']].head(10)


# GarageArea and GarageCars both equal to 0 means no Garage, so other revelant garage features' value should 'None' or 0.

# In[ ]:


train_data.loc[train_data['GarageCars']==0, ['GarageYrBlt']] = train_data.loc[train_data['GarageCars']==0, ['GarageYrBlt']].fillna(0)
train_data.loc[train_data['GarageCars']==0, ['GarageCond','GarageQual','GarageFinish','GarageType']]=train_data.loc[train_data['GarageCars']==0, ['GarageCond','GarageQual','GarageFinish','GarageType']].fillna('None')


# In[ ]:


inx = (train_data['GarageCars']>0)&((train_data['GarageCond'].isnull())|(train_data['GarageQual'].isnull())|(train_data['GarageFinish'].isnull())|(train_data['GarageType'].isnull())|(train_data['GarageArea'].isnull())|(train_data['GarageYrBlt'].isnull()))
train_data.loc[inx, ['GarageArea','GarageCars','GarageYrBlt','GarageCond','GarageQual','GarageFinish','GarageType']]


# In[ ]:


train_data[['GarageArea','GarageCars','GarageYrBlt','GarageCond','GarageQual','GarageFinish','GarageType']].isnull().sum()


# The above shows, all missing data about garage have been filled. 

# **Features about Basement**  
# The same way as Garage feature, here is 5 basement features with Na.

# In[ ]:


train_data.loc[train_data['TotalBsmtSF']==0,['TotalBsmtSF','BsmtFinType1','BsmtFinType2','BsmtExposure','BsmtCond','BsmtQual']]


# In[ ]:


for column in ['BsmtFinType1','BsmtFinType2','BsmtExposure','BsmtCond','BsmtQual']:
    train_data[column]=train_data[column].fillna('None')


# **MasVnrType**

# In[ ]:


inx = train_data['MasVnrArea'].isnull() | train_data['MasVnrType'].isnull()
train_data.loc[inx,['MasVnrArea','MasVnrType']]


# The result shows that the Masonry veneer related data match each other. No masonry veneer, no mas type, no mas area.  
# So, easily, just fill 0 and None.

# In[ ]:


train_data['MasVnrType']=train_data['MasVnrType'].fillna('None')
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)


# Check if all text type features filling work done.

# In[ ]:


print(len(text_feat_index[train_data[text_feat_index].isnull().sum()/len(train_data)==0])/len(text_feat_index)*100,'%')


# Check if all features filling work done.

# In[ ]:


print(len(train_data.columns[train_data.isnull().sum()==0])/len(train_data.columns)*100,'%')


# - ## Text Type Feature Transforming to Dummies Variable  

# In this step, just one thing I think needed to say, that I want to set 'None' as 0, Others variables will be set a number as sequence.

# In[ ]:


# Function collecting dummies variables
def dummies(data, columns):
    dummiesgroup = {}
    for column in columns:
        dummies = {}
        variables = train_data[column].unique()
        num = 1
        for variable in variables:
            if variable == 'None':
                dummies[variable] = 0
            else:
                dummies[variable] = num
            num += 1
        dummiesgroup[column] = dummies
    return(dummiesgroup)   


# In[ ]:


dummiesgroup = dummies(train_data, text_feat_index)
dummiesgroup


# In[ ]:


new_train_data = train_data.copy()
for column in text_feat_index:
    new_train_data[column] = train_data[column].map(dummiesgroup[column])


# ## About Test Set

# In[ ]:


test_data.head(10)


# Following the way of thinking in train data, analyzing test data and handle it.

# ### Numerical features

# In[ ]:


#test_data
num_feat_index = test_data.dtypes[test_data.dtypes != "object"].index
print('Test Data Numerical features are:' )
print(num_feat_index)
test_num_na = test_data[num_feat_index].isnull().sum()/len(test_data)
test_num_na=test_num_na.drop(test_num_na[test_num_na == 0].index).sort_values(ascending = False)
test_num_na = pd.DataFrame({'Test Data Missing Ratio': test_num_na})
test_num_na


# **LotFrontage**

# In[ ]:


#for column in ['LotFrontage','GarageYrBlt']:
 #   test_data[column]=test_data[column].fillna(train_data[column].mode()[0])
test_data['LotFrontage']=test_data['LotFrontage'].fillna(train_data['LotFrontage'].mode()[0])   


# **Masonry veneer Class & Garage Class & Basement Class**

# These three class all have other related text type data, so we will analyzing them holding all data together later. 

# ### Text Type Features

# In[ ]:


txt_feat_index = test_data.dtypes[test_data.dtypes == "object"].index
print('Test Data text type features are:' )
print(txt_feat_index)
test_txt_na = test_data[txt_feat_index].isnull().sum()/len(test_data)
test_txt_na=test_txt_na.drop(test_txt_na[test_txt_na == 0].index).sort_values(ascending = False)
test_txt_na = pd.DataFrame({'Test Data Missing Ratio': test_txt_na})
test_txt_na


# **PoolQC**

# In[ ]:


test_data.loc[test_data['PoolArea']>0, ['PoolQC','PoolArea']]


# About pool, just two relevant information we have, PoolArea, PoolQC. Now, we just know there is pool by PoolArea, but we can't know the quality of it. 

# In[ ]:


test_data.loc[test_data['PoolArea']>0,'PoolQC'] = test_data.loc[test_data['PoolArea']>0,'PoolQC'].fillna(train_data.loc[train_data['PoolArea']>0,'PoolQC'].mode()[0])


# In[ ]:


test_data.loc[test_data['PoolArea']==0,'PoolQC'] = test_data.loc[test_data['PoolArea']==0,'PoolQC'].fillna('None')


# **MiscFeature, Alley, Fence, FireplaceQu**  
# These features we just think the na value means no this feature item. After all we have no way to judge if it has other possibility.

# In[ ]:


columns = ['MiscFeature','Alley','Fence','FireplaceQu']
for column in columns:
    test_data[column] = test_data[column].fillna('None')


# **Features about Garage**  

# In[ ]:


test_data.loc[test_data['GarageArea']==0, ['GarageArea','GarageCars','GarageYrBlt','GarageCond','GarageQual','GarageFinish','GarageType']].head(5)


# If GarageArea is equal to 0, that make sense other Garage relevant column is null. So, just filling 'None'.

# In[ ]:


test_data.loc[test_data['GarageArea']==0, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType']] = test_data.loc[test_data['GarageArea']==0, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType']].fillna('None')
test_data.loc[test_data['GarageArea']==0,'GarageYrBlt'] = test_data.loc[test_data['GarageArea']==0,'GarageYrBlt'].fillna(0)


# Now, check if exists another situation that there is garage but missing garage type.

# In[ ]:


inx = (test_data['GarageArea'] > 0) & (test_data['GarageCond'].isnull() | test_data['GarageQual'].isnull() | test_data['GarageFinish'].isnull() | test_data['GarageType'].isnull() | (test_data['GarageYrBlt'].isnull()) | (test_data['GarageCars'].isnull()))
test_data.loc[inx, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType','GarageCars','GarageYrBlt']]


# Here, the information is missing, We don't know quality, interior finish and condition. 

# In[ ]:


test_data.loc[inx,'GarageCond'] = test_data.loc[inx,'GarageCond'].fillna(train_data.loc[train_data['GarageArea']>0,'GarageCond'].mode()[0])
test_data.loc[inx, 'GarageQual'] = test_data.loc[inx,'GarageQual'].fillna(train_data.loc[train_data['GarageArea']>0,'GarageQual'].mode()[0])
test_data.loc[inx,'GarageFinish'] = test_data.loc[inx,'GarageFinish'].fillna(train_data.loc[train_data['GarageType'] == 'Detchd','GarageFinish'].mode()[0])
test_data.loc[inx,'GarageYrBlt'] = test_data.loc[inx, 'GarageYrBlt'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageYrBlt'].mode()[0])


# Check again, see any data is filling for garage.

# In[ ]:


test_data[['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType']].isnull().sum()


# In[ ]:


inx = (test_data['GarageCond'].isnull() | test_data['GarageQual'].isnull() | test_data['GarageFinish'].isnull() | test_data['GarageType'].isnull())
test_data.loc[inx, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType','GarageCars','GarageYrBlt']]


# Still information was missing, garage type is detached, so garage area could not be 0.  

# In[ ]:


test_data.loc[inx,'GarageArea'] = test_data.loc[inx,'GarageArea'].fillna(train_data.loc[train_data['GarageType'] == 'Detchd','GarageArea'].mode()[0])
test_data.loc[inx,'GarageCars'] = test_data.loc[inx,'GarageCars'].fillna(train_data.loc[train_data['GarageType'] == 'Detchd','GarageCars'].mode()[0])
test_data.loc[inx,'GarageCond'] = test_data.loc[inx, 'GarageCond'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageCond'].mode()[0])
test_data.loc[inx,'GarageQual'] = test_data.loc[inx, 'GarageQual'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageQual'].mode()[0])
test_data.loc[inx,'GarageFinish'] = test_data.loc[inx, 'GarageFinish'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageFinish'].mode()[0])
test_data.loc[inx,'GarageYrBlt'] = test_data.loc[inx, 'GarageYrBlt'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageYrBlt'].mode()[0])


# In[ ]:


test_data[['GarageArea','GarageCars','GarageYrBlt','GarageCond','GarageQual','GarageFinish','GarageType']].isnull().sum()


# **Masonry veneer Class**

# In[ ]:


inx = test_data['MasVnrType'].isnull() | test_data['MasVnrArea'].isnull()
test_data.loc[inx, ['MasVnrType','MasVnrArea']]


# Here, the result shows just one rowd data means missing type, others is one type, none masonry veneer.

# In[ ]:


test_data.loc[(test_data['MasVnrType'].isnull()) & (test_data['MasVnrArea']>0),'MasVnrType'] = test_data.loc[(test_data['MasVnrType'].isnull()) & (test_data['MasVnrArea']>0),'MasVnrType'].fillna(train_data.loc[train_data['MasVnrArea']>0,'MasVnrType'].mode()[0])


# In[ ]:


test_data.loc[inx,'MasVnrType'] = test_data.loc[inx,'MasVnrType'].fillna('None')
test_data.loc[inx,'MasVnrArea'] = test_data.loc[inx,'MasVnrArea'].fillna(0)


# In[ ]:


test_data[['MasVnrType','MasVnrArea']].isnull().sum()


# **Basement**

# In[ ]:


inx = test_data['BsmtQual'].isnull()|test_data['BsmtCond'].isnull()|test_data['BsmtExposure'].isnull()|test_data['BsmtFinType1'].isnull()|test_data['BsmtFinType2'].isnull()|test_data['BsmtFinSF1'].isnull()|test_data['BsmtFinSF2'].isnull()|test_data['TotalBsmtSF'].isnull()|test_data['BsmtUnfSF'].isnull()|test_data['BsmtFullBath'].isnull()|test_data['BsmtHalfBath'].isnull()
test_data.loc[inx,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]


# Looks like it nees many steps to finish filling.

# - Locate no basement row

# In[ ]:


inx1 = (test_data['BsmtQual'].isnull()|test_data['BsmtCond'].isnull()|test_data['BsmtExposure'].isnull()|test_data['BsmtFinType1'].isnull()|test_data['BsmtFinType2'].isnull())&((test_data['BsmtFinSF1'] == 0)&(test_data['BsmtFinSF2']==0)&(test_data['TotalBsmtSF']==0)&(test_data['BsmtUnfSF']==0)&(test_data['BsmtFullBath']==0)&(test_data['BsmtHalfBath']==0))
test_data.loc[inx1, ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]


# In[ ]:


test_data.loc[inx1, ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]=test_data.loc[inx1, ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].fillna('None')


# - Locate unsure row

# In[ ]:


inx = test_data['BsmtQual'].isnull()|test_data['BsmtCond'].isnull()|test_data['BsmtExposure'].isnull()|test_data['BsmtFinType1'].isnull()|test_data['BsmtFinType2'].isnull()|test_data['BsmtFinSF1'].isnull()|test_data['BsmtFinSF2'].isnull()|test_data['TotalBsmtSF'].isnull()|test_data['BsmtUnfSF'].isnull()|test_data['BsmtFullBath'].isnull()|test_data['BsmtHalfBath'].isnull()
test_data.loc[inx,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]


# The No. 660 row, in which all features about basement are missing, just think as none basement

# In[ ]:


test_data.loc[inx,['BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']] = test_data.loc[inx,['BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']].fillna(0)


# In[ ]:


test_data.loc[inx,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]


# In[ ]:


test_data.loc[test_data['TotalBsmtSF']==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']] = test_data.loc[test_data['TotalBsmtSF']==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].fillna('None')
test_data.loc[inx,'BsmtExposure'] = test_data.loc[inx,'BsmtExposure'].fillna('No')
test_data.loc[inx, 'BsmtCond'] = test_data.loc[inx, 'BsmtCond'].fillna(train_data.loc[train_data['TotalBsmtSF']>0, 'BsmtCond'].mode()[0])
test_data.loc[inx, 'BsmtQual'] = test_data.loc[inx, 'BsmtQual'].fillna(train_data.loc[train_data['TotalBsmtSF']>0, 'BsmtQual'].mode()[0])


# In[ ]:


inx1 = (test_data['BsmtQual'].isnull()|test_data['BsmtCond'].isnull()|test_data['BsmtExposure'].isnull()|test_data['BsmtFinType1'].isnull()|test_data['BsmtFinType2'].isnull())&((test_data['BsmtFinSF1'] == 0)&(test_data['BsmtFinSF2']==0)&(test_data['TotalBsmtSF']==0)&(test_data['BsmtUnfSF']==0)&(test_data['BsmtFullBath']==0)&(test_data['BsmtHalfBath']==0))
test_data.loc[inx1, ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]


# **MSZoning & Utilities & Functional**

# In[ ]:


inx = test_data['MSZoning'].isnull() | test_data['Utilities'].isnull() | test_data['Functional'].isnull()
test_data.loc[inx,['MSZoning', 'Utilities', 'Functional']]


# In[ ]:


for column in ['MSZoning', 'Utilities', 'Functional']:
    test_data.loc[inx,column] = test_data.loc[inx,column].fillna(train_data[column].mode()[0])


# **KitchenQual**

# In[ ]:


inx = test_data['KitchenQual'].isnull()
test_data.loc[inx,'KitchenQual']


# In[ ]:


test_data.loc[inx,'KitchenQual'] = test_data.loc[inx,'KitchenQual'].fillna(train_data['KitchenQual'].mode()[0])


# **SaleType**

# In[ ]:


inx = test_data['SaleType'].isnull()
test_data.loc[inx,'SaleType']


# In[ ]:


test_data.loc[inx, 'SaleType'] = test_data.loc[inx, 'SaleType'].fillna(train_data['SaleType'].mode()[0])


# **Exterior**

# In[ ]:


inx = test_data['Exterior1st'].isnull() | test_data['Exterior2nd'].isnull()
test_data.loc[inx,['Exterior1st','Exterior2nd']]


# In[ ]:


test_data.loc[inx,['Exterior1st']] = test_data.loc[inx,['Exterior1st']].fillna(train_data['Exterior1st'].mode()[0])
test_data.loc[inx,['Exterior2nd']] = test_data.loc[inx, ['Exterior2nd']].fillna(train_data['Exterior2nd'].mode()[0])


# In[ ]:


print(len(test_data.columns[test_data.isnull().sum()==0])/len(test_data.columns)*100,'%')


# - ## Text Type Feature Transforming to Dummies Variable  

# In[ ]:


new_test_data = test_data.copy()
for column in text_feat_index:
    new_test_data[column] = test_data[column].map(dummiesgroup[column])


# In[ ]:


new_test_data.head(5)


# - ## 0-1 Standardization

# In[ ]:


train_x = new_train_data[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',       'SaleCondition']]


# In[ ]:


test_x = new_test_data[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',       'SaleCondition']]


# In[ ]:


scaler = preprocessing.MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


# In[ ]:


test_x


# In[ ]:


train_x = pd.DataFrame(train_x, columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',       'SaleCondition'])
test_x = pd.DataFrame(test_x, columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',       'SaleCondition'])


# In[ ]:


train_x.head(5)


# In[ ]:


train_x.shape


# In[ ]:


test_x.head(5)


# Untill now, the feature engineering work for train data and test data have been finished. A little tedious, sometimes boring. However, it's necessary. In this process, not only data transforming, nan data filling, but, more important, understanding what the data means.

# # **3. Data Analysis & Feature Selection**

# Target data analysis

# In[ ]:


sns.distplot(train_data['SalePrice'],fit=norm)
mu, sigma = norm.fit(train_data['SalePrice'])
plt.legend(['Normal dist.\n($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')
plt.figure()
stats.probplot(train_data['SalePrice'],plot=plt)


# Here, the 'SalePrice' looks like right skewed distribution. In order to boost our model predicting performance, I use log transform for target variable. As result, the distributon will be normal distribution. 

# In[ ]:


train_y = np.log1p(train_data['SalePrice'])


# In[ ]:


sns.distplot(train_y,fit=norm)
mu,sigma = norm.fit(train_y)
plt.legend(['Normal dist.\n($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')
plt.figure()
stats.probplot(train_data['SalePrice'],plot=plt)


# Feature Analysis

# Correlation map

# In[ ]:


corr = new_train_data.iloc[:,1::].corr()


# In[ ]:


plt.subplots(figsize=(12,9))
sns.heatmap(corr, vmax = 0.9, square=True)


# Here, normally, I just need to select top 10 features in rank of correlation coefficient. However, in practical case, this way should be carefully used. Actually, we should make sure the selected features are independent for each other high rank features. Of course, you could give this step up if you ignore the dimention disaster. Otherwise, when the model confirmed, this part could optimize our model.

# In[ ]:


#rank features according to the correlation coefficient with SalePrice 
columns = np.abs(corr['SalePrice']).sort_values(ascending = False).index[1::].tolist()
print(columns)


# In[ ]:


#Looking for high correlated variables
n=0
columnsA=[]
columnsB=[]
while n < len(columns):
    if columns[n] not in columnsB:
        cols = corr.columns[corr[columns[n]]>=0.8].tolist() #Assume corr coefficient 0.8 as valve value.
        cols.remove(columns[n])
        if len(cols)>0:
            for col in cols:
                if col!='SalePrice':
                    columnsB.append(col)
            print(n, columns[n], cols)
        columnsA.append(columns[n])
    n+=1


# Above output shows the strong correlated variables which corr efficient is over 0.7. The columnsA means relatively independent features and columnsB means the rest features after selection as shown as below.

# In[ ]:


print(columnsA)


# In[ ]:


print(columnsB)


# Next, I will use columnsA as features to make model selection, and in final, I will try combining columnsB to see the effect on model performance.

# # **4. Model Selection & Model Evaluation**

# ### **Model Selection**

# Four models will be used:  
# - Linear regression  
# - Support vector machine regression  
#     1. Kernel 'linear'  
#     1. Kernel  'poly'
#     1. Kernel 'rbf'
# - Nearest neighbors regression  
# - Decision tree regression  
# - KernelRidge

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
n_folds = 5
def rmse_kfold(model):
    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train_x[columnsA].values)
    rmse = np.sqrt(-cross_val_score(model, train_x[columnsA].values, train_y, scoring = "neg_mean_squared_error", cv = kf ))
    return(rmse)


# **Linear regression**

# In[ ]:


from sklearn import linear_model
lr = linear_model.LinearRegression()
lr_loss = rmse_kfold(lr)


# **SVR**

# In[ ]:


from sklearn.svm import SVR
#linear kernel
linear_svr_loss = []
for i in [1,10,1e2]:
#for i in [1,10,1e2,1e3,1e4]:
    linear_svr = SVR(kernel = 'linear', C=i)
    linear_svr_loss.append(rmse_kfold(linear_svr).mean())
plt.plot([1,10,1e2],linear_svr_loss)
plt.xlabel('C')
plt.ylabel('mean-loss')


# When C is 1, the model loss value is minimum.

# In[ ]:


linear_svr = SVR(kernel = 'linear', C=1)
linear_svr_loss = rmse_kfold(linear_svr)


# In[ ]:


#poly kernel
for i in [1,10,1e2,1e3,1e4]:
    poly_svr_loss=[]
    for j in np.linspace(2,9,10):
        poly_svr = SVR(kernel = 'poly',C=i, degree=j)
        poly_svr_loss.append(rmse_kfold(poly_svr).mean())
    plt.plot(np.linspace(2,9,10), poly_svr_loss, label='C='+str(i))
    plt.legend()
plt.xlabel('degree')
plt.ylabel('mean-loss')


# The rmse is increased with degree increasing and the tendency of different C is almost same. At last, C set 100, degree set 2.

# In[ ]:


poly_svr = SVR(kernel = 'poly',C=100, degree=2)
poly_svr_loss=rmse_kfold(poly_svr)


# In[ ]:


#rbf kernel
for i in [1,10,1e2,1e3,1e4]:
    rbf_svr_loss = []
    for j in np.linspace(0.1,1,10):
        rbf_svr = SVR(kernel = 'rbf', C=i, gamma=j)
        rbf_svr_loss.append(rmse_kfold(rbf_svr).mean())
    plt.plot(np.linspace(0.1,1,10), rbf_svr_loss, label='C='+str(i))
    plt.legend()
plt.xlabel('gamma')
plt.ylabel('mean-loss')


# When gamma is 0.1 and C is 1, the loss is minimum.

# In[ ]:


rbf_svr = SVR(kernel = 'rbf',C=1,gamma=0.1)
rbf_svr_loss = rmse_kfold(rbf_svr)


# **Nearest Neighbors Regression**

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn_loss = []
for n_neighbors in range(1,21):
    knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
    knn_loss.append(rmse_kfold(knn).mean())
plt.plot(np.linspace(1,20,20), knn_loss)
plt.xlabel('n-neighbors')
plt.ylabel('mean-loss')


# In KNN model, the number of neighbors need to be confirmed. The above figure shows that 6 could be best value. 

# In[ ]:


knn = KNeighborsRegressor(6, weights = 'uniform' )
knn_loss = rmse_kfold(knn)


# **Decision Tree Regression**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr_loss=[]
for n in range(1,11):
    dtr = DecisionTreeRegressor(max_depth = n)
    dtr_loss.append(rmse_kfold(dtr).mean())
plt.plot(np.linspace(1,10,10), dtr_loss)
plt.xlabel('max_depth')
plt.ylabel('mean-loss')


# In decision tree model, the tree depth need to be confirmed. The above figure shows that 7 could be best value.

# In[ ]:


dtr = DecisionTreeRegressor(max_depth = 7)
dtr_loss=rmse_kfold(dtr)


# **Kernel Ridge**

# In[ ]:


from sklearn.kernel_ridge import KernelRidge    
#linear kernel
kr_linear_loss = []
for i in np.linspace(0.1,2.0,8):
    kr = KernelRidge(alpha = i,kernel = 'linear')
    kr_linear_loss.append(rmse_kfold(kr).mean())
plt.plot(np.linspace(0.1,2.0,8), kr_linear_loss)
plt.xlabel('alpha')
plt.ylabel('mean-loss')


# When alpha is set between 1.25 and 1.5 in linear kernel of KernelRidge, the rmse values reach minimum. Here, fix alpha equals 1.3.

# In[ ]:


kr_linear = KernelRidge(alpha = 1.3,kernel = 'linear')
kr_linear_loss = rmse_kfold(kr_linear)


# In[ ]:


#poly kernel
for j in np.linspace(0.1,2.0,8):
    kr_poly_loss = []
    for i in np.linspace(0.01,0.1,10):
        kr = KernelRidge(alpha = j,kernel = 'poly', gamma = i)
        kr_poly_loss.append(rmse_kfold(kr).mean())
    plt.plot(np.linspace(0.01,0.1,10), kr_poly_loss,label='alpha:{:.2f}'.format(j))
    plt.legend(loc='upper right')
plt.xlabel('gamma')
plt.ylabel('mean-loss')


# The rmse value drop with alpha decreased and gamma has less effect on model than alpha. Here, fix alpha equals 0.1 and gamma equals 0.05. 

# In[ ]:


kr_poly = KernelRidge(alpha = 0.1,kernel = 'poly',gamma = 0.05)
kr_poly_loss = rmse_kfold(kr_poly)


# In[ ]:


#rbf kernel
for j in np.linspace(0.00001,0.0001,10):
    kr_rbf_loss = []
    for i in np.linspace(0.0005,0.02,10):
        kr = KernelRidge(alpha = j,kernel = 'rbf', gamma = i)
        kr_rbf_loss.append(rmse_kfold(kr).mean())
    plt.plot(np.linspace(0.0005,0.02,10), kr_rbf_loss,label='alpha:{:.5f}'.format(j))
    plt.legend(loc='upper right')
plt.xlabel('gamma')
plt.ylabel('mean-loss')


# The tendency of rmse value with gamma increased. Fix gamma equals 0.0025 and alpha equals 0.0001.

# In[ ]:


kr_rbf = KernelRidge(alpha = 0.0001,kernel = 'rbf', gamma = 0.0025)
kr_rbf_loss=rmse_kfold(kr_rbf)


# ### **Model Evaluation**

# In[ ]:


evaluating = {
    'lr': lr_loss,
    'linear_svr':linear_svr_loss,
    'polyl_svr':poly_svr_loss,
    'rbf_svr':rbf_svr_loss,
    'knn':knn_loss,
    'drt':dtr_loss,
    'kr_linear':kr_linear_loss,
    'kr_poly':kr_poly_loss,
    'kr_rbf':kr_rbf_loss
}
evaluating = pd.DataFrame(evaluating)
print(evaluating)


# In[ ]:


evaluating.plot.hist()


# In[ ]:


evaluating.hist(color='k',alpha=0.6,figsize=(8,7))


# - From above the first figure, clearly, KernelRidge's linear kernel is the worst and KernelRidges's rbf kernel seems well. 
# - In the second figure, the loss of dtr and knnl are more closed and around 0.2. The others model's loss distribution are wide. Here, through range of x axis, 'linear' kernel, 'rbf' kernel and linear model seems better.

# In[ ]:


evaluating.describe()


# ### **The effect of Data Import Dimension on Model Performance**

# In the end of feature selection part, I ranked all features by the correlated efficient with SalePrice. At the same time, I seperate all features into two parts according to corr efficient between each features just for taking out some repetition content.

# In[ ]:


n_folds = 5
def rmse_kfold1(model,corr_valve):
    corr1 = pd.DataFrame(corr.loc[columnsA,'SalePrice'])
    columns_split = corr1.loc[list(abs(corr1.loc[columnsA,'SalePrice'])>=corr_valve),'SalePrice'].index.tolist()
    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train_x[columns_split].values)
    rmse = np.sqrt(-cross_val_score(model, train_x[columns_split].values, train_y, scoring = "neg_mean_squared_error", cv = kf ))
    return(rmse)


# In[ ]:


loss={}
loss_mean={}
for valve in [0.5, 0.3, 0]:
    model_loss=[]
    model_mean_loss=[]
    for model in [lr,linear_svr,poly_svr,rbf_svr,knn,dtr,kr_poly,kr_rbf]:
        model_loss.append(rmse_kfold1(model,valve))
        model_mean_loss.append(rmse_kfold1(model,valve).mean())
    loss[str(valve)]=model_loss
    loss_mean[str(valve)]=model_mean_loss


# In[ ]:


sns.violinplot(data = pd.DataFrame(loss_mean))
plt.xlabel('valve value')
plt.ylabel('mean-loss')


# Through the above mean loss distribution, we can see that average level of model validation loss is lower with data dimention increased. On the other hand, with data dimention expanding, the performance of different models distributes much wider.

# In[ ]:


loss_mean = pd.DataFrame(loss_mean,index = ['lr','linear_svr','poly_svr','rbf_svr','knn','dtr','kr_poly','kr_rbf'])


# Here, I select the 'rbf' SVR model as final model.

# In[ ]:


test_y_predict = np.expm1(rbf_svr.fit(train_x[columnsA].values,train_y).predict(test_x[columnsA].values))


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = test_data['Id']
submission['SalePrice'] = test_y_predict


# In[ ]:


submission.to_csv('submission.csv',index = False)


# Till now, finally, this work has been done. Not perfect, in this project, I realise about model I truly need to learn much deeper. And also I need to be fast to make myself drop new project environment. Anyway, I will keep going and see next project.

# In[ ]:




