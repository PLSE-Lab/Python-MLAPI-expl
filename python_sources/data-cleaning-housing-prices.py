#!/usr/bin/env python
# coding: utf-8

# Data Cleaning is an integral part of Pre-processing, as it helps to prepare the data for ML algorithms, and in some cases it also helps to create more sense of the data. But knowledge of data is equally important as it helps in making decisions like which data is important and which is not so that we can simplify our dataset.
# 
# The Data Cleaning Workflow that I will folow in this notebook is,
# 
# <ul><li>Handle Missing Values</li>
#     <li>Scrub for Duplicate Values</li>
#     <li>Handle Categorical Data</li>
#     <li>Perform Feature Scaling</li></ul>
#     
# I will be using Housing Prices dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# 
# ![housebanner.png](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)
# 
# 

# In[ ]:


#import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

#Read Data
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
#Having quick look at the data
train.head()


# <hr></hr>
# 
# ## Identify Missing values
# 
# Let's start with identifying the missing values

# In[ ]:


null_list = train.isnull().sum()
print(null_list[null_list != 0] )


# In[ ]:


null_list_test=test.isnull().sum()
print(null_list_test[null_list_test!=0])


# ## Handle Missing Values
# 
# There are 6 ways of handling missing values(that i'am aware of)
# <ul><li>Delete rows</li>
#     <li>Delete Column</li>
#     <ul><li>Possible when more than 50% data of the column is missing</li></ul>
#     <li>Replace null values with Mean</li>
#     <li>Replace null value with category(in this case 0, indicating zero floors)</li>
#     <li>predict the missing value</li>
#     <li>use algorithm that can work with missing values</li></ul>

# ### Replace null value with category
# as per the data description provided with data, there are <strong>features that have Na as valid data</strong>,
#     Alley,
#     BsmtQual,
#     BsmtCond,
#     BsmtExposure,
#     BsmtFinType1,
#     BsmtFinType2,
#     FireplaceQu,
#     GarageType,
#     GarageFinish,
#     GarageQual,
#     GarageCond,
#     PoolQC,
#     Fence,
#     MiscFeature
#     
# for example, The NA in FireplaceQu means no Fireplace(can be verified with the Fireplaces Feature)

# In[ ]:


check_fireplace = train[['Fireplaces','FireplaceQu']]
check_fireplace[check_fireplace['FireplaceQu'].isnull()].head()


# Now we need to replace Nan in all above mentioned features with appropriate values.
# 
# For that, we will have a look at their datatype first

# In[ ]:


valid_na_list = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
train[null_list[valid_na_list].index].dtypes


# In[ ]:


test[null_list_test[valid_na_list].index].dtypes


# In[ ]:


#Checking the unique number of values in each
train[null_list[valid_na_list].index].nunique()


# In[ ]:


#Let's replace NA by 0
train[valid_na_list]=train[valid_na_list].fillna('0')
test[valid_na_list]=test[valid_na_list].fillna('0')
#validating the null list again
null_list=train.isnull().sum()
null_list_test=test.isnull().sum()
print('-----NA in train set-----')
print(null_list[null_list != 0])
print()
print('-----NA in test set-----')
print(null_list_test[null_list_test!=0])


# In[ ]:


#Now check the dtype of left features
train[null_list[null_list !=0].index].dtypes


# Next we see that LotFrontage and GarageYrBlt has around 18% and 5% missing values resp.
# As the number is low but not too low, we roll out the possibility to delete Rows/Columns, and go ahead to replace null values with their respective means.
# 
# ### Replace Null with Mean

# In[ ]:


mean_LF = round(train['LotFrontage'].mean(),2)
train['LotFrontage'] = train['LotFrontage'].fillna(mean_LF)

mean_GB = round(train['GarageYrBlt'].mean(),2)
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(mean_GB)

#validating the null list again
null_list=train.isnull().sum()
null_list[null_list != 0]


# These null values contribute < 0.5% to the dataset hence will delete the rows
# 
# ### Delete Rows with null value
# 
# The current shape of Train set is (1460, 81)

# In[ ]:


train.dropna(0,inplace=True)
train.shape


# Now as checked the rows have decreased by 8

# In[ ]:


#validating the null list again
null_list=train.dropna(0).isnull().sum()
null_list[null_list != 0]


# #### Great! We have handeled all the missing values
# 
# So, the next step in our Data Cleaning process is to scrub data for Duplicate values
# <hr></hr>
# 
# ## Scrub Data for Duplicates

# In[ ]:


train.duplicated().sum()


# #### Looks like duplicate rows do not exist. Great!!
# 
# So we move on to the next step.
# <hr></hr>
# 
# ## Handle Categorical Data
# 
# 
# There are multiple ways to handle Categorical Data,
# <ul><li>Drop Categorical features
#         <ul><li>This is easiest way, but will work only if column do not contain meaningful data. So, I will leave this dropping part to feature selection stage</ul>
#     </li>
#     <li>One-Hot Encoding</li>
#     <li>Label Encoding</li>
#     <li>Count Encoding</li>
#     <li>Target Encoding</li>
#     <li>CatBoost Encoding</li></ul>

# In[ ]:


#List out all categorical features
cat_feats = train.columns[train.dtypes== object]
cat_feats


# Now let's check the number of unique values in each so that we can decide on the type of encoding to use

# In[ ]:


train[cat_feats].nunique()


# Now will try to identify the type of data (viz. nominal, ordinal, interval and ratio) and figure out which encoding to perform

# | Feature | Type | # of Unique Values | Encoding |
# |---------|:----:|:------------------:|----------|
# | MSZoning | nominal | 5 | One Hot |
# | Street | nominal | 2 | One Hot |
# | Alley | nominal | 3 | One Hot |
# | LotShape | ordinal | 4 | Label |
# | LandContour | ordinal | 4 | Label |
# | Utilities | ordinal | 2 | Label |
# | LotConfig | nominal | 5 | One Hot |
# | LandSlope | ordinal | 3 | Label |
# | Neighborhood | nominal | 25 | Will see later |
# | Condition1 | nominal | 9 | One Hot |
# | Condition2 | nominal | 8 | One Hot |
# | BldgType | nominal | 5 | One Hot |
# | HouseStyle | nominal | 8 | One Hot |
# | RoofStyle | nominal | 6 | One Hot |
# | RoofMatl | nominal | 8 | One Hot |
# | Exterior1st | nominal | 15 | Will see later |
# | Exterior2nd | nominal | 16 | Will see later |
# | MasVnrType | nominal | 4 | One Hot |
# | ExterQual | ordinal | 4 | Label |
# | ExterCond | ordinal | 5 | Label |
# | Foundation | nominal | 6 | One Hot |
# | BsmtQual | ordinal | 5 | Label |
# | BsmtCond | ordinal | 5 | Label |
# | BsmtExposure | ordinal | 5 | Label |
# | BsmtFinType1 | ordinal | 7 | Label |
# | BsmtFinType2 | ordinal | 7 | Label |
# | Heating | nominal | 6 | One Hot |
# | HeatingQC | ordinal | 5 | Label |
# | CentralAir | nominal | 2 | One Hot |
# | Electrical | nominal | 5 | One Hot |
# | KitchenQual | ordinal | 4 | Label |
# | Functional | ordinal | 7 | Label |
# | FireplaceQu | ordinal | 6 | Label |
# | GarageType | ordinal | 7 | Label |
# | GarageFinish | ordinal | 4 | Label |
# | GarageQual | ordinal | 6 | Label |
# | GarageCond | ordinal | 6 | Label |
# | PavedDrive | nominal | 3 | One Hot |
# | PoolQC | ordinal | 4 | Label |
# | Fence | ordinal | 5 | Label |
# | MiscFeature | nominal | 5 | One Hot |
# | SaleType | nominal | 9 | One Hot |
# | SaleCondition | nominal | 6 | One Hot |

# In[ ]:


feat_for_label = ['LotShape','LandContour','Utilities','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure',
'BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',
'GarageCond','PoolQC','Fence']

feat_for_oneHot = ['MSZoning','Street','Alley','LotConfig','Condition1','Condition2','BldgType','HouseStyle','RoofStyle',
'RoofMatl','MasVnrType','Foundation','Heating','CentralAir','Electrical','PavedDrive','MiscFeature','SaleType','SaleCondition']


# In[ ]:


#Will perform Label Encoding
from sklearn.preprocessing import LabelEncoder
encoded_train=train[train.columns[train.dtypes != object]]

LE = LabelEncoder()
for feature in feat_for_label:
    encoded_train[feature] = LE.fit_transform(train[feature])

print(encoded_train.shape)
print(encoded_train.head())


# In[ ]:


#Will perform One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

OH = OneHotEncoder(sparse=False,handle_unknown='ignore')
OH_vals=pd.DataFrame(OH.fit_transform(train[feat_for_oneHot]),columns=OH.get_feature_names(feat_for_oneHot))

#OH was changing index(adding rows) so needed to reset_index, but reset_index adds new column so added drop=True to avoid that
encoded_train.reset_index(drop=True,inplace=True)
encoded_train=pd.concat([encoded_train,OH_vals],axis=1)
encoded_train.index = train.index

print(encoded_train.shape)
print(encoded_train.head())


# encoded_train = numerical + Label encoded = 59 columns<br></br>
# encoded = encoded_train + sum(unique value in each OH feature) = 59 + 105 = 164 columns
# 
# Which means we are on right track!! Great!
# 
# Next need to figure out what needs to be done with below columns
# 
# | Feature | Type | unique values | Encoding |
# |---------|:----:|:-------------:|----------|
# | Neighborhood | nominal | 25 | Will see now |
# | Exterior1st | nominal | 15 | Will see now |
# | Exterior2nd | nominal | 16 | Will see now |

# Let's perform the Hashing trick

# In[ ]:


obj_list = ['Neighborhood','Exterior1st','Exterior2nd']
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=6, input_type='string')
encoded_train.reset_index(drop=True,inplace=True)
for i in obj_list:
    hashed_features = fh.fit_transform(train[i])
    column = [i+'_1',i+'_2',i+'_3',i+'_4',i+'_5',i+'_6']
    hashed_features = pd.DataFrame(hashed_features.toarray(),columns=column)
    print(hashed_features.shape)
    encoded_train=pd.concat([encoded_train, hashed_features], 
              axis=1)
encoded_train.index=train.index
encoded_train.shape


# In[ ]:


encoded_train.head()


# we have 164 columns after OH
# next there were 3 columns for hashing
# 3 features * 6 hashed columns = 18 new columns
# 164 old columns + 18 new columns = 182 total columns

# In[ ]:


#is there any categorical column?
(encoded_train.dtypes == object).sum()


# There is no categorical column left. Nice!!
# 
# ## Feature Scaling
# 
# Need to convert all the features to approximately same range
# 
# 

# In[ ]:


#Drop the ID column 
encoded_train.drop(axis=1,columns='Id',inplace=True)
encoded_train.describe()


# Performing Mean Normalization,
# ![meannormalization.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/5c591a0eeba163a12f69f937adbae5886d6273db)

# In[ ]:


for i in range(encoded_train.shape[1]):
    mean  = encoded_train.iloc[:,i].min()
    max_min = encoded_train.iloc[:,i].max() - encoded_train.iloc[:,i].min()
    temp = encoded_train.iloc[:,i]
    encoded_train.iloc[:,i] = round((temp - mean) / max_min,3)

encoded_train.describe()


# ### To Be Continued...

# If you have any questions or advice, I will be happy to hear them.

# In[ ]:




