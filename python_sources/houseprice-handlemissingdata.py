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


# ## Libraries Required

# In[ ]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')


# ## Data Path

# In[ ]:


train_data_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
test_data_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'
cured_train_data_path = 'cured_train.csv'
cured_test_data_path = 'cured_test.csv'


# ## Initializers
# Lets read the data, do some initial level stuffs. Lets concat train and test data for preprocessing and analysis purpose

# In[ ]:


dataA = pd.read_csv(train_data_path)
dataB = pd.read_csv(test_data_path)
data = dataA.append(dataB)
dataA.shape, dataB.shape, data.shape


# In[ ]:


# Ensure there are no duplicates, by checking the uniquiness of Id field
data.Id.unique().size


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


lencoder = LabelEncoder()
oencoder = OneHotEncoder(sparse=False)


# In[ ]:


#check feature data types
def print_dtypes(df):
  for name, typ in df.dtypes.iteritems():
    print(f'{name}\t\t{typ}')
print_dtypes(data)


# ## Features with Missing Data

# In[ ]:


# Check for missing Data
def get_features_with_missing_data(df):
  features_with_missing_values = df.isnull().sum().sort_values(ascending=False)
  for feature, number in features_with_missing_values.iteritems():
    if number ==0: break
    print(f'{feature}\t\t\t{number}') 

get_features_with_missing_data(data)


# ## Handle Missing Data
# As seen in the above output, there are multiple fields which are missing values.
# 
# Some fields are allowed to have NA / None as value. So any categorical field which has NaN should be checked with related features before replacing them with NA / None according to the description file.

# ### PoolQC
# Lets Analyse PoolQC missing Values. Note that Pool may not be available for some houses in that case PoolQC value should be 'NA'. In data any NaN in PoolQC with PoolArea == 0 can be considered as house with No Pool and hence replace them with 'NA'

# In[ ]:


mpqc_with_qa = data[(data.PoolQC.isna()) & (data.PoolArea >0)]
print(f'{mpqc_with_qa.shape}, this says that there are 3 missing PoolQC with PoolArea available')
mpqc_with_qa[['PoolQC', 'PoolArea','LotArea','SalePrice']]


# In[ ]:


# Lets find the count of records on each category of PoolQC
pqc_with_qa = data[(data.PoolQC.notna()) & (data.PoolArea >0)]
print(f'{pqc_with_qa.shape}, this says that there are only 10 records with pool quality and pool area')
sns.catplot(data=data, x='PoolQC', kind='count')
plt.show()


# We can see that PoolQC etries are distributed between Ex, Fa, Gd category but not on TA category. Since we dono the QC of the pool and TA category is missing, lets replace the missing PoolQC value for the records which has PoolArea with TA.

# In[ ]:


record_ids = mpqc_with_qa.index
data.at[record_ids, 'PoolQC'] = 'TA'
mpqc_with_qa = data[(data.PoolQC.isna()) & (data.PoolArea >0)]
# replace Other Missing value for PoolQC with 'NA'
data.PoolQC.fillna('NA', inplace=True)


# ### MiscFeature
# Lets analyse MiscFeature Missing Values. Here, this feature is allowed to have NA. So if not MiscVal is > 0 and MiscFeature is NaN, then we can fill the NaN with 'NA'

# In[ ]:


# check if any record has MiscVal and not MiscFeature
missing_mf = data[(data.MiscFeature.isna()) & (data.MiscVal>0)]
missing_mf[['MiscFeature', 'MiscVal', 'SalePrice']]


# We have one record that has MiscVal but MiscFeature and SalePrice is missing. Since this is the only record, we can simply find out the MiscFeature that has MiscVal greater than 15000 and map that Feature to this record.

# In[ ]:


# Lets plot the MiscFeature Vs MiscVal box plot to find out the Val range of each Feature
sns.catplot(data=data, x='MiscFeature', y='MiscVal', kind='box')


# Its clear that Gar2 Feature has value greater than 15000. So we can map Gar2 for the record with missing MiscFeature which has MiscVal.

# In[ ]:


data.at[1089, 'MiscFeature'] = 'Gar2'


# In[ ]:


missing_mf = data[(data.MiscFeature.isna()) & (data.MiscVal>0)]
missing_mf[['MiscFeature', 'MiscVal', 'SalePrice']]


# We have replaced the NaN with 'Gar2' for 1089 record. Now we can replace all other NaN with 'NA'

# In[ ]:


data.MiscFeature.fillna('NA', inplace=True)


# ### Alley
# Discription says that Alley can be 'NA' for house that does not have Alley, so lets simply replace NaN with 'NA'

# In[ ]:


data.Alley.fillna('NA', inplace=True)


# ### Fence
# Discription says that Fence can be 'NA' for house that does not have Fence, so lets simply replace NaN with 'NA'.

# In[ ]:


data.Fence.fillna('NA', inplace=True)


# ### FireplaceQu
# This feature can be 'NA' for house that does not have any Fireplace, Lets corss check this with Fireplaces feature, If Fireplaces is > 0 then FireplaceQA cannot be NaN. If so, have to anaylse and fix it else replace NaN with 'NA'.

# In[ ]:


data[(data.Fireplaces > 0) & (data.FireplaceQu.isna())].shape


# In[ ]:


# Lets replace All NaN with 'NA' for FireplaceQu
data.FireplaceQu.fillna('NA', inplace=True)


# ### LotFrontage
# Linear feet of street connected to property. Lets replace the missing value with median of LotFrontage.

# In[ ]:


lotData =  data[['LotArea', 'LotFrontage', 'LotConfig']]
plt.figure(figsize=(2,3))
sns.boxplot(data=lotData, x='LotFrontage', orient='v')
lotData.LotFrontage.mean(), lotData.LotFrontage.median()


# From the plot we can see that 50% quartile falls around ~68. the same is verified by computing the mean and median. Lets replace the missing value with median.

# In[ ]:


data.LotFrontage.fillna(data.LotFrontage.median(), inplace=True)


# ### Garage
# In this Section lets analyse and understand the contribution of Garage features for SalePrice

# In[ ]:


grg_data = data[['YearBuilt','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond', 'MiscFeature', 'SalePrice']]
fn_gr = grg_data[(grg_data.GarageQual.isna()) & (grg_data.GarageType.notna())] # return the records with GarageType available and GarageQual NotAvailable
fn_gr


# In[ ]:


grg_features = ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']
data.at[fn_gr.index, grg_features] = 'NA'
# fill other NaN with 'NA'
data.at[data[grg_data.GarageType.isna()].index, grg_features] = 'NA'


# ### Basemet
# Following are the list of features which have missing values in them. Some features have lesser missing values when compared to other feature.
# 
# Lets compare the missing features amoung them to find out the records which have missing fields and fix them first.
# 
# Later we can replace all NaN with 'NA'
# 
# 1. BsmtExposure : 82
# 2. BsmtCond : 82
# 3. BsmtQual : 81
# 4. BsmtFinType2 : 80
# 5. BsmtFinType1 : 79
# 6. BsmtFinSF1 : 1
# 7. BsmtFinSF2 : 1
# 

# In[ ]:


bsmt_fts = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 
 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',]
int_fts = bsmt_fts + ['SalePrice']
bsmt_data = data[int_fts]


# In[ ]:


# BsmtExposure
Missing_BsmtExposure = bsmt_data[(bsmt_data.BsmtExposure.isna()) & (bsmt_data.BsmtFinType1.notna())]
Missing_BsmtExposure


# In[ ]:


# Search for similar records with BsmtExposure, find the category with max value and replace the missing value
BsmtExposure_samples = bsmt_data[(bsmt_data.BsmtQual=='Gd') & (bsmt_data.BsmtCond=='TA') & 
          (bsmt_data.BsmtFinType1=='Unf') & (bsmt_data.BsmtFinType2=='Unf') &
          (bsmt_data.BsmtFinSF1==0) & (bsmt_data.BsmtFinSF2==0) ]
sns.catplot(data=BsmtExposure_samples, x='BsmtExposure', kind='count')


# In[ ]:


# above plot shows that most of the other records of similar condition has No Exposure as value.
data.at[Missing_BsmtExposure.index, 'BsmtExposure'] = 'No'


# In[ ]:


# BsmtCond
Missing_BsmtCond = bsmt_data[(bsmt_data.BsmtCond.isna()) & (bsmt_data.BsmtExposure.notna())]
Missing_BsmtCond


# In[ ]:


Missing_BsmtCond_rc1=bsmt_data[(bsmt_data.BsmtQual=='Gd') & (bsmt_data.BsmtExposure=='Mn') & (bsmt_data.BsmtFinType1=='GLQ') & (bsmt_data.BsmtFinType2=='Rec')]
sns.catplot(data=Missing_BsmtCond_rc1, x='BsmtCond', kind='count')


# In[ ]:


Missing_BsmtCond_rc2=bsmt_data[(bsmt_data.BsmtQual=='TA') & (bsmt_data.BsmtExposure=='No') & (bsmt_data.BsmtFinType1=='BLQ') & (bsmt_data.BsmtFinType2=='Unf')]
sns.catplot(data=Missing_BsmtCond_rc2, x='BsmtCond', kind='count')


# In[ ]:


Missing_BsmtCond_rc3=bsmt_data[(bsmt_data.BsmtQual=='TA') & (bsmt_data.BsmtExposure=='Av') & (bsmt_data.BsmtFinType1=='ALQ') & (bsmt_data.BsmtFinType2=='Unf')]
sns.catplot(data=Missing_BsmtCond_rc3, x='BsmtCond', kind='count')


# In[ ]:


# Replace NaN with TA, as per similar records
data.at[Missing_BsmtCond.index, 'BsmtCond'] = 'TA'


# In[ ]:


Missing_BsmtQual = bsmt_data[(bsmt_data.BsmtQual.isna()) & (bsmt_data.BsmtCond.notna())]
Missing_BsmtQual


# In[ ]:


Missing_BsmtQual_rc1=bsmt_data[(bsmt_data.BsmtCond=='Fa') & (bsmt_data.BsmtExposure=='No') & (bsmt_data.BsmtFinType1=='Unf') & (bsmt_data.BsmtFinType2=='Unf')]
Missing_BsmtQual_rc2=bsmt_data[(bsmt_data.BsmtCond=='TA') & (bsmt_data.BsmtExposure=='No') & (bsmt_data.BsmtFinType1=='Unf') & (bsmt_data.BsmtFinType2=='Unf')]
sns.catplot(data=Missing_BsmtQual_rc1, x='BsmtQual', kind='count')
sns.catplot(data=Missing_BsmtQual_rc2, x='BsmtQual', kind='count')


# In[ ]:


data.at[Missing_BsmtQual.index, 'BsmtQual'] = 'TA'


# In[ ]:


Misssing_BsmtFinType2 = bsmt_data[(bsmt_data.BsmtFinType2.isna()) & (bsmt_data.BsmtCond.notna())]
Misssing_BsmtFinType2


# In[ ]:


Misssing_BsmtFinType2_rc1=bsmt_data[(bsmt_data.BsmtQual=='Gd') &(bsmt_data.BsmtCond=='TA') & 
              (bsmt_data.BsmtExposure=='No') & (bsmt_data.BsmtFinType1=='GLQ') 
              & (bsmt_data.BsmtUnfSF > 0) &(bsmt_data.BsmtFinSF2 > 0)&(bsmt_data.BsmtFinSF1 > 0)]
sns.catplot(data=Misssing_BsmtFinType2_rc1, x='BsmtFinType2', kind='count')


# In[ ]:


data.at[Misssing_BsmtFinType2.index, 'BsmtFinType2'] = 'ALQ'


# In[ ]:


Missing_BsmtFinSF1 = bsmt_data[bsmt_data.BsmtFinSF1.isna()]
Missing_BsmtFinSF1


# In[ ]:


Missing_BsmtFinSF2 = bsmt_data[bsmt_data.BsmtFinSF2.isna()]
Missing_BsmtFinSF2


# Both Missing BsmtFinSF1 & BsmtFinSF2 belong to same record, simply replace it with 0

# In[ ]:


Missing_BsmtFullBath = bsmt_data[bsmt_data.BsmtFullBath.isna()]
Missing_BsmtFullBath


# Record 728 is the one which is missing both BsmtFullBath & BsmtHalfBath, simply replace it with 0

# In[ ]:


data.at[660, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath']] = 0.0
data.at[728, ['BsmtFullBath', 'BsmtHalfBath']] = 0.0


# In[ ]:


# Replace other missing values with 'NA'
BsmtNA = bsmt_data[bsmt_data.BsmtQual.isna()]
data.at[BsmtNA.index, bsmt_fts] = 'NA'


# ### MasVnr
# 
# Following are the list of features with missing values.
# 
# 1. MasVnrType : 24
# 2. MasVnrArea : 23

# In[ ]:


masVnr_features = ['MasVnrType', 'MasVnrArea']
masVnr_intd_fets = masVnr_features + ['Exterior1st','Exterior2nd','Foundation', 'RoofMatl','SalePrice']
masVnr = data[masVnr_intd_fets]


# In[ ]:


missing_masVnr = masVnr[(masVnr.MasVnrType.isna()) & (masVnr.MasVnrArea.notna())]
missing_masVnr


# In[ ]:


# Lets get records similar to our **missing_masVnr** record
MasVnrType_samples = masVnr[(masVnr.Exterior1st=='Plywood') &  
                            (masVnr.MasVnrArea > 0) & (masVnr.Foundation=='CBlock') 
                            & (masVnr.RoofMatl=='CompShg') & masVnr.MasVnrArea.between(190,220)]
sns.catplot(data=MasVnrType_samples, x='MasVnrType', kind='count')


# Above plot shows that other similar records with MasVnrType avaialble has **BrkFace** as the only value. so lets replace our missing value with BrkFace

# In[ ]:


data.at[1150, 'MasVnrType'] = 'BrkFace'


# In[ ]:


# Lets replace other missing values with None
act_missing_masvnr = masVnr[masVnr.MasVnrType.isna()]
data.at[act_missing_masvnr.index, masVnr_features] = ['None', 0.0]


# ### MSZoning

# In[ ]:


ms_features = ['MSSubClass', 'MSZoning']
ms_intd_features = ms_features + ['Street','Neighborhood','BldgType','HouseStyle','YearBuilt','RoofStyle','BsmtQual','Foundation','SalePrice']
ms_data = data[ms_intd_features]


# In[ ]:


ms_data[ms_data.MSZoning.isna()]


# In[ ]:


ms_data[(ms_data.Street == 'Pave') & (ms_data.MSSubClass == 20) & (ms_data.HouseStyle.isin(['1Story', '2.5Unf'])) & (ms_data.Neighborhood=='Mitchel') 
      & (ms_data.BldgType=='1Fam')  & (ms_data.YearBuilt.between(1900, 1955, inclusive=True))]


# In[ ]:


ms_data[(ms_data.YearBuilt==1900) & (ms_data.BldgType=='1Fam') & (ms_data.Neighborhood=='IDOTRR')]


# Records similar to #455 shows that it belongs to C (all)
# 
# Records similar to #756 shows that it belongs to C (all)
# 
# Records similar to #790 shows that it belongs to RM
# 
# Records similar to #1444 shows that it belongs to RL

# In[ ]:


idx, val = [455, 756, 790, 1444], ['C (all)', 'C (all)', 'RM', 'RL']
for i, v in (zip(idx, val)):
  data.at[i, 'MSZoning'] = v


# ### Functional

# In[ ]:


func_features = ['Functional']
func_intd_features = func_features + ['Utilities','Street','Neighborhood','BldgType','HouseStyle','YearBuilt','RoofStyle','BsmtQual','Foundation','SalePrice']
func_data = data[func_intd_features]
missing_func = func_data[func_data.Functional.isna()]
missing_func


# In[ ]:


sns.catplot(data=func_data, x='Functional', kind='count')


# In[ ]:


# replace missing values with Typ
data.at[missing_func.index, 'Functional'] = 'Typ'


# ### Utilities

# In[ ]:


sns.catplot(data=data, x='Utilities', kind='count')


# In[ ]:


data[data.Utilities=='NoSeWa'].shape


# Only one record has NoSeWa and all other have AllPub as value for utilities feature, so lets replace the missing value with AllPub

# In[ ]:


data.at[data[data.Utilities.isna()].index,'Utilities'] = 'AllPub'


# ### SaleType

# In[ ]:


sns.catplot(data=data, x='SaleType', kind='count')


# In[ ]:


data.at[data[data.SaleType.isna()].index, 'SaleType'] = 'WD'


# ### Electrical

# In[ ]:


sns.catplot(data=data, x='Electrical', kind='count')


# In[ ]:


data.at[data[data.Electrical.isna()].index, 'Electrical'] = 'SBrkr'


# In[ ]:


kit_fe = ['KitchenAbvGr', 'KitchenQual']
miss_kit = data[data.KitchenQual.isna()]
miss_kit[kit_fe]


# In[ ]:


# find out the max kitchnQual for kitchenAbvGr == 1 and replace the NaN with that value
sns.catplot(data=data[data.KitchenAbvGr==1], x='KitchenQual', kind='count')


# In[ ]:


data.at[miss_kit.index, 'KitchenQual'] = 'TA'


# ### Exterior

# In[ ]:


ext_feat = ['Exterior1st', 'Exterior2nd']
ext_int_feat = ext_feat + ['ExterQual','MasVnrType','SalePrice']
ext_data = data[ext_int_feat]


# In[ ]:


data[ext_data.Exterior1st.isna()]


# In[ ]:


# Check other records with similar feature values and find out which Exterior is used most and replace the missing value
recs=data[(data.Neighborhood=='Edwards') & (data.BldgType=='1Fam') & 
          (data.HouseStyle=='1Story') & (data.MasVnrType=='None') &
           (data.Foundation=='PConc')]
sns.catplot(data=recs, x='Exterior1st', kind='count')
sns.catplot(data=recs, x='Exterior2nd', kind='count')


# In[ ]:


# replace the missing value with VinylSd
data.at[691, ['Exterior1st', 'Exterior2nd']] = 'VinylSd'


# In[ ]:


get_features_with_missing_data(data)


# All Missing values have been handled by now. Only field which has missing value is SalePrice which is expected as the belong test data.

# ## Save Data

# In[ ]:


cured_test_data = data[data.SalePrice.isna()]
cured_test_data.to_csv(cured_test_data_path, index=False)
cured_train_data = data[data.SalePrice.notna()]
cured_train_data.to_csv(cured_train_data_path, index=False)
f'Train Data: {cured_train_data.shape} | Test Data: {cured_test_data.shape}'

