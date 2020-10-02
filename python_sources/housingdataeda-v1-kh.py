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


# plotting and stats libraries
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

# report warnings
import warnings
warnings.filterwarnings('ignore')

# Encoders
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score,mean_squared_error
import re


# # Introduction
# In this text-based competition we are given 79 features about houses located in Ames, Iowa and expected to predict the prices of these houses. Our training set and testing set are relatively small with about 1458 rows each (assuming there are no redundancies).

# In[ ]:


# read in the training data
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# read in the testing data
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# # Exploratory Analysis (Pre-Cleaning)

# In[ ]:


#missing training data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.loc[missing_data["Total"]>0,:]


# In[ ]:


#missing testing data
total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.loc[missing_data["Total"]>0,:]


# So I noticed that the columns with the most null values are features where a null value actually contains information. For example, if null is found in PoolQC, it means that the house does not have a pool. This observation led me to believe I am going to have to fill-in the rows that have NULL variables that mean something (see cleaning below).

# For `LotFrontage` I am confused why some of these values would be null. Could it be possible that these homes do not have a street connected to their property. I need to visualize the distribution of values of Lot Frontage.

# In[ ]:


LotFrontage_ser = df_train.loc[:,"LotFrontage"] +     df_test.loc[:,"LotFrontage"]
notNullLotFrontage_ser = LotFrontage_ser.dropna()
notNullLotFrontage_ser.describe()


# In[ ]:


sns.distplot(notNullLotFrontage_ser, fit=stats.norm)
fig = plt.figure()


# Hmm so it looks like this features has a pretty normal distribution. I'm assuming that every property has a lot (since LotFrontage is always >0) so I'll just imput the mean.

# Homes with missing garage values most likely do not have garages. Therefore, I will impute 0 for most of these columns. Unfortunately setting the GarageYrBuilt to 0 doesn't make any sense so I'll just set these values to the mode. 

# Since there is only 1 row in the training set that has a null Electrical value, I just drop that row.

# In[ ]:


row2drop = df_train.loc[df_train["Electrical"].isnull(),:].index
df_train = df_train.drop(row2drop)


# 
# We have a lot more columns in the testing data with null variables. Since we need to predict these rows we cannot drop rows. Our options are to either drop the column or fill in the null values.
# 
# I decided to fill in these values since I believe I have a general idea of how to go about doing that and I don't want to lose any information yet.

# # Cleaning and Reorganizing the training data

# ## duplicate rows

# check for duplicate rows if there are any...

# In[ ]:


# print shape before removing duplicates
print(df_train.drop(["Id"], axis=1).shape)
# remove the duplicates and print shape
print(df_train.drop(["Id"], axis=1).drop_duplicates().shape)


# In[ ]:


# print shape before removing duplicates
print(df_test.drop(["Id"], axis=1).shape)
# remove the duplicates and print shape
print(df_test.drop(["Id"], axis=1).drop_duplicates().shape)


# turns out there were none...

# ## Deal with Categorical Variables

# Convert categorical-nominal variables into multiple columns using One Hot Encoding

# In[ ]:


# add one-hot encoded columns based on `col` in `df` to `df`
def one_hot_encode(df, col):
    df[col] = pd.Categorical(df[col])
    dfDummies = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df, dfDummies], axis=1)
    #df = df.drop([col],axis=1)
    return(df)


# In[ ]:


cat_ord_vars=["MSSubClass", "MSZoning", "Street", "Alley", "LandContour",              "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType",              "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",              "Foundation", "Electrical", "Heating", "GarageType", "MiscFeature",              "SaleType", "SaleCondition", "MoSold"]

for cat_ord_col in cat_ord_vars:
    df_train = one_hot_encode(df_train,cat_ord_col)
    df_test = one_hot_encode(df_test,cat_ord_col)


# convert categorical-ordinal variables to integers

# In[ ]:


cleanup_nums = {"LotShape":     {"Reg": 0, "IR1": 1, "IR2" : 2, "IR3" : 3},
                "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
               "LandSlope":{"Gtl":0,"Mod":1,"Sev":2},
               "HouseStyle":{"1Story":0.00, "1.5Unf":0.25, "1.5Fin":0.50,
                             "2Story":1.00, "2.5Unf":1.25, "2.5Fin": 1.50,
                            "SFoyer":1.50, "SLvl":2.0},
               "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
               "ExterCond": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
               "BsmtQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
               "BsmtCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
               "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
               "BsmtFinType1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5,
                               "GLQ":6},
               "BsmtFinType2": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5,
                               "GLQ":6},
               "HeatingQC": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
               "CentralAir": {"N": 0, "Y": 1},
               "KitchenQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4,
                              "Min2": 5, "Min1": 6, "Typ": 7},
                "FireplaceQu": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                 "GarageFinish": {"Unf": 1, "RFn": 2, "Fin": 3},
                "GarageQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "GarageCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                "PavedDrive": {"N": 0, "P": 1, "Y": 2},
                "PoolQC": {"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                "Fence": {"MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
               }


# In[ ]:


df_train = df_train.replace(cleanup_nums)
df_test = df_test.replace(cleanup_nums)


# ## Deal with Null Values that contain information content

# In[ ]:


df_train.columns[df_train.isnull().any()]


# In[ ]:


df_test.columns[df_test.isnull().any()]


# ### LotFrontage

# Add `HasLotFrontage` variable to keep track of whether house has a Front
# * Set `HasLotFrontage` to 0 if `LotFrontage` is Null.
# * Set `LotFrontage` to 0 if `LotFrontage` is Null
# * Change `HasLotFrontage` to 1 if `LotFrontage` > 0 

# In[ ]:


df_train.loc[df_train["LotFrontage"].isnull()==True,"HasLotFrontage"]=0
df_train.loc[df_train["LotFrontage"].isnull()==True,"LotFrontage"]=0
df_train.loc[df_train["LotFrontage"]>0,"HasLotFrontage"]=1

df_test.loc[df_test["LotFrontage"].isnull()==True,"HasLotFrontage"]=0
df_test.loc[df_test["LotFrontage"].isnull()==True,"LotFrontage"]=0
df_test.loc[df_test["LotFrontage"]>0,"HasLotFrontage"]=1


# ### FireplaceQu

# Set Fireplace Quality to 0 if there is no fireplace

# In[ ]:


df_train.loc[df_train["Fireplaces"]==0,"FireplaceQu"]=0
df_test.loc[df_test["Fireplaces"]==0,"FireplaceQu"]=0


# ### PoolQC

# Set Pool Quality to 0 if there is no pool

# In[ ]:


df_train.loc[df_train["PoolArea"]==0,"PoolQC"]=0
df_test.loc[df_test["PoolArea"]==0,"PoolQC"]=0


# There are 3 rows in the testing set which have null "PoolQC", but contain a pool.

# In[ ]:


print(df_train["PoolQC"].isnull().sum())
print(df_test["PoolQC"].isnull().sum())


# Therefore we take the average pool quality of pools that are around the same Area of the Pool (+/-1SD) and set the pool quality manually to whatever the average pool quality is of pools that are that size.

# In[ ]:


sd_of_pool_sizes = np.mean(df_train.loc[df_train["PoolArea"] > 0 , "PoolArea"].append(
    df_test.loc[df_test["PoolArea"] > 0 , "PoolArea"]))
print (sd_of_pool_sizes)
for idx, row in df_test.loc[df_test["PoolQC"].isnull(),:].iterrows():
    minPoolArea = row["PoolArea"] - sd_of_pool_sizes
    if minPoolArea < 0:
        minPoolArea = 0
    maxPoolArea = row["PoolArea"] + sd_of_pool_sizes
    df_train_poolqc_df = df_train.loc[(df_train["PoolArea"] > minPoolArea) &                                       (df_train["PoolArea"] < maxPoolArea),
                                      ["PoolQC"]]
    df_test_poolqc_df = df_test.loc[(df_test["PoolArea"] > minPoolArea) &                                       (df_test["PoolArea"] < maxPoolArea),
                                      ["PoolQC"]]
    meanPoolQC = np.round(df_train_poolqc_df.append(df_test_poolqc_df).mean(skipna=True),0)
    df_test.loc[idx,"PoolQC"] = meanPoolQC.values[0]
    #.mean(skipna=True)
    


# ### Functionality
# In the "data_description.txt" file it says to "Assume typical unless deductions are warranted". Therefore, we set the rows with null functionality to 7 which means its typical

# In[ ]:


df_train.loc[df_train["Functional"].isnull(),"Functional"]=7
df_test.loc[df_test["Functional"].isnull(),"Functional"]=7


# ### Fence

# Set Fence Quality to 0 if there is no fence

# In[ ]:


df_train.loc[df_train["Fence"].isnull(),"Fence"]=0
df_test.loc[df_test["Fence"].isnull(),"Fence"]=0


# ### MasVnrType and MasVnrArea

# Add `HasMasVnr` variable to keep track of whether house has a Masonry veneer area
# * Change `HasMasVnr` to 0 if `MasVnrArea` and `MasVnrType` are Null.
# * Change `MasVnrArea` to 0 if `MasVnrArea` is Null
# * Change `HasMasVnr` to 0 if `MasVnrArea` > 0 or `MasVnrType` is not Null.

# In[ ]:


df_train.loc[     df_train.loc[:,["MasVnrArea","MasVnrType"]].isnull().all(axis=1),     "HasMasVnr"]=0
df_train.loc[              df_train.loc[:,["MasVnrArea","MasVnrType"]].isnull().all(axis=1),
             "MasVnrArea"]=0
df_train.loc[(df_train["MasVnrArea"]>0) | (df_train["MasVnrType"].isnull()==False),"HasMasVnr"]=1

df_test.loc[     df_test.loc[:,["MasVnrArea","MasVnrType"]].isnull().all(axis=1),     "HasMasVnr"]=0
df_test.loc[              df_test.loc[:,["MasVnrArea","MasVnrType"]].isnull().all(axis=1),
             "MasVnrArea"]=0
df_test.loc[(df_test["MasVnrArea"]>0) | (df_test["MasVnrType"].isnull()==False),"HasMasVnr"]=1


# ### Garage Features

# Add `HasGarage` variable to keep track of whether house has a Garage
# * Set `HasGarage` to 0 if 
#     * `GarageArea` is 0
#     * `GarageCars` is 0
#     * the following are Null:
#         * GarageType
#         * GarageYrBlt
#         * GarageFinish
#         * GarageCars
#         * GarageQual
#         * GarageCond
# * Otherwise set `HasGarage` to 1 

# In[ ]:


garage_cat_features=["GarageType","GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"]
garage_features = garage_cat_features + ["GarageArea", "GarageCars"]
df_train.loc[     (df_train.loc[:,garage_cat_features].isnull().all(axis=1)) &              ((df_train["GarageArea"] == 0) |  (df_train["GarageArea"].isnull() == True)) &              ((df_train["GarageCars"] == 0) |  (df_train["GarageCars"].isnull() == True)),     "HasGarage"]=0
df_train.loc[     (df_train.loc[:,garage_cat_features].notnull().any(axis=1)) |              (df_train["GarageArea"] > 0) |              (df_train["GarageCars"] > 0),     "HasGarage"]=1

df_test.loc[     (df_test.loc[:,garage_cat_features].isnull().all(axis=1)) &              ((df_test["GarageArea"] == 0) |  (df_test["GarageArea"].isnull() == True)) &              ((df_test["GarageCars"] == 0) |  (df_test["GarageCars"].isnull() == True)),     "HasGarage"]=0
df_test.loc[     (df_test.loc[:,garage_cat_features].notnull().any(axis=1)) |              (df_test["GarageArea"] > 0) |              (df_test["GarageCars"] > 0),     "HasGarage"]=1


# In[ ]:


df_train.loc[df_train["HasGarage"]==0,"GarageFinish"]=0
df_train.loc[df_train["HasGarage"]==0,"GarageQual"]=0
df_train.loc[df_train["HasGarage"]==0,"GarageCond"]=0
df_train.loc[df_train["HasGarage"]==0,"GarageArea"]=0
df_train.loc[df_train["HasGarage"]==0,"GarageCars"]=0

df_test.loc[df_test["HasGarage"]==0,"GarageFinish"]=0
df_test.loc[df_test["HasGarage"]==0,"GarageQual"]=0
df_test.loc[df_test["HasGarage"]==0,"GarageCond"]=0
df_test.loc[df_test["HasGarage"]==0,"GarageArea"]=0
df_test.loc[df_test["HasGarage"]==0,"GarageCars"]=0


# If the there is no Garage, just set GarageYrBlt to the average year that Garages are built.

# In[ ]:


df_train.loc[df_train["GarageYrBlt"].isnull(),"GarageYrBlt"] = df_train["GarageYrBlt"].mode(dropna=True)
df_test.loc[df_test["GarageYrBlt"].isnull(),"GarageYrBlt"] = df_test["GarageYrBlt"].mode(dropna=True)


# unfortunately we are not done with the null features in the Garage Columns

# In[ ]:


for gcol in garage_features:
    print("Number of %s Nulls in Training + Testing Respectively:" % gcol)
    print(df_train[gcol].isnull().sum())
    print(df_test[gcol].isnull().sum())


# In[ ]:


garage_corrmat = df_train.loc[:,garage_features+["SalePrice"]].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(garage_corrmat, square=True);


# Since there are only 2 rows with null values in the GarageFinish,GarageQual, GarageCond, GarageArea, and GarageCars variables in the testing data I'll just set them to the average values since there are so few rows with nulls anyway.

# In[ ]:


df_test.loc[df_test.loc[:,["GarageFinish","GarageQual", "GarageCond", "GarageArea", "GarageCars"]].isnull().any(axis=1),             ["GarageFinish","GarageQual", "GarageCond", "GarageArea", "GarageCars"]]


# In[ ]:


for grow in ["GarageFinish","GarageQual", "GarageCond", "GarageArea", "GarageCars"]:
    mean_val = df_train[grow].append(df_test[grow]).mean(skipna=True)
    if (df_train[grow].dtypes == "int64"):
        mean_val = int(np.round(mean_val,0))
    df_test.loc[df_test[grow].isnull()==True,grow]=mean_val


# In[ ]:


df_test.iloc[[666,1116],:].loc[:,["GarageFinish","GarageQual", "GarageCond", "GarageArea", "GarageCars"]]


# There seem to be a lot of nulls in GarageType. GarageType is fine because I dealth with that by turning it into multiple columns with One Hot Coding.

# ### Barement Features: 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'

# Add `HasBasement` variable to keep track of whether house has a Basement
# * Set `HasBasement` to 0 if
#     * the following are Null:
#         * BsmtQual
#         * BsmtCond
#         * BsmtExposure
#         * BsmtFinType1
#         * BsmtFinType2
#     * BsmtFinSF1 is 0 or Null
#     * BsmtFinSF2 is 0 or Null
#     * BsmtUnfSF is 0 or Null
#     * TotalBsmtSF is 0 or Null
#     * BsmtFullBath is 0 or Null
#     * BsmtHalfBath is 0 or Null
# * Otherwise set `HasBasement` to 1 

# In[ ]:


basement_cat_features=["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
basement_features= basement_cat_features +     ["HasBasement","BsmtFinSF1","BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]

df_train.loc[     (df_train.loc[:,basement_cat_features].isnull().all(axis=1)) &              ((df_train["BsmtFinSF1"] == 0) | (df_train["BsmtFinSF1"].isnull() == True)) &              ((df_train["BsmtFinSF2"] == 0) | (df_train["BsmtFinSF2"].isnull() == True)) &              ((df_train["BsmtUnfSF"] == 0) | (df_train["BsmtUnfSF"].isnull() == True)) &              ((df_train["TotalBsmtSF"] == 0) | (df_train["TotalBsmtSF"].isnull() == True)) &              ((df_train["BsmtFullBath"] == 0) | (df_train["BsmtFullBath"].isnull() == True)) &              ((df_train["BsmtHalfBath"] == 0) | (df_train["BsmtHalfBath"].isnull() == True)),     "HasBasement"]=0
#df_train.loc[ \
#    (df_train.loc[:,basement_features].isnull().all(axis=1)), \
#    "HasBasement"]=0
df_train.loc[     (df_train.loc[:,basement_cat_features].notnull().any(axis=1)) |              (df_train["BsmtFinSF1"] > 0) |              (df_train["BsmtFinSF2"] > 0) |              (df_train["BsmtUnfSF"] > 0) |              (df_train["TotalBsmtSF"] > 0) |              (df_train["BsmtFullBath"] > 0) |              (df_train["BsmtHalfBath"] > 0),     "HasBasement"]=1

df_test.loc[     (df_test.loc[:,basement_cat_features].isnull().all(axis=1)) &              ((df_test["BsmtFinSF1"] == 0) | (df_test["BsmtFinSF1"].isnull() == True)) &              ((df_test["BsmtFinSF2"] == 0) | (df_test["BsmtFinSF2"].isnull() == True)) &              ((df_test["BsmtUnfSF"] == 0) | (df_test["BsmtUnfSF"].isnull() == True)) &              ((df_test["TotalBsmtSF"] == 0) | (df_test["TotalBsmtSF"].isnull() == True)) &              ((df_test["BsmtFullBath"] == 0) | (df_test["BsmtFullBath"].isnull() == True)) &              ((df_test["BsmtHalfBath"] == 0) | (df_test["BsmtHalfBath"].isnull() == True)),     "HasBasement"]=0
#df_test.loc[ \
#    (df_test.loc[:,basement_features].isnull().all(axis=1)), \
#    "HasBasement"]=0
df_test.loc[     (df_test.loc[:,basement_cat_features].notnull().any(axis=1)) |              (df_test["BsmtFinSF1"] > 0) |              (df_test["BsmtFinSF2"] > 0) |              (df_test["BsmtUnfSF"] > 0) |              (df_test["TotalBsmtSF"] > 0) |              (df_test["BsmtFullBath"] > 0) |              (df_test["BsmtHalfBath"] > 0),     "HasBasement"]=1


# Convert null variables which actually just mean that there is no basement.

# In[ ]:


df_train.loc[df_train["HasBasement"]==0,"BsmtQual"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtCond"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtExposure"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtFinType1"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtFinType2"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtFinSF1"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtFinSF2"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtUnfSF"]=0
df_train.loc[df_train["HasBasement"]==0,"TotalBsmtSF"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtFullBath"]=0
df_train.loc[df_train["HasBasement"]==0,"BsmtHalfBath"]=0

df_test.loc[df_test["HasBasement"]==0,"BsmtQual"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtCond"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtExposure"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtFinType1"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtFinType2"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtFinSF1"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtFinSF2"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtUnfSF"]=0
df_test.loc[df_test["HasBasement"]==0,"TotalBsmtSF"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtFullBath"]=0
df_test.loc[df_test["HasBasement"]==0,"BsmtHalfBath"]=0


# Unfortunately we still have not dealt with all of the null values for basement features.
# 

# In[ ]:


for bcol in basement_features:
    print("Number of %s Nulls in Training + Testing Respectively:" % bcol)
    print(df_train[bcol].isnull().sum())
    print(df_test[bcol].isnull().sum())


# There are:
# * 2 rows with null values for `BsmtQual` in the testing set 
# * 3 rows with null values for `BsmtCond` in the testing set 
# * 1 row with null values for `BsmtExposure` in the training set 
# * 2 rows with null values for `BsmtExposure` in the testing set
# * 1 row with null values for `BsmtFinType2` in the training set 
# 
# We can either use the average value to replace the null values (since isBasement is 1 we cannot set these to 0), or we can base them off other features that are not null.

# In[ ]:


basement_corrmat = df_train[basement_features].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(basement_corrmat, square=True);


# We see using the correlation plot that `BsmtCond` and `BsmtQual` are strongly correlated. I check to make sure that these two features are never both null when in our current dataframe.

# In[ ]:


print(df_train.loc[:,["BsmtQual","BsmtCond"]].isnull().all(axis=1).shape)


# Great! Since there are only 2 and 3 missing values in these columns respectively and their values have the same range I just set the null values to the value of the other.

# In[ ]:


df_train.loc[df_train["BsmtQual"].isnull(),"BsmtQual"]=df_train.loc[df_train["BsmtQual"].isnull(),"BsmtCond"]
df_test.loc[df_test["BsmtQual"].isnull(),"BsmtQual"]=df_test.loc[df_test["BsmtQual"].isnull(),"BsmtCond"]
df_train.loc[df_train["BsmtCond"].isnull(),"BsmtCond"]=df_train.loc[df_train["BsmtCond"].isnull(),"BsmtQual"]
df_test.loc[df_test["BsmtCond"].isnull(),"BsmtCond"]=df_test.loc[df_test["BsmtCond"].isnull(),"BsmtQual"]


# For basement exposure, I just set it to the average value based on both the training and test set where HasBasement is true.

# In[ ]:


avg_basement_exposure_given_basement = np.round(
    df_train.loc[df_train["HasBasement"]==1,"BsmtExposure"].append(
       df_test.loc[df_test["HasBasement"]==1,"BsmtExposure"]).mean(skipna=True),0)
print(avg_basement_exposure_given_basement)
df_train.loc[df_train["BsmtExposure"].isnull(),"BsmtExposure"] = avg_basement_exposure_given_basement
df_test.loc[df_test["BsmtExposure"].isnull(),"BsmtExposure"] = avg_basement_exposure_given_basement


# Since there is only 1 null row with `BsmtFinType2` and `BsmtFinType2` is highly correlated with `BsmtFinSF2`, I got the "BsmtFinSF2" value of the null row:

# In[ ]:


df_train.loc[df_train["BsmtFinType2"].isnull(),"BsmtFinSF2"]


# Then I found the standard deviation of the `BsmtFinSF2` in houses with basements

# In[ ]:


BsmtFinSF2_train_test_df =     df_train.loc[df_train["BsmtFinSF2"]>0,["BsmtFinSF2","BsmtFinType2"]].append(     df_test.loc[df_test["BsmtFinSF2"]>0,["BsmtFinSF2","BsmtFinType2"]])
np.std(BsmtFinSF2_train_test_df["BsmtFinSF2"])


# I took the average of "BsmtFinType2" values within 1 SD around the `BsmtFinSF2` value

# In[ ]:


min_BsmtFinSF2 = df_train.loc[df_train["BsmtFinType2"].isnull(),"BsmtFinSF2"] -     np.std(BsmtFinSF2_train_test_df["BsmtFinSF2"])
max_BsmtFinSF2 = df_train.loc[df_train["BsmtFinType2"].isnull(),"BsmtFinSF2"] +     np.std(BsmtFinSF2_train_test_df["BsmtFinSF2"])
subset_BsmtFinSF2_train_test_df = BsmtFinSF2_train_test_df.loc[     (BsmtFinSF2_train_test_df["BsmtFinSF2"] > min_BsmtFinSF2.values[0]) &     (BsmtFinSF2_train_test_df["BsmtFinSF2"] < max_BsmtFinSF2.values[0]),"BsmtFinType2"]

df_train.loc[df_train["BsmtFinType2"].isnull(),"BsmtFinType2"] =     np.round(subset_BsmtFinSF2_train_test_df.mean(skipna=True),0)


# ### KitchenQual
# 

# There is 1 row in the testing dataframe with no value for "KitchenQual"

# In[ ]:


print(df_train["KitchenQual"].isnull().sum())
print(df_test["KitchenQual"].isnull().sum())


# If there is no Kitchen Quality value then set it to the average kitchen quality value

# In[ ]:


df_train.loc[df_train["KitchenQual"].isnull(),"KitchenQual"] = np.round(df_train["KitchenQual"].append(df_test["KitchenQual"]).mean(skipna=True),0)
df_test.loc[df_test["KitchenQual"].isnull(),"KitchenQual"] = np.round(df_train["KitchenQual"].append(df_test["KitchenQual"]).mean(skipna=True),0)


# ### Utilities
# Only 2 rows in the testing set have null values for `Utilities`

# In[ ]:


print(df_train["Utilities"].isnull().sum())
print(df_test["Utilities"].isnull().sum())


# If there is no Utlities value then set it to the average kitchen quality value

# In[ ]:


df_train.loc[df_train["Utilities"].isnull(),"Utilities"] = np.round(df_train["Utilities"].append(df_test["Utilities"]).mean(skipna=True),0)
df_test.loc[df_test["Utilities"].isnull(),"Utilities"] = np.round(df_train["Utilities"].append(df_test["Utilities"]).mean(skipna=True),0)


# ### Drop Columns not found in Training or Testing Set

# In[ ]:


train_cols = list(df_train.columns.sort_values().unique())
test_cols = list(df_test.columns.sort_values().unique())
uniq_train_cols = [x for x in train_cols if (x not in test_cols and x != "SalePrice")]
uniq_test_cols = [x for x in test_cols if x not in train_cols]

df_train = df_train.drop(uniq_train_cols,axis=1)
df_test = df_test.drop(uniq_test_cols,axis=1)


# In[ ]:


# drop categorical ordinal columns that are not encoded
df_train_numerical = df_train.drop(cat_ord_vars, axis=1)
# drop categorical ordinal columns that are not encoded
df_test_numerical = df_test.drop(cat_ord_vars, axis=1)


# # Exploratory Analysis (Post-Cleaning)
# ## Target Variable (Sales Price)

# There are a lot of different ways to explore the variables so I am just going to start with the ones I think is important. Obviously, since SalePrice is the target variable it will be the first one I am interested in.

# The easiest way I know to summarize the variables is by using the `.describe()` function

# In[ ]:


df_train_numerical["SalePrice"].describe()


# Observations:
# * The minimum is price is not 0 which make sense since houses aren't free. If there were free houses we may question our target values and drop those rows for training purposes
# * The median is about $20,000 less than the mean which means that the distribution will have at least a slight positive skew. I am interested in checking the "normality" of the target variable since this will effect the statistical tests I can do down the line.

# In[ ]:


sns.distplot(df_train_numerical['SalePrice'], fit=stats.norm)
fig = plt.figure()
res = stats.probplot(df_train_numerical['SalePrice'], plot=plt)


# We see that the sales price:
# * Deviates from the normal distribution.
# * Has appreciable positive skewness.
# * Shows peakedness.

# Let's look at how much skew is in Sales Price. In case you forgot, here are the general rules for skewness:
# * If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
# * If the skewness is between -1 and -0.5(negatively skewed) or between 0.5 and 1(positively skewed), the data are moderately skewed.
# * If the skewness is less than -1(negatively skewed) or greater than 1(positively skewed), the data are highly skewed.

# In[ ]:


print("Skewness: %f" % df_train_numerical['SalePrice'].skew())


# It looks like the data is positively skewed

# Kurtosis is used to describe the extreme values in one tail versus the other.
# 
# High kurtosis means there are a lot of outliers. (>3)
# 
# Low kurtosis means there are not a lot of outliers. (<3)

# In[ ]:


print("Kurtosis: %f" % df_train_numerical['SalePrice'].kurt())


# Cool so our data is positively skewed and has lots of outliers, but what happens if we transform the Sales Price using the log transformation?

# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(np.log(df_train_numerical['SalePrice']), fit=stats.norm);
fig = plt.figure()
res = stats.probplot(df_train_numerical['SalePrice'], plot=plt)

print("Skewness: %f" % np.log(df_train_numerical['SalePrice']).skew())
print("Kurtosis: %f" % np.log(df_train_numerical['SalePrice']).kurt())


# So much better! The data points are now fairly symmetrical and there isn't as many outliers on one particular tail.

# In[ ]:


#applying log transformation
#df_train_numerical['logSalePrice'] = np.log(df_train_numerical['SalePrice'])


# In[ ]:


#dropping log transformation
#df_train_numerical = df_train_numerical.drop('logSalePrice', axis=1)


# # Exploring Dependent Features
# Now we have so many features but which ones are the most important? First test is to compare correlation between the features and the target variable `SalePrice`

# In[ ]:


#correlation matrix
corrmat = df_train_numerical.corr()
corrSalePrice = corrmat.loc[abs(corrmat["SalePrice"]) > .5, "SalePrice"]
corrSalePrice = corrSalePrice.sort_values(ascending=False)
#corrSalePrice
ax = corrSalePrice.plot.barh()


# There are 15 features with correlation values about 0.5. The next question is how much they are correlated to one another?

# In[ ]:


# how much are they correlated to each other?
corr2SalePrice = list(corrSalePrice.index)

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat.loc[corr2SalePrice,corr2SalePrice], square=True);


# It seems like correlated variables include:
# * total Basement square feet (TotalBsmtSF) and total First Floor Square Ft (1stFlrSF) 
# * Garage Area (GarageArea) and total Total Cars in Garage (Garage Cars)
# * Total rooms above grade (TotRmsAbvGrd) and square feet of Above grade (ground) living area (GrLivArea)
# 
# 
# I am interested in looking at these variables more specifically...

# Let's also check these out for categorical variables as well.
# 
# To this end, we evaluate the correlation with mutual information.
# 
# Since we are interested in the correlation between 
# 1. categorical features
# 2. categorical fearues and numerical features
# We visualize both here.

# In[ ]:


df_cat_ord_imputed = pd.DataFrame(columns=cat_ord_vars,data = np.where(df_train[cat_ord_vars].isnull(),'None',df_train[cat_ord_vars]))


# In[ ]:


mat = np.zeros((len(df_cat_ord_imputed.columns),len(df_cat_ord_imputed.columns)))
for i,col_i in enumerate(df_cat_ord_imputed.columns):
    for j,col_j in enumerate(df_cat_ord_imputed.columns):
        mat[i,j] = normalized_mutual_info_score(df_cat_ord_imputed[col_i],df_cat_ord_imputed[col_j])


# In[ ]:


corr_mat = pd.DataFrame(mat,columns=df_cat_ord_imputed.columns,index=df_cat_ord_imputed.columns)
sns.heatmap(corr_mat)


# Lools like some features are somewhat correlated within categorical features, such as
# * MSSubClass & BldgType
# * Exterior 1st & Exterior 2nd
# * SaleType and SaleCondition
# 
# Next, let's see how categorical features and numerical features are correlated. 
# One of the downside is this mutual information calculation is designed to work for discrete
# values. To this end, we "discretize" numerical features into some bins (let's say min(10,# of unique elements).

# In[ ]:


df_train_numerical_binned = df_train_numerical.copy()
for col in df_train_numerical_binned.columns:
    df_train_numerical_binned[col] = pd.cut(df_train_numerical_binned[col],min(10,df_train_numerical_binned[col].nunique()),labels = False)


# In[ ]:


cat_num_mat = np.zeros((len(df_cat_ord_imputed.columns),len(df_train_numerical_binned.columns)))
for col_i in df_cat_ord_imputed.columns:
    for col_j in df_train_numerical_binned.columns:
        cat_num_mat[i,j] = normalized_mutual_info_score(df_cat_ord_imputed[col_i],df_train_numerical_binned[col_j])


# In[ ]:


corr_mat_cat_num = pd.DataFrame(cat_num_mat,columns=df_train_numerical_binned.columns,index=df_cat_ord_imputed.columns)
sns.heatmap(corr_mat_cat_num)


# Looks like they are mostly uncorrelated ?
# Next, let's checkout the relationship of specific features and output variable one by one. 

# ## OverallQual

# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# looks like a strong linear relationship!

# ## GrLivArea

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# A pretty good linear relationship!

# # ExterQual

# In[ ]:


#box plot overallqual/saleprice
var = 'ExterQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# Looks pretty linear to me!

# # KitchenQual

# In[ ]:


#box plot overallqual/saleprice
var = 'KitchenQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# Looks pretty linear to me!

# # Garage Cars

# In[ ]:


#box plot overallqual/saleprice
var = 'GarageCars'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# I feel like I could merge 3 and 4 into a single category as "Above 2"...

# In[ ]:


garageCarsReplaceDict = {"GarageCars":{4:3}}

df_train = df_train.replace(garageCarsReplaceDict)
df_test = df_test.replace(garageCarsReplaceDict)


# In[ ]:


#box plot overallqual/saleprice
var = 'GarageCars'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# Yep! Looks better now!

# # Garage Area

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GarageArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# This looks rough. Lets log transform it... It still look questionable...

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GarageArea'
data = pd.concat([df_train['SalePrice'], np.log(df_train[var])], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GarageArea'
data = pd.concat([df_train['SalePrice'], df_train[var] * np.log(df_train[var])], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# I don't think I should use this variable...

# ## TotalBsmtSF

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# This relationship isn't super clear. I wonder what would happen if we log transform

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], np.log(df_train[var])], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], np.log(df_train[var]) * df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# I feel like that may have helped! The relation looks more exponential than linear though.

# In[ ]:


df_train['TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'] + 1e-8) * df_train['TotalBsmtSF']
df_train_numerical['TotalBsmtSF'] = np.log(df_train_numerical['TotalBsmtSF'] + 1e-8) * df_train_numerical['TotalBsmtSF']


# ## FullBath

# In[ ]:


#box plot overallqual/saleprice
var = 'FullBath'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# Oh Damn. I didn't realize you could have 0 FullBath. You need a bath in a house right? Does this mean we are missing information. Probably. Oy vey... Well at least its linear when we have the information... Maybe I should set the 0's to 1's?

# In[ ]:


FullBathReplaceDict = {"FullBath":{0:1}}

df_train = df_train.replace(FullBathReplaceDict)
df_test = df_test.replace(FullBathReplaceDict)


# In[ ]:


#box plot overallqual/saleprice
var = 'FullBath'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# Yeah, I think that makes more sense.

# ## GarageFinish

# In[ ]:


#box plot overallqual/saleprice
var = 'GarageFinish'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# ## FireplaceQu

# In[ ]:


var = 'FireplaceQu'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
plt.xticks(rotation=90);


# Interesting... This does seem to be generally linear
# 
# What if log/xlogx/square transform ?

# In[ ]:


var = 'FireplaceQu'
data = pd.concat([df_train['SalePrice'], df_train[var] ** 2], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
plt.xticks(rotation=90);


# In[ ]:


var = 'FireplaceQu'
data = pd.concat([df_train['SalePrice'], np.log(df_train[var]) * df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
plt.xticks(rotation=90);


# ## YearRemodAdd

# In[ ]:


var = 'YearRemodAdd'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


#  I'd say that 'SalePrice' is more prone to spend more money if it was recently renovated.

# # Feature Engineering
# To perform feature engineering we may want to remove some of the One Hot Encoding columns so we would use `df_train` which has the original variables in them.
# 
# Features I want to engineer by spliting them into multiple features:
# * `MSSubClass`: "Identifies the type of dwelling involved in the sale"
# * `SaleType`: "Type of sale"

# # Modeling
# Our target value, 'SalesPrice' is continuous, therefore this challenge is a regression problem.
# 
# ## Linear Regression

# ### Feature Selection
# 

# Below, I will try a couple of different methods for Feature Selection.
# 
# First, I am only going to use the variables that I belive to show strong LINEAR correlation with Saleprice.
# 
# To evaluate the model performance, I set up validation set separetely.

# In[ ]:


fav_variables=['OverallQual','YearRemodAdd','FireplaceQu',                'GarageFinish', 'FullBath', 'GarageCars',                'KitchenQual', 'ExterQual', 'GrLivArea',               'OverallQual']


# In[ ]:


df_train_tr,df_train_val = train_test_split(df_train,test_size = .2)


# In[ ]:


df_train_tr = pd.DataFrame(data = df_train_tr, columns = df_train.columns)
df_train_val = pd.DataFrame(data = df_train_val,columns = df_train.columns)


# In[ ]:


from sklearn.linear_model import LinearRegression
X_train = df_train_tr.loc[:,fav_variables].copy()
y_train = df_train_tr.loc[:,"SalePrice"].copy()
y_val = df_train_val.loc[:,'SalePrice'].copy()
X_pred_val = df_train_val.loc[:,fav_variables].copy()
X_pred = df_test_numerical.loc[:,fav_variables].copy()
reg = LinearRegression().fit(X_train, y_train)
pred_val = reg.predict(X_pred_val)
pred = reg.predict(X_pred)
#pred=np.power(10,pred)


# In[ ]:


print('mean squared error is %f' %mean_squared_error(y_val,pred_val))


# In[ ]:


df_test.loc[:,"SalePrice"]=pd.Series(pred)
sub_df = df_test.loc[:,["Id","SalePrice"]]
sub_df.to_csv("submission.csv", index=False)


# In[ ]:


sub_df.head()


# Let's see how it works with other feature selection methods.
# While focusing on features that has high correlation with Sale Price,
# I would pick up features with small correlations to each other.
# 
# To this end, let's look at the feature correlation map again.

# In[ ]:


corrmat = df_train_numerical.corr()
corrSalePrice = corrmat.loc[abs(corrmat["SalePrice"]) >= .5, "SalePrice"]
corrSalePrice = corrSalePrice.sort_values(ascending=False)
corr2SalePrice = list(corrSalePrice.index)

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat.loc[corr2SalePrice,corr2SalePrice], square=True);


# Ok, so it looks like there are some candidates to drop from this heatmap.
# * One of GarageCars / Garage Area
# * One of 1stFlrSF / TotalBsmtSF
# * One of GrLivArea / TotRmsAbvGrd
# * One of ExterQual / Kitchen Qual
# 
# Let's see how it goes with these features dropped.

# In[ ]:


indep_variables=['OverallQual','YearRemodAdd','GarageCars','FireplaceQu',                'GarageFinish', 'FullBath', 'GarageCars',                'KitchenQual', '1stFlrSF', 'GrLivArea',               'OverallQual']


# In[ ]:


from sklearn.linear_model import LinearRegression,Lasso
X_train = df_train_tr.loc[:,indep_variables].copy()
y_train = df_train_tr.loc[:,"SalePrice"].copy()
y_val = df_train_val.loc[:,'SalePrice'].copy()
X_pred_val = df_train_val.loc[:,indep_variables].copy()
X_pred = df_test.loc[:,indep_variables].copy()
reg = LinearRegression().fit(X_train, y_train)
pred_val = reg.predict(X_pred_val)
pred = reg.predict(X_pred)
#pred=np.power(10,pred)


# In[ ]:


print('mean squared error is %f' %mean_squared_error(y_val,pred_val))


# Hmmm, looks slightly better.
# How about letting the model choose the features?
# 
# To this end, I apply Lasso regression.

# In[ ]:


from sklearn.linear_model import LinearRegression,Lasso,lasso_path
X_train = df_train_tr.loc[:,corr2SalePrice].drop(columns = 'SalePrice').copy()
y_train = df_train_tr.loc[:,"SalePrice"].copy()
y_val = df_train_val.loc[:,'SalePrice'].copy()
X_pred_val = df_train_val.loc[:,corr2SalePrice].drop(columns = 'SalePrice').copy()
X_pred = df_test.loc[:,corr2SalePrice].drop(columns = 'SalePrice').copy()
reg = Lasso(normalize=True).fit(X_train, y_train)
pred_val = reg.predict(X_pred_val)
pred = reg.predict(X_pred)
#pred=np.power(10,pred)


# In[ ]:


print('mean squared error is %f' %mean_squared_error(y_val,pred_val))


# In[ ]:


for i,coef in enumerate(reg.coef_):
    print('coef of ' + X_train.columns[i] + ' is %f' % coef)


# Looks like TotalBsmtSF and YearRemodAdd is supressed.

# Finally, let's see how it goes if we apply PCA to the data.
# The interpretability would be totally lost, but the features are guratenteed to be independent.
# In this case, let's halve # of features.

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(8)
X_train = pca.fit_transform(df_train_tr.loc[:,corr2SalePrice].drop(columns = 'SalePrice'))
X_pred_val = pca.fit_transform(df_train_val.loc[:,corr2SalePrice].drop(columns = 'SalePrice'))
X_pred = pca.fit_transform(df_test.loc[:,corr2SalePrice].drop(columns = 'SalePrice'))
reg = LinearRegression().fit(X_train, y_train)
pred_val = reg.predict(X_pred_val)
pred = reg.predict(X_pred)


# In[ ]:


print('mean squared error is %f' %mean_squared_error(y_val,pred_val))


# In[ ]:


df_test.loc[:,"SalePrice"]=pd.Series(pred)
sub_df = df_test.loc[:,["Id","SalePrice"]]
sub_df.to_csv("submission.csv", index=False)

