#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Machine Learning Libraries

# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import xgboost as xgb


# ignore future warnings...

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# # Introduction
# This kernel is a continuation of my EDA notebook: [https://www.kaggle.com/sklasfeld/housingdataeda-v1]
#     
# In this kernel I look to test different models using the data I analyzed in the previous notebook.
# 
# For reference I also used the follwing kaggle kernels:
# * https://www.kaggle.com/apapiu/regularized-linear-models
# * https://www.kaggle.com/humananalog/xgboost-lasso

# Let's open the training and testing data

# In[ ]:


# read in the training data
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# read in the testing data
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# # Modify Datasets
# In this section we will
# * perform feature engineering (create new features)
# * transform features
# * replace values of features
# * drop columns
# 
# in no particular order

# ## Transform SalePrice (Feature Transformation)

# In[ ]:


df_train['SalePrice'] = np.log(df_train['SalePrice'])


# ## Exterior variables (Feature Engineering)

# In[ ]:


# before I start, I know there this is a typo in the Exterior2nd column

# Cement Board is labeled as CmentBd instead of CemntBd. Therefore we must
# update it.
df_train = df_train.replace({"Exterior2nd":{"CmentBd":"CemntBd"}})
df_test = df_test.replace({"Exterior2nd":{"CmentBd":"CemntBd"}})

# Cement Board is labeled as Wd Shng instead of Wd Sdng. Therefore we must
# update it.
df_train = df_train.replace({"Exterior2nd":{"Wd Shng":"Wd Sdng"}})
df_test = df_test.replace({"Exterior2nd":{"Wd Shng":"Wd Sdng"}})

combined = pd.concat([df_train, df_test], sort=False)


#  I want to one-hot code these 2 columns by treating them as 1 so that we can check if the exterior has a specific type of covering. For example, `Exterior_CBlock` would be 1 if there is a Cinder Block covering on the house and 0 would be if `Exterior1st` and `Exterior2nd` are both not equal to "CBlock".

# In[ ]:


combined = pd.concat([df_train, df_test], sort=False)

exterior1_uniqVals = list(combined["Exterior1st"].dropna().unique())
exterior2_uniqVals = list(combined["Exterior2nd"].dropna().unique())
exterior_uniqVals = [x for x in exterior2_uniqVals if x not in exterior1_uniqVals]
exterior_uniqVals = exterior_uniqVals +exterior1_uniqVals

exterior_cols = []
for exterior_type in exterior_uniqVals:
    new_exType_col = "Exterior_"+exterior_type
    exterior_cols.append(new_exType_col)
    df_train.loc[((df_train["Exterior1st"]==exterior_type) |
                  (df_train["Exterior2nd"]==exterior_type)),new_exType_col] = 1
    df_train.loc[((df_train["Exterior1st"]!=exterior_type) &
                  (df_train["Exterior2nd"]!=exterior_type)),new_exType_col] = 0
    df_test.loc[((df_test["Exterior1st"]==exterior_type) |
                  (df_test["Exterior2nd"]==exterior_type)),new_exType_col] = 1
    df_test.loc[((df_test["Exterior1st"]!=exterior_type) &
                  (df_test["Exterior2nd"]!=exterior_type)),new_exType_col] = 0


# In addition, I create a feature that checks whether the exterior is made of 1 or 2 materials.

# In[ ]:


df_train.loc[df_train["Exterior1st"] != df_train["Exterior2nd"],"TwoExteriorMaterials"]=1
df_train.loc[df_train["Exterior1st"] == df_train["Exterior2nd"],"TwoExteriorMaterials"]=0


# Given our new features, we can drop "Exterior1st" and "Exterior2nd"

# In[ ]:


df_train = df_train.drop(["Exterior1st","Exterior2nd"],axis=1)
df_test = df_test.drop(["Exterior1st","Exterior2nd"],axis=1)


# ## Condition (Feature Engineering)

# In[ ]:


combined = pd.concat([df_train, df_test], sort=False)
condition1_uniqVals = list(combined["Condition1"].dropna().unique())
condition2_uniqVals = list(combined["Condition2"].dropna().unique())
condition_uniqVals = [x for x in condition2_uniqVals if x not in condition1_uniqVals]
condition_uniqVals = condition_uniqVals + condition1_uniqVals

condition_cols=[]
for condition_type in condition_uniqVals:
    new_condType_col = "Condition_"+condition_type
    condition_cols.append(new_condType_col)
    df_train.loc[((df_train["Condition1"]==condition_type) |
                  (df_train["Condition2"]==condition_type)),new_condType_col] = 1
    df_train.loc[df_train[new_condType_col].isnull(),new_condType_col] = 0
    df_test.loc[((df_test["Condition1"]==condition_type) |
                  (df_test["Condition2"]==condition_type)),new_condType_col] = 1
    df_test.loc[df_test[new_condType_col].isnull(),new_condType_col] = 0


# In[ ]:


df_train = df_train.drop(["Condition1","Condition2"],axis=1)
df_test = df_test.drop(["Condition1","Condition2"],axis=1)


# ## MSSubClass (Feature Engineering)

# Column for houses that have a 1946 and Newer Style

# In[ ]:


df_train.loc[((df_train["MSSubClass"]==20) | 
             (df_train["MSSubClass"]==60) |
             (df_train["MSSubClass"]==120) |
             (df_train["MSSubClass"]==160)),"MSSubClass_1946nNewer"] = 1

df_train.loc[df_train["MSSubClass_1946nNewer"].isnull()==True,"MSSubClass_1946nNewer"]=0

df_test.loc[((df_test["MSSubClass"]==20) | 
             (df_test["MSSubClass"]==60) |
             (df_test["MSSubClass"]==120) |
             (df_test["MSSubClass"]==160)),"MSSubClass_1946nNewer"] = 1

df_test.loc[df_test["MSSubClass_1946nNewer"].isnull()==True,"MSSubClass_1946nNewer"]=0


# Column for houses that share walls.

# In[ ]:


df_train.loc[((df_train["MSSubClass"]==90) | 
             (df_train["MSSubClass"]==190)),"ShareFloor"] = 1

df_train.loc[df_train["ShareFloor"].isnull()==True,"ShareFloor"]=0


# In[ ]:


df_test.loc[((df_test["MSSubClass"]==90) | 
             (df_test["MSSubClass"]==190)),"ShareFloor"] = 1

df_test.loc[df_test["ShareFloor"].isnull()==True,"ShareFloor"]=0


# In[ ]:


df_train = df_train.drop("MSSubClass",axis=1)
df_test = df_test.drop("MSSubClass",axis=1)


# ## HouseStyle (Feature Engineering)

# Column for homes with 2 stories or not.

# In[ ]:


df_train.loc[((df_train["HouseStyle"]=='1Story')  |
              (df_train["HouseStyle"]=='1.5Unf') |
              (df_train["HouseStyle"]=='1.5Fin') |
              (df_train["HouseStyle"]=='SFoyer') |
              (df_train["HouseStyle"]=='SLvl')),"2Stories"] = 0.0

df_train.loc[((df_train["HouseStyle"]=='2Story')  |
              (df_train["HouseStyle"]=='2.5Unf') |
             (df_train["HouseStyle"]=='2.5Fin')),"2Stories"] = 1

df_test.loc[((df_test["HouseStyle"]=='1Story')  |
               (df_test["HouseStyle"]=='1.5Unf') |
             (df_test["HouseStyle"]=='1.5Fin') |
              (df_test["HouseStyle"]=='SFoyer') |
               (df_test["HouseStyle"]=='SLvl')),"2Stories"] = 0.0

df_test.loc[((df_test["HouseStyle"]=='2Story')  |
              (df_test["HouseStyle"]=='2.5Unf') |
             (df_test["HouseStyle"]=='2.5Fin')),"2Stories"] = 1


# Column for homes with unfurnished floors.

# In[ ]:


df_test.loc[((df_test["HouseStyle"]=='1Story')  |
               (df_test["HouseStyle"]=='2Story') |
             (df_test["HouseStyle"]=='1.5Fin') |
              (df_test["HouseStyle"]=='SFoyer') |
               (df_test["HouseStyle"]=='SLvl') |
               (df_test["HouseStyle"]=='2.5Fin')),"Unfurnished"] = 0.0

df_test.loc[((df_test["HouseStyle"]=='1.5Unf')  |
              (df_test["HouseStyle"]=='2.5Unf')),"Unfurnished"] = 1


# In[ ]:


df_train = df_train.drop("HouseStyle",axis=1)
df_test = df_test.drop("HouseStyle",axis=1)


# ## Fix Weird Values (MasVnrArea)
# We expect that if the "MasVnrArea" is 0 that the "MasVnrType" is none. If the "MasVnrType" is not none then we should recalculate the "MasVnrArea" since it cannot be 0.  I want to reset these values to the average "MasVnrArea".

# In[ ]:


combined = pd.concat([df_train, df_test], sort=False)
fill_masvnrarea = (np.mean(np.sqrt(combined.loc[combined["MasVnrArea"]>0,
    "MasVnrArea"].dropna().values)))**2

df_train.loc[(df_train["MasVnrArea"]==0) &              (df_train["MasVnrType"]!="None"),"MasVnrArea"] = fill_masvnrarea

df_test.loc[(df_test["MasVnrArea"]==0) &              (df_test["MasVnrType"]!="None"),"MasVnrArea"] = fill_masvnrarea

df_train.loc[(df_train["MasVnrArea"]==0) &              (df_train["MasVnrType"].isnull()),"MasVnrArea"] = fill_masvnrarea

df_test.loc[(df_test["MasVnrArea"]==0) &              (df_test["MasVnrType"].isnull()),"MasVnrArea"] = fill_masvnrarea


# ## NeighborhoodTier (Feature Engineering)
# I want to create a columns that tells what type of "Tier" neigborhood the house is located in. 
# * The top Tier will have an average Sale price above or equal to the 75% percentile of the SalePrices. 
# * The middle upper Tier will have an average Sale price between the 50th and 75th percentile. 
# * The middle low Tier will have an average Sale price between the 25th and 50th percentile. 
# * The bottom tier will have an average Sale price below the 25th percentile.

# In[ ]:


neighboorhoodBySalePrice_df = df_train.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)

top_tier_n = list(neighboorhoodBySalePrice_df[neighboorhoodBySalePrice_df.values >= df_train['SalePrice'].quantile(.75)].index)
mid_upper_tier_n = list(neighboorhoodBySalePrice_df[(neighboorhoodBySalePrice_df.values < df_train['SalePrice'].quantile(.75)) &                                               (neighboorhoodBySalePrice_df.values >= df_train['SalePrice'].quantile(.50))].index)
mid_low_tier_n = list(neighboorhoodBySalePrice_df[(neighboorhoodBySalePrice_df.values < df_train['SalePrice'].quantile(.50)) &                                               (neighboorhoodBySalePrice_df.values >= df_train['SalePrice'].quantile(.25))].index)
lowest_tier_n = list(neighboorhoodBySalePrice_df[neighboorhoodBySalePrice_df.values < df_train['SalePrice'].quantile(.25)].index)
print(top_tier_n)
print(mid_upper_tier_n)
print(mid_low_tier_n)
print(lowest_tier_n)

df_train.loc[df_train['Neighborhood'].isin(top_tier_n), "neighborhoodTier"]=3
df_train.loc[df_train['Neighborhood'].isin(mid_upper_tier_n), "neighborhoodTier"]=2
df_train.loc[df_train['Neighborhood'].isin(mid_low_tier_n), "neighborhoodTier"]=1
df_train.loc[df_train['Neighborhood'].isin(lowest_tier_n), "neighborhoodTier"]=0
df_test.loc[df_test['Neighborhood'].isin(top_tier_n), "neighborhoodTier"]=3
df_test.loc[df_test['Neighborhood'].isin(mid_upper_tier_n), "neighborhoodTier"]=2
df_test.loc[df_test['Neighborhood'].isin(mid_low_tier_n), "neighborhoodTier"]=1
df_test.loc[df_test['Neighborhood'].isin(lowest_tier_n), "neighborhoodTier"]=0


# ## Transform Categorical Ordinal Variables

# convert categorical-ordinal variables to integers

# In[ ]:


cleanup_nums = {"Alley":     {"Grvl": 1, "Pave": 2},
                "LotShape":     {"Reg": 0, "IR1": 1, "IR2" : 2, "IR3" : 3},
               "LandSlope":{"Gtl":0,"Mod":1,"Sev":2},
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


# ## Set up Categorical Ordinal Variables

# In[ ]:


cat_ord_vars=[ "MSZoning", "Street", "LandContour",              "LotConfig", "Neighborhood", "BldgType",              "RoofStyle", "RoofMatl", "MasVnrType",              "Foundation", "Electrical", "Heating", "GarageType", "MiscFeature",              "SaleType", "SaleCondition", "MoSold"]


# ## Drop Utilities Column
# the value for Utilities is almost always the same 2916/2917 times. Let's drop this column.

# In[ ]:


df_train = df_train.drop(['Utilities'] , axis=1)
df_test = df_test.drop(['Utilities'] , axis=1)


# ## Impute Null Values

# ### Alley
# If row is missing the `Alley` variable then there is no Alley Access so it should be set to 0.

# In[ ]:


df_train.loc[df_train["Alley"].isnull()==True,"Alley"]=0
df_test.loc[df_test["Alley"].isnull()==True,"Alley"]=0


# ### Electrical
# Over 90% of the rows that contain `Electrical` values have the same value. Therefore, if row is missing the electrical variable then set it to the mode value of Electrical.

# In[ ]:


all_electrical_series = df_train["Electrical"].append(df_test["Electrical"])
mostCommonElectricalValue=all_electrical_series.mode().values[0]
print(mostCommonElectricalValue)

df_train.loc[df_train["Electrical"].isnull()==True,"Electrical"]=mostCommonElectricalValue
df_test.loc[df_test["Electrical"].isnull()==True,"Electrical"]=mostCommonElectricalValue


# 
# ### Lot frontage
# I'm assuming that every property has a lot (since LotFrontage is always >0) so I'll just imput the median based on the neighborhood of the house.

# In[ ]:


combined = pd.concat([df_train, df_test], sort=False)
for nieghborhood in list(combined["Neighborhood"].unique()):
    df_train.loc[(df_train["LotFrontage"].isnull()==True) &         (df_train["Neighborhood"]==nieghborhood),         "LotFrontage"] =         combined["LotFrontage"].groupby(combined["Neighborhood"]).median()[nieghborhood]
    df_test.loc[(df_test["LotFrontage"].isnull()==True) &         (df_test["Neighborhood"]==nieghborhood),         "LotFrontage"] =         combined["LotFrontage"].groupby(combined["Neighborhood"]).median()[nieghborhood]


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


# We take the average pool quality of pools that are around the same Area of the Pool (+/-1SD) and set the pool quality manually to whatever the average pool quality is of pools that are that size.

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
    df_test_poolqc_df = df_test.loc[(df_test["PoolArea"] > minPoolArea) &                                         (df_test["PoolArea"] < maxPoolArea),
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
# * Change `MasVnrArea` to 0 if `MasVnrArea` is Null and `MasVnrType` are Null.
# * Change `HasMasVnr` to 0 if `MasVnrArea` == 0 and `MasVnrType` are Null.
# * Change `HasMasVnr` to 1 if `MasVnrArea` > 0 or `MasVnrType` is not Null.

# In[ ]:


# Change `HasMasVnr` to 0 if `MasVnrArea` and `MasVnrType` are Null.
df_train.loc[              df_train.loc[:,["MasVnrArea","MasVnrType"]].isnull().all(axis=1),              "HasMasVnr"]=0
df_test.loc[             df_test.loc[:,["MasVnrArea","MasVnrType"]].isnull().all(axis=1),             "HasMasVnr"]=0

# Change `MasVnrArea` to 0 if `MasVnrArea` is Null and `MasVnrType` are Null.
df_train.loc[              df_train.loc[:,["MasVnrArea","MasVnrType"]].isnull().all(axis=1),
             "MasVnrArea"]=0
df_test.loc[             df_test.loc[:,["MasVnrArea","MasVnrType"]].isnull().all(axis=1),
            "MasVnrArea"]=0

# Change `HasMasVnr` to 0 if `MasVnrArea` == 0 and `MasVnrType` are Null.
df_train.loc[              (df_train.loc[:,["MasVnrType"]].isnull().all(axis=1)) &              (df_train["MasVnrArea"]==0),              "HasMasVnr"]=0
df_test.loc[             (df_test.loc[:,["MasVnrType"]].isnull().all(axis=1)) &             (df_test["MasVnrArea"]==0),             "HasMasVnr"]=0

# Change `HasMasVnr` to 1 if `MasVnrArea` > 0 or `MasVnrType` is not Null.
df_train.loc[(df_train["MasVnrArea"]>0) | (df_train["MasVnrType"].isnull()==False),"HasMasVnr"]=1
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


garage_cat_features=["GarageType","GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "GarageFinish"]
#garage_cat_features = [x for x in garage_cat_features_all if x not in lowInfoGainCols]
garage_features = garage_cat_features + ["GarageArea", "GarageCars"]
df_train.loc[     (df_train.loc[:,garage_cat_features].isnull().all(axis=1)) &              ((df_train["GarageArea"] == 0) |  (df_train["GarageArea"].isnull() == True)) &              ((df_train["GarageCars"] == 0) |  (df_train["GarageCars"].isnull() == True)),     "HasGarage"]=0
df_train.loc[     (df_train.loc[:,garage_cat_features].notnull().any(axis=1)) |              (df_train["GarageArea"] > 0) |              (df_train["GarageCars"] > 0),     "HasGarage"]=1

df_test.loc[     (df_test.loc[:,garage_cat_features].isnull().all(axis=1)) &              ((df_test["GarageArea"] == 0) |  (df_test["GarageArea"].isnull() == True)) &              ((df_test["GarageCars"] == 0) |  (df_test["GarageCars"].isnull() == True)),     "HasGarage"]=0
df_test.loc[     (df_test.loc[:,garage_cat_features].notnull().any(axis=1)) |              (df_test["GarageArea"] > 0) |              (df_test["GarageCars"] > 0),     "HasGarage"]=1


# If there is no Garage then we can set `GarageQual`,`GarageFinish`, `GarageCond`, `GarageArea`, and `GarageCars` to 0/

# In[ ]:


for gar_feat in ["GarageQual","GarageFinish", "GarageCond", "GarageArea", "GarageCars","GarageType"]:
    df_train.loc[df_train["HasGarage"]==0,gar_feat]=0
    df_test.loc[df_test["HasGarage"]==0,gar_feat]=0


# If GarageYrBlt is null, just set GarageYrBlt to the most common year that Garages are built.

# In[ ]:


combined = pd.concat([df_train, df_test], sort=False)
garageYrBlt_mode = combined["GarageYrBlt"].mode(dropna=True).values[0]
print("GarageYrBlt Mode: "+ str(garageYrBlt_mode))

df_train.loc[df_train["HasGarage"]==0,"GarageYrBlt"]=garageYrBlt_mode
df_test.loc[df_test["HasGarage"]==0,"GarageYrBlt"]=garageYrBlt_mode

df_train.loc[df_train["GarageYrBlt"].isnull(),"GarageYrBlt"] = garageYrBlt_mode
df_test.loc[df_test["GarageYrBlt"].isnull(),"GarageYrBlt"] = garageYrBlt_mode


# unfortunately we are not done with the null features in the Garage Columns

# In[ ]:


low_cols=[]
for gcol in garage_features:
    print("Number of %s Nulls in Training + Testing Respectively:" % gcol)
    print(df_train[gcol].isnull().sum())
    print(df_test[gcol].isnull().sum())
    if df_train[gcol].isnull().sum() + df_test[gcol].isnull().sum() < 3:
        low_cols.append(gcol)
print("rows with low amount of null values")
print(low_cols)


# Since there are only 2 rows with null values in the GarageFinish,GarageQual, GarageCond, GarageArea, and GarageCars variables in the testing data I'll just set them to the mode values since there are so few rows with nulls anyway.

# In[ ]:


df_test.loc[df_test.loc[:,low_cols].isnull().any(axis=1),             low_cols]


# In[ ]:


for grow in low_cols:
    mode_val = combined[grow].mode().values[0]
    if (df_train[grow].dtypes == "int64"):
        mode_val = int(np.round(mode_val,0))
    df_test.loc[df_test[grow].isnull(),grow]=mode_val
    print(mode_val)
    #print(df_test.loc[df_test[grow].isnull(),grow])


# In[ ]:


df_test.iloc[[666,1116],:].loc[:,low_cols]


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
#basement_cat_features = [x for x in basement_cat_features_all if x not in lowInfoGainCols]
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


for basem_feat in basement_features:
    if basem_feat not in cat_ord_vars:
        df_train.loc[df_train["HasBasement"]==0,basem_feat]=0
        df_test.loc[df_test["HasBasement"]==0,basem_feat]=0


# Unfortunately we still have not dealt with all of the null values for basement features.

# In[ ]:


for bcol in basement_features:
    print("Number of %s Nulls in Training + Testing Respectively:" % bcol)
    print(df_train[bcol].isnull().sum())
    print(df_test[bcol].isnull().sum())


# Since there are only 2 and 3 missing values in "BsmtQual" and "BsmtCond"columns respectively and their values have the same range I just set the null values to the value of the other. 

# In[ ]:


df_train.loc[df_train["BsmtQual"].isnull(),"BsmtQual"]=df_train.loc[df_train["BsmtQual"].isnull(),"BsmtCond"]
df_test.loc[df_test["BsmtQual"].isnull(),"BsmtQual"]=df_test.loc[df_test["BsmtQual"].isnull(),"BsmtCond"]
df_train.loc[df_train["BsmtCond"].isnull(),"BsmtCond"]=df_train.loc[df_train["BsmtCond"].isnull(),"BsmtQual"]
df_test.loc[df_test["BsmtCond"].isnull(),"BsmtCond"]=df_test.loc[df_test["BsmtCond"].isnull(),"BsmtQual"]


# For basement exposure, I just set it to the mode value based on both the training and test set where HasBasement is true.

# In[ ]:


mode_basement_exposure_given_basement = np.round(
    df_train.loc[df_train["HasBasement"]==1,"BsmtExposure"].append(
        df_test.loc[df_test["HasBasement"]==1,"BsmtExposure"]).mode().values[0],0)
print(mode_basement_exposure_given_basement)
df_train.loc[df_train["BsmtExposure"].isnull(),"BsmtExposure"] = mode_basement_exposure_given_basement
df_test.loc[df_test["BsmtExposure"].isnull(),"BsmtExposure"] = mode_basement_exposure_given_basement


# Since there is only 1 null row with `BsmtFinType2` and `BsmtFinType2` is highly correlated with `BsmtFinSF2`, I got the "BsmtFinSF2" value of the null row:

# In[ ]:


print(df_train.loc[df_train["BsmtFinType2"].isnull(),"BsmtFinSF2"])


# Then I found the standard deviation of the `BsmtFinSF2` in houses with basements

# In[ ]:


BsmtFinSF2_train_test_df =     df_train.loc[df_train["BsmtFinSF2"]>0,["BsmtFinSF2","BsmtFinType2"]].append(
    df_test.loc[df_test["BsmtFinSF2"]>0,["BsmtFinSF2","BsmtFinType2"]])
np.std(BsmtFinSF2_train_test_df["BsmtFinSF2"])


# I took the average of "BsmtFinType2" values within 1 SD around the `BsmtFinSF2` value

# In[ ]:


min_BsmtFinSF2 = df_train.loc[df_train["BsmtFinType2"].isnull(),"BsmtFinSF2"] -     np.std(BsmtFinSF2_train_test_df["BsmtFinSF2"])
max_BsmtFinSF2 = df_train.loc[df_train["BsmtFinType2"].isnull(),"BsmtFinSF2"] +     np.std(BsmtFinSF2_train_test_df["BsmtFinSF2"])
subset_BsmtFinSF2_train_test_df = BsmtFinSF2_train_test_df.loc[     (BsmtFinSF2_train_test_df["BsmtFinSF2"] > min_BsmtFinSF2.values[0]) &     (BsmtFinSF2_train_test_df["BsmtFinSF2"] < max_BsmtFinSF2.values[0]),"BsmtFinType2"]

df_train.loc[df_train["BsmtFinType2"].isnull(),"BsmtFinType2"] =     np.round(subset_BsmtFinSF2_train_test_df.mean(skipna=True),0)


# ### KitchenQual
# If there is no Kitchen Quality value then set it to the average kitchen quality value

# In[ ]:


df_train.loc[df_train["KitchenQual"].isnull(),"KitchenQual"] = np.round(df_train["KitchenQual"].append(df_test["KitchenQual"]).mean(skipna=True),0)
df_test.loc[df_test["KitchenQual"].isnull(),"KitchenQual"] = np.round(df_train["KitchenQual"].append(df_test["KitchenQual"]).mean(skipna=True),0)


# ## Transform Continuous Variables

# In[ ]:


num_sqrt_candidates = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',                        'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',                        '2ndFlrSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']
for num_feat in num_sqrt_candidates:
    new_num_col = num_feat +"_SQRT"
    df_train.loc[:,new_num_col]=np.sqrt(df_train.loc[:,num_feat].values)
    df_test.loc[:,new_num_col]=np.sqrt(df_test.loc[:,num_feat].values)
    
    
num_log_candidates = ['LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
for num_feat in num_log_candidates:
    new_num_col = num_feat +"_LOG"
    df_train.loc[:,new_num_col]=np.log(df_train.loc[:,num_feat].values+.001)
    df_test.loc[:,new_num_col]=np.log(df_test.loc[:,num_feat].values+.001)


# # Categorical Nominal Variables
# * One-Hot Encode Categorical Nominal Variables

# In[ ]:


# add one-hot encoded columns based on `col` in `df` to `df`
def one_hot_encode(df, col):
    df[col] = pd.Categorical(df[col])
    dfDummies = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df, dfDummies], axis=1)
    #df = df.drop([col],axis=1)
    return(df)


# In[ ]:


for cat_ord_col in cat_ord_vars:
        df_train = one_hot_encode(df_train,cat_ord_col)
        df_test = one_hot_encode(df_test,cat_ord_col)


# ### Drop Columns not found in Training or Testing Set

# In[ ]:


train_cols = list(df_train.columns.sort_values().unique())
test_cols = list(df_test.columns.sort_values().unique())
uniq_train_cols = [x for x in train_cols if (x not in test_cols and x != "SalePrice")]
uniq_test_cols = [x for x in test_cols if x not in train_cols]

print("length of unique training columns: "+ str(len(uniq_train_cols)))
print("length of unique test columns: "+ str(len(uniq_test_cols)))
df_train = df_train.drop(uniq_train_cols,axis=1)
df_test = df_test.drop(uniq_test_cols,axis=1)


# ## A lot of Feature Engineering
# The code was adapted came from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

# In[ ]:


# Overall quality of the house
df_train["OverallGrade"] = df_train["OverallQual"] * df_train["OverallCond"]
df_test["OverallGrade"] = df_test["OverallQual"] * df_test["OverallCond"]
# Overall quality of the garage
df_train["GarageGrade"] = df_train["GarageQual"] * df_train["GarageCond"]
df_test["GarageGrade"] = df_test["GarageQual"] * df_test["GarageCond"]
# Overall quality of the exterior
df_train["ExterGrade"] = df_train["ExterQual"] * df_train["ExterCond"]
df_test["ExterGrade"] = df_train["ExterQual"] * df_train["ExterCond"]
# Overall kitchen score
df_train["KitchenScore"] = df_train["KitchenAbvGr"] * df_train["KitchenQual"]
df_test["KitchenScore"] = df_train["KitchenAbvGr"] * df_train["KitchenQual"]
# Overall fireplace score
df_train["FireplaceScore"] = df_train["Fireplaces"] * df_train["FireplaceQu"]
df_test["FireplaceScore"] = df_train["Fireplaces"] * df_train["FireplaceQu"]
# Overall garage score
df_train["GarageScore"] = df_train["GarageArea"] * df_train["GarageQual"]
df_test["GarageScore"] = df_train["GarageArea"] * df_train["GarageQual"]
# Overall pool score
df_train["PoolScore"] = df_train["PoolArea"] * df_train["PoolQC"]
df_test["PoolScore"] = df_train["PoolArea"] * df_train["PoolQC"]
# Total number of bathrooms
df_train["TotalBath"] = df_train["BsmtFullBath"] + (0.5 * df_train["BsmtHalfBath"]) + df_train["FullBath"] + (0.5 * df_train["HalfBath"])
df_test["TotalBath"] = df_train["BsmtFullBath"] + (0.5 * df_train["BsmtHalfBath"]) + df_train["FullBath"] + (0.5 * df_train["HalfBath"])
# Total SF for house (incl. basement)
df_train["AllSF"] = df_train["GrLivArea"] + df_train["TotalBsmtSF"]
df_test["AllSF"] = df_train["GrLivArea"] + df_train["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
df_train["AllFlrsSF"] = df_train["1stFlrSF"] + df_train["2ndFlrSF"]
df_test["AllFlrsSF"] = df_train["1stFlrSF"] + df_train["2ndFlrSF"]
# Total SF for porch
df_train["AllPorchSF"] = df_train["OpenPorchSF"] + df_train["EnclosedPorch"] + df_train["3SsnPorch"] + df_train["ScreenPorch"]
df_test["AllPorchSF"] = df_train["OpenPorchSF"] + df_train["EnclosedPorch"] + df_train["3SsnPorch"] + df_train["ScreenPorch"]
# Has masonry veneer or not
df_train["HasMasVnr"] = df_train.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, "Stone" : 1, "None" : 0})
df_test["HasMasVnr"] = df_train.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, "Stone" : 1, "None" : 0})
# House completed before sale or not
df_train["BoughtOffPlan"] = df_train.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
df_test["BoughtOffPlan"] = df_train.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})


# In[ ]:


# drop categorical ordinal columns that are not encoded
df_train_numerical = df_train.drop(cat_ord_vars, axis=1)
# drop categorical ordinal columns that are not encoded
df_test_numerical = df_test.drop(cat_ord_vars, axis=1)


# ## Fix GarageCars
# In my exploratory notebook I found that I should merge houses with 3 or more GarageCars. However, when I implemented this into the model I got worse predictions.

# In[ ]:


if 1==1:
    garageCarsReplaceDict = {"GarageCars":{4:3}}

    df_train_numerical = df_train_numerical.replace(garageCarsReplaceDict)
    df_test_numerical = df_test_numerical.replace(garageCarsReplaceDict)


# ## Transform GarageScore
# get the log of the garageScore
# 

# In[ ]:


df_train_numerical.loc[:,'GarageScore_Log']=np.log(df_train_numerical.loc[:,'GarageScore'].values + .01)
df_test_numerical.loc[:,'GarageScore_Log']=np.log(df_test_numerical.loc[:,'GarageScore'].values + .01)


# ## Fix FullBath
# You need a bath in a house right? I should set the 0's to 1's? 
# 
# When I implemented this into the model I got worse predictions.

# In[ ]:


if 1==1:
    FullBathReplaceDict = {"FullBath":{0:1}}
    df_train_numerical = df_train_numerical.replace(FullBathReplaceDict)
    df_test_numerical = df_test_numerical.replace(FullBathReplaceDict)


# ## Transform Year Features

# In[ ]:


year_data_released=2010
df_train_numerical['YearRemodAdd'] = year_data_released - df_train_numerical['YearRemodAdd']
df_train_numerical['YearBuilt'] = year_data_released - df_train_numerical['YearBuilt']
df_train_numerical['YearRemodelPlusBuilt'] = df_train_numerical['YearRemodAdd'] + df_train_numerical['YearBuilt']
df_test_numerical['YearRemodAdd'] = year_data_released - df_test_numerical['YearRemodAdd']
df_test_numerical['YearBuilt'] = year_data_released - df_test_numerical['YearBuilt']
df_test_numerical['YearRemodelPlusBuilt'] = df_test_numerical['YearRemodAdd'] + df_test_numerical['YearBuilt']


# # Dropping outliers from training set

# The author of the dataset specifically recommends removing 'any houses with more than 4000 square feet' from the dataset. Reference : https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# In[ ]:


df_train_numerical.drop(df_train_numerical[df_train_numerical["GrLivArea"] > 4000].index, inplace=True)


# # Choosing Best Variables

# In[ ]:


combined = pd.concat([df_train_numerical, df_test_numerical], sort=False)
one_hot_cols_lofl = []
for car_ord_predix in cat_ord_vars + ["Exterior", "Condition"]:
    one_hot_cols_lofl.append(
        [x for x in list(combined.columns) if car_ord_predix in x])
one_hot_cols = [item for sublist in one_hot_cols_lofl for item in sublist]


# In[ ]:


one_hot_cols = [x for x in one_hot_cols if x in list(df_train_numerical.columns)]

best_one_hot_cols=[]
for var in one_hot_cols:
    val0_mean = df_train_numerical.groupby(var)['SalePrice'].mean()[0]
    val0_std = df_train_numerical.groupby(var)['SalePrice'].std()[0]
    val1_mean = df_train_numerical.groupby(var)['SalePrice'].mean()[1]
    if val1_mean < (val0_mean-val0_std) or val1_mean >(val0_mean+val0_std):
        if var not in best_one_hot_cols:
            best_one_hot_cols.append(var)
        data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
print(best_one_hot_cols)


# # Cross-Validate
# In this competition we are evaluated by the root-mean-squared-error (RMSE)

# ## Favorite Variables (Set1)
# 

# In some models you can import all of the features

# In[ ]:


combined.columns[combined.isna().any()].tolist()


# In[ ]:


combined = pd.concat([df_train_numerical, df_test_numerical], sort=False)
nonNA_cols=list(combined.dropna(axis=1).drop('Id',axis=1).columns)


# However, in some models you want to perform feature selection to improve the algorithm.

# In[ ]:


fav_variables1=['OverallQual','YearRemodAdd','FireplaceQu',                'GarageFinish', 'FullBath', 'GarageCars',                'KitchenQual', 'ExterQual', 'ExterQual',
              'GrLivArea', 'GarageScore', 'TotalBsmtSF','YearRemodelPlusBuilt'] \
                + one_hot_cols


# In[ ]:


for fv in fav_variables1:
    if fv not in list(df_train_numerical.columns):
        print("train no")
        print(fv)
    if fv not in list(df_test_numerical.columns):
        print("test no")
        print(fv)


# ## Linear model
# Let's try cross-validating a basic linear regression model (cv=5) using only features from my favorite variables

# In[ ]:


def linear_reg(df_train,df_test,features):
    X_train = df_train.loc[:,features].copy()
    y_train = df_train.loc[:,"SalePrice"].copy()
    X_pred = df_test.loc[:,features].copy()
    
    reg = LinearRegression().fit(X_train, y_train)
    scores = np.sqrt(-cross_val_score(reg,X_train,y_train, cv=5, scoring='neg_mean_squared_error'))
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    pred = reg.predict(X_pred)
    return(pred)


# In[ ]:


pred1 = linear_reg(df_train_numerical,
                  df_test_numerical,
                  fav_variables1)
df_test.loc[:,"SalePrice"]=np.exp(pred1)
sub1_df = df_test.loc[:,["Id","SalePrice"]].copy()
sub1_df.to_csv("submission_linearRegress_favVars1.csv", index=False)


# In[ ]:


pred2 = linear_reg(df_train_numerical,
                  df_test_numerical,
                  nonNA_cols)
df_test.loc[:,"SalePrice"]=np.exp(pred2)
sub2_df = df_test.loc[:,["Id","SalePrice"]].copy()
sub2_df.to_csv("submission_linearRegress_allVars.csv", index=False)


# We get a low RMSE score from this model.

# ## Ridge Regression
# Ridge Regression uses L2 regularization. The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.

# In[ ]:


def ridge_reg(df_train,df_test,features, alpha, predict=False):
    from sklearn import linear_model
    X_train = df_train.loc[:,features].copy()
    y_train = df_train.loc[:,"SalePrice"].copy()
    X_pred = df_test.loc[:,features].copy()
    
    clf = linear_model.Ridge(alpha=alpha).fit(X_train, y_train)
    scores = np.sqrt(-cross_val_score(clf,X_train,y_train, cv=5, scoring='neg_mean_squared_error'))
    #print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    if predict:
        pred = clf.predict(X_pred)
        return(pred)
    else:
        return(scores)

alphas=[0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge_mean_scores=[]

for i in alphas:
    print("alpha="+str(i))
    cv_ridge = np.mean(ridge_reg(df_train_numerical,
                df_test_numerical,
                fav_variables1,i))
    cv_ridge_mean_scores.append(cv_ridge)


# In[ ]:


cv_ridge_ser = pd.Series(cv_ridge_mean_scores, index = alphas)
cv_ridge_ser.plot(title = "Ridge Regression Validation (favorite variables)")
plt.xlabel("alpha")
plt.ylabel("rmse")


# Note the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model begins to overfit. A value of alpha = 5 is about right based on the plot above.

# In[ ]:


# let's summit with alpha=5
pred3 = ridge_reg(df_train_numerical,
                  df_test_numerical,
                  fav_variables1,5, True)
df_test.loc[:,"SalePrice"]=np.exp(pred3)
sub3_df = df_test.loc[:,["Id","SalePrice"]]
sub3_df.to_csv("submission_ridge_favoriteVars1.csv", index=False)


# In[ ]:


alphas=[0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge_mean_scores=[]

for i in alphas:
    print("alpha="+str(i))
    cv_ridge = np.mean(ridge_reg(df_train_numerical,
                df_test_numerical,
                nonNA_cols,i))
    cv_ridge_mean_scores.append(cv_ridge)


# In[ ]:


cv_ridge_ser = pd.Series(cv_ridge_mean_scores, index = alphas)
cv_ridge_ser.plot(title = "Ridge Regression Validation (all variables)")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[ ]:


# let's summit with alpha=10
pred4 = ridge_reg(df_train_numerical,
                  df_test_numerical,
                  nonNA_cols,10, True)
df_test.loc[:,"SalePrice"]=np.exp(pred4)
sub4_df = df_test.loc[:,["Id","SalePrice"]]
sub4_df.to_csv("submission_ridge_allVars.csv", index=False)


# ## Lasso Regression
# Lasso Regression uses L1 regularization. The main tuning parameter for the Lasso model is also alpha but it works like the inverse of the alpha in the ridge model. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.

# In[ ]:


def lasso_reg(df_train,df_test,features, alpha, predict=False):
    X_train = df_train.loc[:,features].copy()
    y_train = df_train.loc[:,"SalePrice"].copy()
    X_pred = df_test.loc[:,features].copy()
    
    clf = Lasso(alpha=alpha).fit(X_train, y_train)
    scores = np.sqrt(-cross_val_score(clf,X_train,y_train, cv=5, scoring='neg_mean_squared_error'))
    #print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    if predict:
        pred = clf.predict(X_pred)
        return(pred)
    else:
        return(scores)

alphas=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1]
cv_lasso_mean_scores=[]
for i in alphas:
    print("alpha="+str(i))
    cv_lasso = np.mean(lasso_reg(df_train_numerical,
                  df_test_numerical,
                  fav_variables1,i))
    cv_lasso_mean_scores.append(cv_lasso)


# In[ ]:


cv_lasso_ser = pd.Series(cv_lasso_mean_scores, index = alphas)
cv_lasso_ser.plot(title = "Lasso Regression Validation (favorite variables)")
plt.xlabel("alpha")
plt.ylabel("rmse")


# We get the best alpha value using scikit-learn LassoCV

# In[ ]:


lassoCVresults = LassoCV(alphas = alphas, cv=5).fit(
    df_train_numerical.loc[:,fav_variables1].copy(), 
    df_train_numerical.loc[:,"SalePrice"].copy())
print(lassoCVresults.alpha_)

coef = pd.Series(lassoCVresults.coef_, index = fav_variables1)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
coef.sort_values(ascending=False).head(10)


# In[ ]:


# let's summit with alpha=0.0005
pred5 = lasso_reg(df_train_numerical,
                  df_test_numerical,
                  fav_variables1,lassoCVresults.alpha_, True)
df_test.loc[:,"SalePrice"]=np.exp(pred5)
sub5_df = df_test.loc[:,["Id","SalePrice"]]
sub5_df.to_csv("submission_lasso_favoriteVars1.csv", index=False)


# Note that LassoCV method can do feature selection for you so we did not necessarily need to narrow down our favorite features. In fact, LassoCV outputs `coef_` which contains a vector of the weights for the final linear model. Let's see what happens when we use all the features.

# In[ ]:


lassoCVresults = LassoCV(alphas = alphas, cv=5).fit(
    df_train_numerical.loc[:,nonNA_cols].copy(), 
    df_train_numerical.loc[:,"SalePrice"].copy())
print(lassoCVresults.alpha_)


# In[ ]:


coef = pd.Series(lassoCVresults.coef_, index = nonNA_cols)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# Good job Lasso. One thing to note here however is that the features selected are not necessarily the "correct" ones - especially since there are a lot of collinear features in this dataset. One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is. Lets look at some of the weights given to the variables.

# In[ ]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# Interesting choices. These are definitely different then the ones I chose as my favorite. It is weird to me that it chose "1stFlrSF_SQRT" to be weighted higher than "GrLivArea"...

# In[ ]:


# let's summit with alpha=0.0005
pred6 = lasso_reg(df_train_numerical,
                  df_test_numerical,
                  nonNA_cols,lassoCVresults.alpha_, True)
df_test.loc[:,"SalePrice"]=np.exp(pred6)
sub6_df = df_test.loc[:,["Id","SalePrice"]]
sub6_df.to_csv("submission_lasso_allVars.csv", index=False)


# ## KNN
# Now let's try K-nearest neighbors and set K equal to 1-20.

# In[ ]:


def run_knn(df_train,df_test,features):
    X_train = df_train.loc[:,features].copy()
    y_train = df_train.loc[:,"SalePrice"].copy()
    X_pred = df_test.loc[:,features].copy()
    
    
    for K in range(20):
        K=K+1
        knn = neighbors.KNeighborsRegressor(n_neighbors = K)
        scores = np.sqrt(-cross_val_score(knn,X_train,y_train, cv=5, scoring='neg_mean_squared_error'))
        print(K)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


pred_knn = run_knn(df_train_numerical,
                  df_test_numerical,
                  fav_variables1)


# This model does not do as well as previous ones.

# ## XGBoost
# To understand XGBoost I read https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/. Here is my summary:
# 
# XGBoost has 3 types of parameters:
# * General Parameters: Guide the overall functioning
# * Booster Parameters: Guide the individual booster (tree/regression) at each step
# * Learning Task Parameters: Guide the optimization performed
# 
# I chose the default general parameters. The only one that really matters is whether we want to run a tree-based(default) or a linear model. Tree-based models usually perform better so we set it to tree-based.
# 
# Next, the tree booster parameters are
# * max_depth: depth of tree (default:6)
#   * too high--> overfitting
#   * usually ranges from 3-10
# * learning_rate (default:0.3)
#   * usualy ranges from 0.01-0.2
# * n_estimators: number of trees to fit
# * min_child_weight: minimum sum of weights of all obsercations required in a child (default:1)
#   * too high --> underfitting
#   * too low --> overfitting
# * max_leaf_nodes: terminal nodes/leaves in a tree
#   *  used instead of max_depth
#   * a depth of `n` would produces `n^2` leaves
# * gamma: minimum loss reduction required to split (default:0)
# * max_delta_step (default:0)
#   * I probably will ignore this parameter
#   * higher values make model more conservative
#   * can help if class is extremely unbalanced
# * subsample: the fraction of observations to be randomly sampled for each tree (default:1)
#   * usually ranges from 0.5-1
#   * low values -> overfitting
#   * high values -> underfitting
# * colsample_bytree: the fraction of columns to be randomly samples for each tree (default:1)
#   * usually ranges from 0.5-1
# * colsample_bylevel: the subsample ratio of columns for each split, in each level (default:1)
#   * we can probably ignore this parameter since the parameters `subsample` and `colsample_bytree` are similar
# * reg_lambda (default:1)
#   * l2 regularization parameter
#   * used to reduce overfitting
#   * not often used
# * reg_alpha (default:0)
#   * l1 regularization parameter
# * scale_pos_weight (default:1)
#   * used if you have class imbalance
#   * always greater than 0
#   
# The learning tast parameters are the `objective` and the `eval_metric`. We have a regression linear objective and we are measing error with rmse.

# In[ ]:


# for some reason these columns were not already type float
for col in ["BsmtQual", "BsmtCond", "GarageQual", "GarageGrade"]:
    df_train_numerical[col] = df_train_numerical[col].astype(float)
    df_test_numerical[col] = df_test_numerical[col].astype(float)


# Create method for XGBoost

# In[ ]:


def xgboost_reg(df_train,df_test,features, regr, predict=False):
    X_train = df_train.loc[:,features].copy()
    y_train = df_train.loc[:,"SalePrice"].copy()
    X_pred = df_test.loc[:,features].copy()
    
    xgb_fit = regr.fit(X_train, y_train)
    scores = np.sqrt(-cross_val_score(xgb_fit,X_train,y_train, cv=5, scoring='neg_mean_squared_error'))
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    if predict:
        pred = xgb_fit.predict(X_pred)
        return(pred)
    else:
        return(scores)


# To test my code I will use the paramers found in [this kernel](https://www.kaggle.com/humananalog/xgboost-lasso). They are not at all cross-validated so I will need to perform a grid search to get better parameters

# In[ ]:


xgb1 = xgb.XGBRegressor(
    colsample_bytree=0.2,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=1.5,
    n_estimators=7200,
    gamma=0.0,
    reg_alpha=0.9,
    reg_lambda=0.6,
    subsample=0.2,
    objective= 'reg:squarederror',
    seed=14)

xgb_pred_1 = xgboost_reg(df_train_numerical,
                  df_test_numerical,
                  nonNA_cols, xgb1, True)


# In[ ]:


df_test.loc[:,"SalePrice"]=np.exp(xgb_pred_1)
sub7_df = df_test.loc[:,["Id","SalePrice"]].copy()
sub7_df.to_csv("submission_xgBoost_allVars.csv", index=False)

