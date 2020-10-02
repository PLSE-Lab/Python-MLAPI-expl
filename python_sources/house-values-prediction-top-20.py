#!/usr/bin/env python
# coding: utf-8

# The following kernel does some of the simple but essential steps to perform the Regression Analysis. By doing the it, it gives nice performence which puts the kernel in top 20% with the score of 0.12023 at the time of submission.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Data Preview

# In[ ]:


df_train = pd.read_csv(r"../input/train.csv")
df_test = pd.read_csv(r"../input/test.csv")
pd.set_option('display.width', 700)
df_train.head(5)


# In[ ]:


df_test.head(5)


# In[ ]:


# Size of data before any operations:
# Here one less col in df_test becoz SalePrice is missing.
print("The train data shape : {} ".format(df_train.shape))
print("The test data shape : {} ".format(df_test.shape))


# ## Primary Exploration of the Data

# In[ ]:


print(df_train.dtypes)
df_train.describe()


# ## Check for Nans/NA/Missing Data

# In[ ]:


# Remove Id column. It doesn't contribute to the price prediction
df_train.drop("Id", inplace=True, axis=1)
df_test.drop("Id", inplace=True, axis=1)

print("The train data shape after removing Id col: {} ".format(df_train.shape))
print("The test data shape after removing Id col: {} ".format(df_test.shape))


# In[ ]:


# Remove all the rows and columns from the data which has all values Nans/Empty cells
df_train.dropna(axis=1, how="all", inplace=True)
df_train.dropna(axis=0, how="all", inplace=True)

df_test.dropna(axis=1, how="all", inplace=True)
df_test.dropna(axis=0, how="all", inplace=True)

print("The train data shape after removing all col and rows with Nans : {} ".format(df_train.shape))
print("The test data shape after removing all col and rows with Nans: {} ".format(df_test.shape))


# In[ ]:


# Find out the frequency of nulls in the columns

# For training Data
count_nans = len(df_train) - df_train.count()
df_count_nans = count_nans.to_frame()
df_count_nans.columns=["train_nan_count"]
df_count_nans["%_train_nans"]=(df_count_nans["train_nan_count"]/df_train.shape[0]) * 100

# For test data
df_count_nans["test_nan_count"] = len(df_test) - df_test.count()
df_count_nans["%_test_nans"]=(df_count_nans["test_nan_count"]/df_test.shape[0]) * 100

df_count_nans.sort_values("train_nan_count", ascending=False, inplace=True)
df_count_nans.query('train_nan_count > 0 or test_nan_count > 0')


# In[ ]:


# take out the SalePrice from the train data before further processing
y_train = df_train.SalePrice.values
print(y_train)
df_train.drop("SalePrice", inplace=True, axis=1)


# In[ ]:


# Combining all data for the further processing
df_all_data = pd.concat([df_train, df_test])
df_all_data.reset_index(inplace=True, drop=True)
print(df_all_data.shape)
df_all_data.columns
df_all_data.head()


# In[ ]:


# By looking at the analysis of Nans it is clear that the PoolQC, Alley, MiscFeature, Fence has many null values.
# So those aren't contributing to the price a lot. So these columns will be dropped.
df_all_data.drop(["PoolQC", "Alley", "MiscFeature", "Fence"], axis=1, inplace=True)


# In[ ]:


# Fill the values for the rest of the NAs

# No changes in FirplaceQu. Because the NAs indicates NA=No Fireplace. Fill with None
df_all_data["FireplaceQu"].fillna("None", inplace=True)

# GarageCond, GarageType, GarageFinish, GarageQual. Fill with None
df_all_data[["GarageCond", "GarageType", "GarageFinish", "GarageQual"]] = df_all_data[["GarageCond", "GarageType", "GarageFinish", "GarageQual"]].fillna("None")

# For GarageYrBlt filling with the 0. Assuming garage isn't available.
df_all_data["GarageYrBlt"].fillna(0,  inplace=True)

# Fill with None/0 for basement. It is likely that basment isn't available in the houses
df_all_data[["BsmtExposure","BsmtFinType1", "BsmtFinType2", "BsmtCond", "BsmtQual"]] = df_all_data[["BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtCond", "BsmtQual"]].fillna("None")
df_all_data[["BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtHalfBath"]] = df_all_data[["BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtHalfBath"]].fillna(0)

# MasVnrArea. Fill with 0. Prob it is not available in these houses
df_all_data["MasVnrArea"].fillna(0, inplace=True)
df_all_data["MasVnrType"].fillna("None", inplace=True)

# Electrical. Fill with the most common one.
most_common = df_all_data["Electrical"].value_counts().index[0]
df_all_data["Electrical"].fillna(most_common, inplace=True)

# Functional. As directed in data definition, use "Typ" as the default val
df_all_data["Functional"].fillna("Typ", inplace=True)

# KitchenQual. Considering "TA" as default val, which is short of Typical/Average
df_all_data["KitchenQual"].fillna("TA", inplace=True)
   
# Fill with the most common val
most_common =  df_all_data["SaleType"].value_counts().index[0]
df_all_data["SaleType"].fillna(most_common, inplace=True)
    
# No assumption can be made for the following columns val
df_all_data["Utilities"].fillna("None", inplace=True)
df_all_data["Exterior1st"].fillna("None", inplace=True)
df_all_data["Exterior2nd"].fillna("None", inplace=True)

# Fill with the most common val
most_common =  df_all_data["MSZoning"].value_counts().index[0]
df_all_data["MSZoning"].fillna(most_common, inplace=True)


# In[ ]:


# Filling GarageCars per neighborhood. It is most likely that per neighborhood the car space is similar.
grp=df_all_data.groupby("Neighborhood")["GarageCars"].mean()
nan_idx = df_all_data[df_all_data["GarageCars"].isnull()==True].index.tolist()
for idx in nan_idx:
    df_all_data.loc[idx, "GarageCars"] = int(round(grp.loc[df_all_data.iloc[idx]["Neighborhood"]]))

# Filling GarageArea per neighborhood. It is most likely that per neighborhood the car space is similar.
grp=df_all_data.groupby("Neighborhood")["GarageArea"].mean()
nan_idx = df_all_data[df_all_data["GarageArea"].isnull()==True].index.tolist()
for idx in nan_idx:
    df_all_data.loc[idx, "GarageArea"] = int(round(grp.loc[df_all_data.iloc[idx]["Neighborhood"]]))


# In[ ]:


# LotFrontage: The linear feet of the street connected to the property can be based on the building types. So per building type it will be filed in with the avg.
df_all_data["LotFrontage"] = df_all_data.groupby("BldgType")["LotFrontage"].transform(lambda x: x.fillna(x.mean()))


# ## Correlation Study

# In[ ]:


# Find highly correlated features
corr_matrix = df_all_data.corr().abs()
plt.subplots(figsize=(15,10))
sns.heatmap(corr_matrix, cmap="jet")


# In[ ]:


# Only for Trainingdata. The same relationship can be seen as above
corr_matrix = df_train.corr().abs()
plt.subplots(figsize=(15,10))
sns.heatmap(corr_matrix, cmap="jet")


# In[ ]:


# Only for Test data. The same relationship can be seen as above
corr_matrix = df_test.corr().abs()
plt.subplots(figsize=(15,10))
sns.heatmap(corr_matrix, cmap="jet")


# In[ ]:


# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.75)]
print(to_drop)


# The description below inffered based on the data defintions and the correlation matrix.
# * GarageYrBlt and YearBlt has the high correlation. Because garages are mostly built when the houses are built.
# * GarageArea and GarageCars has very high correlation. Because if more cars can be parked, then there will be more garages space and vice versa.
# * The '1stFlrSF', 'TotalBsmtSF'  are correlated. It makes sense because usually basements are usually right below the first floor and mostly similar in size.  
# * TotRmsAbvGrd and the GrLivArea are correlated. It also makes sense because in both of the columns the basement isn't considered.
# 
# So the above features will be dropped for future calculation. 

# In[ ]:


df_all_data.drop(to_drop, axis=1, inplace=True)


# ## Outliers
# 
# Based on the orignal [Ames data defintion](http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt) there are outliers present into the data. Especially in the GrLivArea column. So need to find out those observations and remove it if possible.

# In[ ]:


sns.regplot(x=df_train["GrLivArea"], y=y_train)


# In the above plot we can see the 2 dots in the bottom-right. They are above 4000 sq ft. but has been sold very less amount. Whereas other sellings like top right corner are sold at much more higher prices. So those 2 are outliers and should be deleted. Otherwise the models will try to capture those points and resulting overfit. The data definition also tells there're more outliers but we can keep it. By looking at the plot, it seems like all other points are following trend.

# In[ ]:


drop_points = df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]["GrLivArea"]
drop_points_list = drop_points.index.tolist()
df_all_data.drop(drop_points_list, inplace=True)

# Updating the indexes for training data.
y_train = np.delete(y_train, drop_points_list)
df_train_last_index=df_train.shape[0]-len(drop_points_list)


# ## Categorical Data Transformation

# In[ ]:


#MSSubClass=The building class
df_all_data['MSSubClass'] = df_all_data['MSSubClass'].apply(str)
df_all_data["MSSubClass"] = LabelEncoder().fit_transform(df_all_data["MSSubClass"])


# ## Handle the Skewness in the Data

# In[ ]:


# detect skewed columns
skew_thresh = 0.5
skewed = df_all_data.skew().sort_values(ascending=False)
a=skewed[abs(skewed)>skew_thresh]


# Based on above data there're some skewed data present in the data. So it will be log transformed

# In[ ]:


skewed_cols = skewed[abs(skewed)>skew_thresh].index.tolist()
print(len(skewed_cols))
print(skewed_cols)
df_all_data[skewed_cols] = df_all_data[skewed_cols].apply(np.log1p)
df_all_data[skewed_cols].head()


# In[ ]:


# Use the one hot encoder to change the categorical data in the numeric data
categorical_data_cols = df_all_data.select_dtypes(include=['object'])
print(categorical_data_cols.columns.tolist())
df_all_data = pd.get_dummies(df_all_data)


# # Normalize the data. 
# 

# ## Log Transformation of the SalePrice

# In[ ]:


# Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.
y_train = np.log1p(y_train)


# After applying OneHotEncoder, the data is at different scales now. It needs to be normalized.

# In[ ]:


df_all_data = (df_all_data - df_all_data.mean()) / (df_all_data.max() - df_all_data.min())


# In[ ]:


# Create training and test dataset after data munging
df_tr = df_all_data.iloc[:df_train_last_index]
df_te = df_all_data.iloc[df_train_last_index:]
print(df_tr.shape)
print(df_te.shape)


# 
# ## Applying Regression Algorithms
# Now is the time to apply the regression algorithms! Following algorithms are tried the ElasticNetCV gives the best result.
# 
# * ElasticNet
# * Ridge
# * Random Forest

# In[ ]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def _ApplyLinearAlgo(model_obj, df_tr, df_te, y_train):
    model_obj.fit(df_tr, y_train)
    y_predict = model_obj.predict(df_tr)
    print("r2 score train " + str(r2_score(y_train, y_predict)))
    print("rmse score train " + str(mean_squared_error(y_train, y_predict)))

    print(df_tr.shape)
    print(df_te.shape)
    y_te_pred = np.expm1(model_obj.predict(df_te))
    
    return y_te_pred


# In[ ]:


print("\n")
print("ElasticNetCV")
from sklearn.linear_model import ElasticNetCV
lr = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000)
y_pred_Elastic = _ApplyLinearAlgo(lr, df_tr, df_te, y_train)

print("\n")
print ("\nRidgeCV")
from sklearn.linear_model import RidgeCV
lr=RidgeCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10])
y_te_Ridge = _ApplyLinearAlgo(lr, df_tr, df_te, y_train)

print("\n")
print("RandomForestRegressor")
from sklearn.ensemble import RandomForestRegressor
lr = RandomForestRegressor()
y_te_RF = _ApplyLinearAlgo(lr, df_tr, df_te, y_train)


# With the few submission trials , the ElasticNet gives the best score. So submitting with that. Other algorithms are also giving similar performance.

# In[ ]:


idx = pd.read_csv("../input/test.csv").Id
my_submission = pd.DataFrame({'Id': idx, 'SalePrice': y_pred_Elastic})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
my_submission.head()


# **References:**
# 
# Took some ideas from the https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
