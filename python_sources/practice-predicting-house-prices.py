# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr



## Load in test and training data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Columns in training data: \n", train.columns)
train["SalePrice"].describe()

#%% Make histogram of SalePrice to see distribution


print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train["SalePrice"].kurt())



# %% Look like we have some outliers. (very big but cheap houses)
# Remove houses bigger than 4000 and more expensive than 300000 (outliers)
train = train [-((train.GrLivArea > 4000) & (train.SalePrice < 300000))]

# %% concatenate test and train data 
# ( Dont include SalePrice since this is target variable)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                                test.loc[:,'MSSubClass':'SaleCondition']))

print("all_data shape : ", all_data.shape)

# %% Take a look at our missing data

total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data = missing_data[missing_data.Percent > 0]
print(missing_data[0:10])


# %% Remove variables with more than 15% missing data

all_data = all_data.drop((missing_data[missing_data["Percent"] > 0.15]).index,1)





# %% take a look at our target, and take log of it to normalize it
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
prices = pd.DataFrame({"price":train["SalePrice"], 
                       "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()

# Log transform the Target
train["SalePrice"] = np.log1p(train["SalePrice"])


# %% Fill in missing data (NA's) from catagorical features 

# Check how many bedrooms each home has)
print(all_data.BedroomAbvGr.value_counts())
all_data.loc[:, "BedroomAbvGr"] = all_data.loc[:, "BedroomAbvGr"].fillna(0)

# BsmtQual etc : data description says NA for basement features is "no basement"
all_data.loc[:, "BsmtQual"] = all_data.loc[:, "BsmtQual"].fillna("No")
all_data.loc[:, "BsmtCond"] = all_data.loc[:, "BsmtCond"].fillna("No")
all_data.loc[:, "BsmtExposure"] = all_data.loc[:, "BsmtExposure"].fillna("No")
all_data.loc[:, "BsmtFinType1"] = all_data.loc[:, "BsmtFinType1"].fillna("No")
all_data.loc[:, "BsmtFinType2"] = all_data.loc[:, "BsmtFinType2"].fillna("No")
all_data.loc[:, "BsmtFullBath"] = all_data.loc[:, "BsmtFullBath"].fillna(0)
all_data.loc[:, "BsmtHalfBath"] = all_data.loc[:, "BsmtHalfBath"].fillna(0)
all_data.loc[:, "BsmtUnfSF"] = all_data.loc[:, "BsmtUnfSF"].fillna(0)

# CentralAir : NA most likely means No
all_data.loc[:, "CentralAir"] = all_data.loc[:, "CentralAir"].fillna("N")

# Condition : NA most likely means Normal
all_data.loc[:, "Condition1"] = all_data.loc[:, "Condition1"].fillna("Norm")
all_data.loc[:, "Condition2"] = all_data.loc[:, "Condition2"].fillna("Norm")

# EnclosedPorch : NA most likely means no enclosed porch
all_data.loc[:, "EnclosedPorch"] = all_data.loc[:, "EnclosedPorch"].fillna(0)

# External stuff : NA most likely means average
all_data.loc[:, "ExterCond"] = all_data.loc[:, "ExterCond"].fillna("TA")
all_data.loc[:, "ExterQual"] = all_data.loc[:, "ExterQual"].fillna("TA")

all_data.loc[:, "Fireplaces"] = all_data.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
all_data.loc[:, "Functional"] = all_data.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
all_data.loc[:, "GarageType"] = all_data.loc[:, "GarageType"].fillna("No")
all_data.loc[:, "GarageFinish"] = all_data.loc[:, "GarageFinish"].fillna("No")
all_data.loc[:, "GarageQual"] = all_data.loc[:, "GarageQual"].fillna("No")
all_data.loc[:, "GarageCond"] = all_data.loc[:, "GarageCond"].fillna("No")
all_data.loc[:, "GarageArea"] = all_data.loc[:, "GarageArea"].fillna(0)
all_data.loc[:, "GarageCars"] = all_data.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
all_data.loc[:, "HalfBath"] = all_data.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
all_data.loc[:, "HeatingQC"] = all_data.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
all_data.loc[:, "KitchenAbvGr"] = all_data.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
all_data.loc[:, "KitchenQual"] = all_data.loc[:, "KitchenQual"].fillna("TA")


# MasVnrType : NA most likely means no veneer
all_data.loc[:, "MasVnrType"] = all_data.loc[:, "MasVnrType"].fillna("None")
all_data.loc[:, "MasVnrArea"] = all_data.loc[:, "MasVnrArea"].fillna(0)

all_data.loc[:, "MiscVal"] = all_data.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
all_data.loc[:, "OpenPorchSF"] = all_data.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
all_data.loc[:, "PavedDrive"] = all_data.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"

all_data.loc[:, "PoolArea"] = all_data.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
all_data.loc[:, "SaleCondition"] = all_data.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
all_data.loc[:, "ScreenPorch"] = all_data.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
all_data.loc[:, "TotRmsAbvGrd"] = all_data.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
all_data.loc[:, "Utilities"] = all_data.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
all_data.loc[:, "WoodDeckSF"] = all_data.loc[:, "WoodDeckSF"].fillna(0)

# Some numerical features are actually really categories
all_data = all_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 
                                       190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 
                                   6 : "Jun", 7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 
                                   11 : "Nov", 12 : "Dec"}
                      })

# %% Make new features based on existing features
all_data["OverallGrade"] = all_data["OverallQual"] * all_data["OverallCond"]

# %%  Log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
categorical_feats = all_data.dtypes[all_data.dtypes == "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[abs(skewed_feats) > 0.5]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#%% Check at numerical and categorical data

print("Numerical features : " , str(len(numeric_feats)))
print("Categorical features : " , str(len(categorical_feats)))


# %% filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())

# make dummy variables
all_data = pd.get_dummies(all_data)






# %% check if all our data is filled in
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data = missing_data[missing_data.Percent > 0]

print(missing_data)

# %% find most important features relative to the target
print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
corr_SalePrice = corr.SalePrice
print(corr_SalePrice.head(),"\n")


# %% creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

print("X_train shape", X_train.shape)
print("X_test shape",X_test.shape,"\n")


# %% make our own RMSE with cross-validation
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)
 
# %% Make the linear model (Make sure only numerical values and dummy variables)
lr = LinearRegression()
lr.fit(X_train, y)


# apply the model to X_test, and convert it from log to normal with np.expm1
y_test_pred = np.expm1(lr.predict(X_test))

# Check rmse on test set
print("RMSE on train set = ", rmse_cv(lr).mean())

# %% Write predictions to csv file called "solutions"
solution = pd.DataFrame({"id":test.Id, "SalePrice":y_test_pred})
solution.to_csv("simple_model_3.csv", index = False)


















