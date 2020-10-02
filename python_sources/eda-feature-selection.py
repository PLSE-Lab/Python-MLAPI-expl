#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook includes:
# - Giving a look into the dataset's features through some simple EAD
# - Selecting appropriate data transformation methods, and 
# - (To our best abilities) finding variables that are most predictive of the sale prices.

# The output of this notebook contains a `Columns_importances.csv` (containing importance scores of features trained by our models), a `train_preprocessed.csv` and a `test_preprocessed.csv` (containing proprocessed, feature-selected train and test datasets, respectively). It should be noted that the preprocessing steps and the preprocessed data are by no means optimal for all cases and models. However, I hope that they serve as useful examples or suggestions for further preprocessing.<br>
# <br>
# Please feel free to use the output data (they can be downloaded from [here](https://www.kaggle.com/hoangnguyen719/house-prices-preprocessed-feature-selected) for more consistency) if you find them useful (and give this notebook some credits if you do), as well as drop any comments or suggestions you have. Thank you!
# ### Table of content
# ...<a href='#preparation'>I. Preparation</a><br>
# ......<a href='#package_data_loading'>1. Package & Data Loading</a><br>
# ......<a href='#null_imputation_preprocessing'>2. Null Imputation & Preprocessing</a><br>
# ...<a href='#exploratory_data_analysis'>II. Exploratory Data Analysis</a><br>
# ......<a href='#basement'>1. Basement</a><br>
# .........<a href='#basement_numerical_features'>1.1. Basement - Numerical Features</a><br>
# .........<a href='#basement_categorical_features'>1.2. Basement - Categorical Features</a><br>
# ......<a href='#bath'>2. Bath</a><br>
# ......<a href='#garage'>3. Garage</a><br>
# .........<a href='#garage_categorical_features'>3.1. Garage - Categorical Features</a><br>
# .........<a href='#garage_numerical_features'>3.2. Garage - Numerical Features</a><br>
# ......<a href='#miscellaneous_features'>4. Miscellaneous Features</a><br>
# .........<a href='#heating'>4.1. Heating</a><br>
# .........<a href='#kitchen'>4.2. Kitchen</a><br>
# .........<a href='#fireplace'>4.3. Fireplace</a><br>
# .........<a href='#masonry'>4.4. Masonry</a><br>
# .........<a href='#pool'>4.5. Pool</a><br>
# .........<a href='#miscellaneous'>4.6. Miscellaneous</a><br>
# .........<a href='#porch'>4.7. Porch</a><br>
# ......<a href='#other_features'>5. Other Features</a><br>
# .........<a href='#other_categorical_features'>5.1. Other - Categorical Features</a><br>
# .........<a href='#other_numerical_features'>5.2. Other - Numerical Features</a><br>
# ...<a href='#feature_engineering_testing'>III. Feature Engineering & Testing</a><br>

# # I. Preparation <a id='preparation'></a>
# ## 1. Package & Data Loading <a id='package_data_loading'></a>

# In[ ]:


import os, math, re

from copy import deepcopy
from datetime import datetime as dt
from itertools import product
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import    GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import LinearSVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras import     callbacks, layers, optimizers, regularizers, Sequential
from tqdm import tqdm
import warnings
from xgboost import XGBRegressor

sns.set(style="white")
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def reload(df="train.csv", dropped_columns=['Id'], msg=True): 
    # set dropped_columns = [] if want to keep Id column
    data_path = "../input/house-prices-advanced-regression-techniques/"
    if df == "all_data":
        train = pd.read_csv(data_path + "train.csv")
        test = pd.read_csv(data_path + "test.csv")
        data = train.drop(columns="SalePrice").append(test)
    else:
        data = pd.read_csv(data_path + df)
    if msg:
        print(df + " loaded successfully!")
    return data.reset_index(drop=True).drop(columns=dropped_columns)
train = reload()
test = reload("test.csv")
all_data = reload("all_data")


# ## 2. Null Imputation & Preprocessing <a id='null_imputation_preprocessing'></a>
# Null imputing and other preprocessing steps are considered, perfomred and tested in  [this other notebook](https://www.kaggle.com/hoangnguyen719/null-imputation). Here I will perform only the optimal method.

# In[ ]:


train = reload()
# based on this notebook: https://www.kaggle.com/hoangnguyen719/null-imputation
class PreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, old_new, numeric_to_object):
        self.numeric_to_object = numeric_to_object
        self.old_new = old_new
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        sold_time = X[["MoSold", "YrSold"]].astype(str).agg("/".join, axis=1)
        sold_time = pd.to_datetime(sold_time)
        for old in self.old_new:
            X[self.old_new[old]] = (sold_time - pd.to_datetime(X[old], format="%Y"))                / pd.Timedelta(days=365.25)
        X[self.numeric_to_object] = X[self.numeric_to_object].astype(str)
        X.drop(columns=list(self.old_new.keys()), inplace=True)
        return X
    
class RelationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, related, threshold=None, strategy="most_frequent", fill_value=None
                 , missing_num=np.nan, missing_obj=np.nan
                 , final_num=0, final_obj="None"):
        self.related = related if isinstance(related[0], list) else [related]
        self.threshold = threshold
        if strategy == 'constant':
            self.fill_value = fill_value
        elif strategy not in ['mean', 'median', 'most_frequent']:
            raise Exception("Wrong strategy type! --{}".format(strategy))
        self.strategy = strategy
        self.missings = [missing_num, missing_obj, final_num, final_obj]
        self.missing_num, self.missing_obj, self.final_num, self.final_obj =            self.missings            
            
    def fit(self, X, y=None):
        if self.strategy == 'constant':
            self.statistics_ = pd.Series(data=self.fill_value, index=X.columns)
        else:
            # object columns always have strategy="most_frequent"
            self.statistics_ = X[X.select_dtypes(object).columns].mode().T[0]
            num_cols = X.select_dtypes("number").columns
            if self.strategy ==  "most_frequent":
                self.statistics_ = self.statistics_.append(X[num_cols].mode().T[0])
            elif self.strategy in ["mean", "median"]:
                strat = eval("X[num_cols].{}()".format(self.strategy))
                self.statistics_ = self.statistics_.append(strat)
        return self
    
    def _indexes(self, X, cols):
        """
            Return indexes of the nulls that will be imputed.
        """
        if not self.threshold:
            l = len(cols)
        elif (self.threshold > 0) & (self.threshold < 1):
            l = round(len(cols)*self.threshold)
        else:
            l = self.null_threshold
        missing = X[cols].isin(self.missings).sum(axis=1)
        index = (missing > 0) & (missing < l)
        return index
        
    def transform(self, X, y=None):
        X = X.copy()
        obj_cols = list(X.select_dtypes(object).columns)
        num_cols = list(X.select_dtypes("number").columns)
        obj_len = len(obj_cols)
        num_len = len(num_cols)
        to_replace = pd.Series(
            data=[self.missing_obj]*obj_len + [self.missing_num]*num_len
            , index=obj_cols + num_cols)
        final_value = pd.Series(
            data=[self.final_obj]*obj_len + [self.final_num]*num_len
            , index=obj_cols + num_cols)
        
        for cols in self.related:
            index = self._indexes(X, cols)
            X.loc[index, cols] = X.loc[index, cols].replace(
                to_replace=to_replace, value=self.statistics_[cols])
            X[cols] = X[cols].replace(to_replace=to_replace, value=final_value)
        
        flat_related = [c for sublist in self.related for c in sublist]
        remaining = [c for c in X.columns if c not in flat_related]
        X[remaining] = X[remaining].replace(to_replace=to_replace
                                            , value=self.statistics_[remaining])
        return X

RELATED = [["Alley"]
           , ["Condition1", "Condition2"]
           , ["Exterior2nd"]
           , ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1"
              , "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF"
              , "BsmtFullBath", "BsmtHalfBath"] # Basement
           , ["BsmtFinType2", "BsmtFinSF2"] # Basement 2
           , ["Heating", "HeatingQC"] # Heating
           , ["KitchenQual", "KitchenAbvGr"] # Kitchen
           , ["FireplaceQu", "Fireplaces"] # Fireplaces
           , ["GarageType", "YrSinceGarageBlt", "GarageFinish" # Garage
              , "GarageCars", "GarageArea", "GarageQual", "GarageCond"]
           , ["MasVnrType", "MasVnrArea"] # Masonry Veneer
           , ["PoolArea", "PoolQC"] # Pool
           , ["Fence"] # Fence
           , ["MiscFeature", "MiscVal"] # Other miscellaneous
            ]
nonrelated = [
    "MSSubClass", "MSZoning", "LotArea", "Street", "Alley"
    , "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope"
    , "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle"
    , "OverallQual", "OverallCond", "RoofStyle", "RoofMatl", "Exterior1st"
    , "Exterior2nd", "ExterQual", "ExterCond", "Foundation", "CentralAir"
    , "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea"
    , "BedroomAbvGr", "TotRmsAbvGrd", "Functional", "PavedDrive"
    , "WoodDeckSF", "Fence", "MoSold", "YrSold", "SaleType", "LotFrontage"
    , "SaleCondition", "Age", "YrSinceRemod", "YrSinceGarageBlt"
]

flat_related = [c for cols in RELATED for c in cols]

old_new = {"YearBuilt":"Age", "YearRemodAdd":"YrSinceRemod"
           , "GarageYrBlt": "YrSinceGarageBlt"}
num_to_obj = ["MSSubClass", "MoSold", "YrSold"]
categorical_columns = list(all_data.select_dtypes(object).columns) + num_to_obj
cat_to_transform = [c for c in categorical_columns if c in flat_related]

PreI = PreProcessor(old_new, num_to_obj)
# CI = ConstantImputer(columns=cat_to_transform + ["Fence", "Alley"], fill_value="None")
RI = RelationImputer(related = RELATED, final_obj="None")
train = PreI.fit_transform(train)
train = RI.fit_transform(train)

numerical_columns = [col for col in all_data.columns if col not in categorical_columns]


# In[ ]:


train.SalePrice.hist(bins=20)


# In[ ]:


train["SalePrice_log"] = np.log1p(train.SalePrice)
train.drop(columns="SalePrice", inplace=True)
train.SalePrice_log.hist(bins=20)


# # II. Exploratory Data Analysis <a id='exploratory_data_analysis'></a>
# ## 1. Basement <a id='basement'></a>

# In[ ]:


basement = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1'
            , 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF','BsmtFinType2'
            , 'BsmtFinSF2', "BsmtFullBath", "BsmtHalfBath"]
print(*basement, sep=", ")


# ### 1.1. Basement - Numerical features <a id='basement_numerical_features'></a>

# In[ ]:


def corr_heatmap(columns=None, saleprice=["SalePrice_log"], df=train
                 , figsize=(8,6), vmin=-1, vmax=1, showvalue=True):
    columns = df.columns if columns == None else columns + saleprice
    corr = df[columns].corr()
    plt.figure(figsize=figsize)
    return sns.heatmap(corr, vmin=vmin, vmax=vmax, annot=showvalue)
corr_heatmap(basement)


# In[ ]:


def pairplot(columns, include_sale=True, data=train, kwargs={}):
    if include_sale & ("SalePrice_log" not in columns):
        columns = columns + ["SalePrice_log"]
    sns.pairplot(data=data[columns], **kwargs)
pairplot(basement, kwargs={"markers":"+", "height":1.25})


# 
# The overall-area variable `TotalBsmtSF` seems  the most linearly predictive. If people are interested in area more than other characteristics of the basement (finished, unfinished, etc.), it may be worth removing the other three area variables `BsmtFinSF1`, `BsmtFinSF2` and `BsmtUnfSF` to save running time and prevent overfitting. At the end, we will test whether dropping these variables is a good idea. In addition, as for `BsmtFinSF1` and `BsmtFinSF2`, let's see if they are more predictive when combined with `BsmtFinType`.<br>
# <br>
# It should also be noted that the majority of `BsmtHalfBath` is 0. Let's convert it to a dummy that evaluates to 0 if there is no basement halfbath and 1 otherwise.

# In[ ]:


train.BsmtHalfBath.value_counts()


# Next, we will test a few potential variables.

# In[ ]:


train["BsmtFinSF1_UnfSF"] = train.BsmtFinSF1 + train.BsmtUnfSF
train["BsmtFinSF"] = train.BsmtFinSF1 + train.BsmtFinSF2
train["BsmtBathPerSF"] = (train.BsmtFullBath + train.BsmtHalfBath/2) / train.TotalBsmtSF
train["BsmtBathPerFinSF"] = (train.BsmtFullBath + train.BsmtHalfBath/2)    / (train.TotalBsmtSF - train.BsmtUnfSF)


# In[ ]:


def scatterplot(x, y="SalePrice_log", df=train, figsize=(8,6), kwargs={}):
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, data=df, **kwargs)
corr_heatmap(["BsmtFinSF1_UnfSF", "BsmtFinSF", "BsmtBathPerSF", "BsmtBathPerFinSF"]
             , figsize=(5,4)
            )


# The new variables created above doesn't show any new information, so we won't recreate them at the end.<br>

# ### 1.2. Basement - Categorical Features <a id='basement_categorical_features'></a>

# In[ ]:


print(*[c for c in categorical_columns if c in basement], sep=", ")


# All of these variables are ordinal so we can easily convert them to numerical features later. Let's take a look at those features (and few other newly created variables).

# In[ ]:


Score_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "None":1, "Po":0}
BsmtFin_map = {"GLQ":5, "ALQ":4, "BLQ":3, "Rec":2, "LwQ":1, "Unf":0, "None":0}
BsmtEx_map = {"Gd":3, "Av":2, "Mn":1, "No":0, "None":0}
train["BsmtFinType1"] = train.BsmtFinType1.replace(BsmtFin_map)
train["BsmtFinType2"] = train.BsmtFinType2.replace(BsmtFin_map)
train["BsmtFinTypeAvg"] = (train["BsmtFinType1"] + train["BsmtFinType2"]) / 2
train["BsmtFinTypeProd"] = train["BsmtFinType1"] * train["BsmtFinType2"]


# In[ ]:


def multiple_violinplots(Xs, df=train, y="SalePrice_log"
                         , size=4, ncol=3, nrow=None, hue=None):
    nrow = nrow or (len(Xs) // ncol + 1)
    fig = plt.figure(figsize=(size*ncol, size*nrow))
    axes = []
    for i in range(len(Xs)):
        ax = plt.subplot(nrow, ncol, i+1)
        sns.violinplot(x=Xs[i], y=y, data=df, hue=hue, ax=ax)
        if (i % ncol) != 0:
            ax.set_ylabel("")
        axes.append(ax)
    plt.suptitle(y, fontsize=16)
    fig.tight_layout(rect=[0,0,1,1 - 0.5/size/nrow])
    return fig, axes

for col, m in zip(["BsmtQual", "BsmtCond", "BsmtExposure"]
                 , [Score_map]*2 + [BsmtEx_map]):
    train[col] = train[col].replace(m)

fig, axes = multiple_violinplots(
    ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1"
     , "BsmtFinType2", "BsmtFinTypeAvg", "BsmtFinTypeProd"]
    , size=3
)


# `BsmtQual`, `BsmtCond` and `BsmtExposure` all seems predictive enough to be kept, but `BsmtFinType1` and `BsmtFinType2` are not very so. We will keep them for now and review their predictive capability at the end. The additional variables don't generate much insights so we will ignore them.<br>
# <br>
# Next, let's try combining `BsmtFinSF` and `BsmtFinType`.

# In[ ]:


train["BsmtFinType_SF1"] = train.BsmtFinType1 * train.BsmtFinSF1
train["BsmtFinType_SF2"] = train.BsmtFinType2 * train.BsmtFinSF2
bsmt_ScSf = ["BsmtFinSF1", "BsmtFinType_SF1", "BsmtFinSF2", "BsmtFinType_SF2"]
pairplot(bsmt_ScSf, kwargs={"height":1.3})


# In[ ]:


corr_heatmap(bsmt_ScSf, figsize=(4,3))


# `BsmtFinType_SF` is more powerful than its creaters in the case of type 1 but not so in type 2; this is strange as we expect the behavior would be consistent. It's possibly because there are not many samples with type 2 basement so much of what we see here of type 2 is just noise. We will keep `BsmtFinType_SF1` and `BsmtFinType_SF2` inplace of `BsmtFinSF1` and `BsmtFinSF2`, respectively.<br>
# <br>
# I've also looked at in details how `BsmtCond` and `BsmtExposure` may possible interact with other basement numerical features. Not much information surfaced except one tiny thing between `BsmtExposure` and `BsmtUnfSF`.<br>
# It's not very clear but we can see in the graph below that Good `BsmtExposure` has different y-intercept compared to other type of basement exposure. Let's add a variable `BsmtUnfExp`=`BsmtExposure`*`BsmtUnfSF`.

# In[ ]:


# pairplot(basement, kwargs={"hue":"BsmtCond", "diag_kind":"hist", "height":1.5})
# pairplot(basement, kwargs={"hue":"BsmtExposure", "diag_kind":"hist"
#                            , "height":1.5})
def multiple_regplots(x, hue, y="SalePrice_log", df=train, kwargs={}):
    g = sns.FacetGrid(train, hue=hue, size=5)
    g.map(sns.regplot, x, y, ci=None, robust=1, **kwargs)
    g.add_legend()
multiple_regplots("BsmtUnfSF", "BsmtExposure"
                  , kwargs={"scatter_kws":{"alpha":0.5}})


# **SUMMARY**: for basement features, we will
# - Convert the following to numerical:
#     - `BsmtHalfBath` to `BsmtHfBthDummy`: 0 if `BsmtHalfBath`=0, 1 otherwise
#     - `BsmtFinType1`, `BsmtFinType2`
#         - Mapping: `Bsmt_map` = {"GLQ":5, "ALQ":4, "BLQ":3, "Rec":2, "LwQ":1, "Unf":0, "None":0}
#     - `BsmtExposure`
#         - Mapping: `Bsmt_ExMap` = {"Gd":3, "Av":2, "Mn":1, "No":0, "None":0}
#     - `BsmtQual`, `BsmtCond`
#         - Mapping: `Score_map` = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "None":1, "Po":0}
# - Create the following new variables:
#     - `BsmtFinType_SF1` = `BsmtFinType1` * `BsmtFinSF1`
#     - `BsmtFinType_SF2` = `BsmtFinType2` * `BsmtFinSF2`
#     - `BsmtUnfExp`=`BsmtExposure`*`BsmtUnfSF`

# We will also test the predictive power of the following features during model evaluation (and drop them if appropriate).
# - `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `BsmtFullBath`, `BsmtFinType1`, `BsmtFinType2`, `BsmtUnfExp`

# In[ ]:


test_features = [["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"
                  ,"BsmtFinType1", "BsmtFinType2", "BsmtUnfExp"]]


# ## 2. Bath <a id='bath'></a>

# In[ ]:


bath = ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]
print(*bath, sep=", ")


# We will look at these features in addition to a few of their possible combinations.

# In[ ]:


div_0_factor = 0.0001 # To avoid division by 0
train["AllBath"] = train.BsmtFullBath + train.FullBath +    (train.BsmtHalfBath + train.HalfBath)/2 # Half bath is considered 1/2 of Fullbath
train["AllBathPerSF"] = train.AllBath / (train.TotalBsmtSF + train.GrLivArea + div_0_factor)
train["BathPerSF"] = (train.FullBath + train.HalfBath/2) / (train.GrLivArea + div_0_factor)
train["BathPerBedrooms"] = (train.FullBath + train.HalfBath/2)    / (train.BedroomAbvGr + div_0_factor)

corr_heatmap(bath + ["AllBath", "AllBathPerSF", "BathPerSF", "BathPerBedrooms"])


# In[ ]:


pairplot(bath + ["AllBath", "AllBathPerSF", "BathPerSF", "BathPerBedrooms"]
        , kwargs={"height":1.24})


# Few things to note from the graphs above:
# - People seem to care more about upper-ground baths than about basement baths.
# - The bath-to-room and bath-to-area ratio does not matter as much as the absolute number of baths itself.

# Therefore, an overall `AllBath` seems more reasonable than 4 separate bath variables. We will create the aggregate variable and compare it with the four separate ones to confirm our assumption.

# In[ ]:


test_features.append(
    ["AllBath", "BsmtHalfBath", "BsmtFullBath", "HalfBath", "FullBath"]
)


# ## 3. Garage <a id='garage'></a>
# ### 3.1. Garage - Categorical Features <a id='garage_categorical_features'></a>

# In[ ]:


print(*["GarageType", "GarageQual", "GarageCond", "GarageFinish"], sep=", ")


# We will first convert `GarageQual`, `GarageCond` and `GarageFinish` to ordinal, numerical variables.

# In[ ]:


GargFin_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
for c, m in zip(["GarageQual", "GarageCond", "GarageFinish"]
               , [Score_map]*2 + [GargFin_map]):
    train[c] = train[c].replace(m)
fig, axes = multiple_violinplots(
    ["GarageType", "GarageQual", "GarageCond", "GarageFinish"]
    , ncol=2
)
axes[0].set_xticklabels(labels=axes[0].get_xticklabels(), rotation=45)
fig.tight_layout(rect=[0,0,1,0.95])


# It doesn't seem obvious but there is some similarity between different values of `GarageType`. Specifically:
# - *Attchd*, *BuiltIn* and *Basement* are all built-in garages.
# - *Detch* and *CarPort* are all separate-from-house garages.<br>
# <br>
# Therefore it seems reasonable to consolidate the information.

# In[ ]:


GargType_map = {"BuiltIn": "Attchd"
                , "Basment": "Attchd"
                , "CarPort": "Detchd"}
train["GarageTypeSum"] = train.GarageType.replace(GargType_map)
fig, axes = multiple_violinplots(["GarageTypeSum"])


# ### 3.2. Garage - Numerical Features <a id='garage_numerical_features'></a>

# In[ ]:


garage_cat = ["YrSinceGarageBlt", "GarageCars", "GarageArea"]
print(*garage_cat, sep=", ")
corr_heatmap(garage_cat + ["GarageCond", "GarageQual"], figsize=(4.5, 4))


# `GarageCond` and `GarageQual` are highly correlated. Intuitively, these variables (one is "Garage condition", other "Garage quality", as described in the data description) seem very similar, or at least represent features that are similar. Therefore, we will drop one (`GarageCond` in this case).

# In[ ]:


pairplot(garage_cat, kwargs={"height":1.5})


# Both `GarageCars` and `GarageArea` are indicator of the area's size so one may be tempting to drop one. In fact it may be good to do so. However, it should be noted that these two variables do NOT measure the same thing: `GarageArea` measures the total SPACE of the garage, while `GarageCars` in some senses represent the garage's WIDTH. Therefore, we will keep both fields for now and compare them during model training before choosing which field to drop, if any.
# <br><br>
# **SUMMARY**: For garage features, we will
# - Convert the following:
#     - `GarageQual`
#         - Using Score_map
#     - `GarageFinish`
#         - GargFin_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
#     - `GarageType` to `GarageTypeSum`
#         - GargType_map = {"BuiltIn": "Attchd", "Basment": "Attchd", "CarPort": "Detchd"}
# - Drop `GarageCond`
# - Compare `GarageArea` and `GarageCars`.

# In[ ]:


test_features.append(["GarageCars", "GarageArea"])


# ## 4. Miscellaneous features <a id='miscellaneous_features'></a>
# ### 4.1. Heating <a id='heating'></a>

# In[ ]:


heating = ['Heating', 'HeatingQC']
fig, ax = multiple_violinplots(heating, size=5, ncol=2)


# [This notebook](https://www.kaggle.com/humananalog/xgboost-lasso) suggests using a score systems where *Fa* and *Po* are grouped together and *Ex*, *Gd* and *TA* are grouped together. That's an option, but based on the graph above it doesn't seem optimal. If anything, we may either use the score mapping above (*Ex*=5, *Gd*=4, *TA*=3, *Fa*=2, *None*=1, *Po*=0), or group *TA* and *FA* together and leave the rest as is.<br><br>
# As for `Heating`, based on my own research, *Grav* is a pretty old system (invented and used during 1800s and 1900s). However, other than that there is no clear information about the preference/ordinality of the different types of heating system. Therefore, we will convert those to dummy variables.<br>
# <br>
# **SUMMARY**: 
# - Convert `HeatingQC` to `HeatingQCScore` using our standard`Score_map`.
# - Convert `Heating` to dummies

# ### 4.2. Kitchen <a id='kitchen'></a>
# It can be easily seens that `KitchenQual` can be converted to an ordinal numerical feature as we did above.

# In[ ]:


train["KitchenQualScore"] = train.KitchenQual.replace(Score_map)
train["KitchenScore"] = train.KitchenQualScore * train.KitchenAbvGr
corr_heatmap(["KitchenAbvGr", "KitchenQualScore", "KitchenScore"], figsize=(4,3))


# In[ ]:


train.KitchenAbvGr.value_counts()


# Based on the matrix above `KitchenAbvGr` is not at all linearly predictive of sales price. The reason could be that most of the sample has only 1 kitchen and there is not much information about properties with more (or even less) than 1 kitchen. Therefore, let's drop this variable. We willt try to retain its information in the newly created `KitchenScore`.<br>
# <br>
# **SUMMARY**:
# - Convert `KitchenQual` to `KitchenQualScore` using `Score_map`
# - Create `KitchenScore` = `KitchenQualScore` * `KitchenAbvGr*
# - Test `KitchenAbvGr`

# In[ ]:


test_features.append(["KitchenAbvGr"])


# ### 4.3. Fireplace <a id='fireplace'></a>
# Similar to `KitchenQual`, let's first convert `FireplaceQu` to numerical.

# In[ ]:


train["FireplaceQu"] = train.FireplaceQu.replace(Score_map)
train["FireplaceScore"] = train.FireplaceQu * train.Fireplaces
corr_heatmap(["Fireplaces", "FireplaceQu", "FireplaceScore"]
            , figsize=(4,3))


# In[ ]:


pairplot(["Fireplaces", "FireplaceQu", "FireplaceScore"]
        , kwargs={"plot_kws":{"alpha":0.5}, "height":1.75})


# Clearly that properties with more and better fireplaces imply better fire safety and come with higher prices. The additional `FireplaceScore` doesn't seem to bring much more information, however, so we won't reproduce it at the end.<br>
# <br>
# **SUMMARY**
# - Convert `FireplaceQu` to numerical using `Score_map`.

# ### 4.4. Masonry <a id='masonry'></a>

# In[ ]:


multiple_regplots("MasVnrArea", "MasVnrType", kwargs={"scatter_kws":{"alpha":0.5}})


# Together the two fields seem to have some predictive power (specifically, generally higher `MasVnrArea` is correlated with higher `SalePrice`, but we can see that different `MasVnrType` has different y-intercept). Let's keep both.

# ### 4.5. Pool <a id='pool'></a>
# Let's convert `PoolQC` to numerical values.

# In[ ]:


train["PoolQC"] = train.PoolQC.replace(Score_map)
train["PoolArea_QC"] = train.PoolArea * train.PoolQC
corr_heatmap(["PoolArea", "PoolQC", "PoolArea_QC"]
            , figsize=(4,3))


# In[ ]:


train.PoolArea.value_counts()


# Pool is often a luxurious features that often suggest a higher housing price. However in this case, again similar to the kitchen features, there are two few samples with pools and therefore the information they carry may be misleading. We will retain `PoolArea` and `PoolQC` but will need to keep an eye on them during model training. As for the addition feature `PoolArea_QC`, we will ignore it.<br>
# <br>
# **SUMMARY**:
# - Convert `PoolQC` to numerical
# - Test: `PoolArea` and `PoolQC`.

# In[ ]:


test_features.append(["PoolArea", "PoolQC"])


# ### 4.6. Miscellaneous <a id='miscellaneous'></a>
# Similar to Pool, miscellaneous features don't contain much information. However, different from Pool, we don't have any intuitive reasoning for these miscellaneous features' practical importance. It's likely we will drop them.

# In[ ]:


fig, axes = multiple_violinplots(["MiscFeature"], ncol=1)


# In[ ]:


corr_heatmap(["MiscVal"], figsize=(3.5,3))


# In[ ]:


test_features.append(["MiscFeature", "MiscVal"])


# ### 4.7. Porch <a id='porch'></a>

# In[ ]:


porch = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
print("In the train set, there are:")
for p in porch:
    print("\t{}/{} samples with {}".format(sum(train[p]>0), len(train), p))
train["PorchSF"] = train[porch].sum(axis=1)
fig = corr_heatmap(porch + ["PorchSF"], figsize=(6,5))


# In[ ]:


pairplot(porch + ["PorchSF"], kwargs={"height":1.5})


# The majority of samples either has only open porch or doesn't have a porch. However, different from Pool and Kichen, the number of samples is still large enough not to be ignored. Therefore we will keep all of these 4 features (and not reproduce the overall `PorchSF`)

# ## 5. Other features <a id='other_features'></a>
# (that are not related to any other features in the set)

# In[ ]:


print(*nonrelated, sep=", ")
obj_nrlt = list(train[nonrelated].select_dtypes(object).columns)
num_nonrlt = list(train[nonrelated].select_dtypes(np.number).columns)


#  ### 5.1. Other - Categorical Features <a id='other_categorical_features'></a>
#  Lots of things about these variables can be infered from their [data description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). With a quick look into it, here are few things we can notice immediately:
# - `MSSubClass` seems to be a combination of `HouseStyle`, `BldgType` and `YearBuilt`, so let's drop it to avoid noise.
# - Many of the features are ordinal (`LotShape`, `LandSlope`, `PavedDrive` etc.) that can be converted to numerical values.
# - `Condition1` and `Condition2` can be combined and converted to 4 features of approximity as follows
#     - `Str`: Feedr=1, Artery=2, otherwise=0
#     - `NSRR`: RRNn=1, RRAn=2, otherwise=0
#     - `EWRR`: RRNe=1, RRAe=2, otherwise=0
#     - `Offsite`: PosN=1, PosA=2, otherwise=0<br>
#     <br>
#     One thing to note is that `Utilities` and `Electrical` are heavily skewed in the train set. Might make sense to drop them later. We'll see how they perform during model training.

# In[ ]:


print(train.Utilities.value_counts(), end="\n\n")
print(train.Electrical.value_counts())


# The remaining features will be one-hot encoded. Check out the Feature Engineering part for more details.

# ### 5.2. Numerical features <a id='other_numerical_features'></a>

# In[ ]:


train["2floorsSF"] = train["1stFlrSF"] + train["2ndFlrSF"]
train["TotalArea"] = train[["1stFlrSF", "2ndFlrSF", "TotalBsmtSF"]].sum(axis=1)
corr_heatmap(nonrelated + ["2floorsSF", "TotalArea"], figsize=(12,10))


# Things to note:
# - `1stFlrSF`+`2ndFlrSF` = `GrLivArea`. Let's drop them.
#     - Drop: `1stFlrSF`, `2ndFlrSF`
# - Surprisingly, `OverallCond` is almost non-correlated with sale prices. Based on the data description, I had expected it to be somewhat related to `OverallQual` and more-or-less predictive of `SalePrice`.
# - Test:
#     - `Utilities`, `Electrical`, `OverallCond`, `LowQualFinSF`
#     - `Age` vs. `YrSinceRemod`

# In[ ]:


test_features.append(
    ["Utilities", "Electrical", "OverallCond", "LowQualFinSF"]
)
test_features.append(["Age", "YrSinceRemod"])


# # III. Feature Engineering & Testing <a id='feature_engineering_testing'></a>
# In this part we are going to test the data transformation ideas we have proposed above using some basic regressor models. We will be using the following models to test how our feature processing and selections affect the score:
# - `GradientBoostingRegressor`
# - `ExtraTreesRegressor`
# - `RandomForestRegressor`
# - `XGBRegressor`
# - `RidgeCV`
# - `ElasticNetCV`
# - `LinearSVR`
# - `LGBMRegressor`

# *Note: in the code and previous version of this notebook, I also tried incorporating a simple feedforward neural network; however, without careful hyper-param tuning, the net often experienced gradient-exploding and was very unstable, so I've left it out. You can try adding it again, but it will need some tweaking to avoid gradient explosion and ensuring the score is not so off*

# In[ ]:


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    This class performs the data transformation steps
    we've discussed above.
    """
    Score_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "None":1, "Po":0}
    BsmtFin_map = {"GLQ":5, "ALQ":4, "BLQ":3, "Rec":2, "LwQ":1, "Unf":0, "None":0}
    BsmtEx_map = {"Gd":3, "Av":2, "Mn":1, "No":0, "None":0}
    GargFin_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
    GargType_map = {"BuiltIn": "Attchd", "Basment": "Attchd"
                    , "CarPort": "Detchd",}
    StreetAlley_map = {"Pave":2, "Grvl":1, "None":0}
    CentralAir_map = {"Y":1, "N":0}
    LotShape_map = {"Reg":0, "IR1":1, "IR2":2, "IR3":3}
    LandSlope_map = {"Gtl":0, "Mod":1, "Sev":2}
    Electrical_map = {"SBrkr":3, "FuseA":2, "FuseF":1, "FuseP":0}
    ElectricMix_map = {"Mix":1}
    Functional_map = {"Typ":5, "Min1":4, "Min2":4, "Mod":3
                      , "Maj1":2, "Maj2":2, "Sev":1, "Sal":0}
    PavedDrive_map = {"Y":2, "P":1, "N":0}
    Fence_map = {"GdPrv":4, "MnPrv":3, "GdWo":2, "MnWw":1, "None":0}
    CondStr_map = {"Feedr":1, "Artery":2}
    CondNSRR_map = {"RRNn":1, "RRAn":2}
    CondEWRR_map = {"RRNe":1, "RRAe":2}
    CondOff_map = {"PosN":1, "PosA":2}
    
    to_drop_ = ["MSSubClass", "GarageCond", "1stFlrSF", "2ndFlrSF"]
    test_cols_ = [["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"
                   ,"BsmtFinType1", "BsmtFinType2", "BsmtUnfExp"]
                  , ["AllBath", "BsmtFullBath", "BsmtHalfBath"
                     , "FullBath", "HalfBath"]
                  , ["GarageCars", "GarageArea"]
                  , ['KitchenAbvGr']
                  , ["PoolArea", "PoolQC"]
                  , ["MiscFeature", "MiscVal"]
                  , ["Utilities", "Electrical", "OverallCond", "LowQualFinSF"]
                  , ["Age", "YrSinceRemod"]
                 ]
    
    def __init__(self, execute=True
                 , cols_to_keep_or_drop=None
                 , keep_or_drop=None):
        """
        ``execute``: if True then ``_feature_engineer`` method,
        otherwise pass
        ``cols_to_keep_or_drop``: list of columns to be kept or dropped
        ``keep_or_drop``: if "keep" then keeping ``cols_to_keep_or_drop``
        columns, else if "drop" then dropping ``cols_to_keep_or_drop`` columns,
        otherwise pass
        """
        self.execute = execute
        self.cols_to_keep_or_drop = cols_to_keep_or_drop
        self.keep_or_drop=keep_or_drop
    
    def map_val(self, x, mp):
        return mp[x] if x in mp else 0
    
    def map_s(self, s, mp):
        return s.apply(lambda x: self.map_val(x, mp))
    
    def _feature_engineer(self, X):
        X = X.copy()
        # Score_map mapping
        for c in ["BsmtQual", "BsmtCond", "GarageQual"
                  , "HeatingQC", "KitchenQual", "FireplaceQu"
                  , "PoolQC", "ExterQual", "ExterCond"]:
            X[c] = self.map_s(X[c], self.Score_map)
        
        # Other mapping
        for c, mpping in (
            ["BsmtFinType1", self.BsmtFin_map]
            , ["BsmtFinType2", self.BsmtFin_map]
            , ["BsmtExposure", self.BsmtEx_map]
            , ["GarageFinish", self.GargFin_map]
            , ["Alley", self.StreetAlley_map]
            , ["Street", self.StreetAlley_map]
            , ["CentralAir", self.CentralAir_map]
            , ["LotShape", self.LotShape_map]
            , ["LandSlope", self.LandSlope_map]
            , ["Electrical", self.Electrical_map]
            , ["Functional", self.Functional_map]
            , ["PavedDrive", self.PavedDrive_map]
            , ["Fence", self.Fence_map]
        ):
            if c == "Electrical":
                X[c + "Mix"] = self.map_s(X[c], self.ElectricMix_map)
            X[c] = self.map_s(X[c], mpping)
        # GarageType
        # self.map_s works for string->int conversion only
        # and therefore won't work for GarageType
        X["GarageType"] = X.GarageType.replace(self.GargType_map)
        # Condition1 and Condition2
        for c, m in [
            ["Str", self.CondStr_map]
            , ["NSRR", self.CondNSRR_map]
            , ["EWRR", self.CondEWRR_map]
            , ["Offsite", self.CondOff_map]
        ]:
            cond1 = self.map_s(X.Condition1, m)
            cond2 = self.map_s(X.Condition2, m)
            X[c] = pd.concat([cond1, cond2], axis=1).max(axis=1)
        X.drop(columns = ["Condition1", "Condition2"], inplace=True)
            
        # Additional features
        X["BsmtHfBthDummy"] = np.where(X.BsmtHalfBath == 0, 0, 1)
        X["BsmtFinType_SF1"] = X.BsmtFinType1 * X.BsmtFinSF1
        X["BsmtFinType_SF2"] = X.BsmtFinType2 * X.BsmtFinSF2
        X["BsmtUnfExp"] = X.BsmtUnfSF * X.BsmtExposure
        X["AllBath"] = X.BsmtFullBath + X.FullBath +            (X.BsmtHalfBath + X.HalfBath)/2
        X["KitchenScore"] = X.KitchenQual * X.KitchenAbvGr  
        
        # Dropping unnecessary columns
        X.drop(columns = self.to_drop_, inplace=True)
        return X
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.execute:
            X = self._feature_engineer(X)
        if self.cols_to_keep_or_drop:
            if self.keep_or_drop == "keep":
                X = X[self.cols_to_keep_or_drop]
            elif self.keep_or_drop == "drop":
                X = X.drop(columns=self.cols_to_keep_or_drop, inplace=True)
        return X
    
class CustomizedOHE(BaseEstimator, TransformerMixin):
    """Perform one-hot encoding. This class always returns a dense pandas
    dataframe (sklearn's always returns np.array).
    
    ``columns``: "categorical" (transformer applied to all categorical
    features) or list of column names to be transformed.
    ``keep_header``: if True returns pd.DataFrame with
    column names, otherwise return np.array
    """
    def __init__(self, columns="categorical"
                 , OHE=OneHotEncoder(handle_unknown="ignore")
                 , keep_header=True):
        self.columns = columns
        self.OHE = OHE
        self.keep_header = keep_header
    
    def fit(self, X, y=None):
        self.header_ = list(X.columns)
        if self.columns == "categorical":
            self.transformed_cols_ = X.select_dtypes(object).columns
        else:
            self.transformed_cols_ = self.columns
        self.not_transformed_cols_ = [
            c for c in self.header_ if c not in self.transformed_cols_
        ]
        if self.keep_header:
            self.CT = ColumnTransformer(
                [("OHE", self.OHE, self.transformed_cols_)]
                , remainder="passthrough"
                , sparse_threshold=0
            )
            self.CT.fit(X)
            if len(self.transformed_cols_) > 0:
                self.transformed_cols_new_ =                     self.CT.transformers_[0][1].get_feature_names()
            else:
                self.transformed_cols_new_ = []
            self.header_new_ = np.concatenate(
                (self.transformed_cols_new_,
                 self.not_transformed_cols_)
            )
        else:
            self.CT = ColumnTransformer(
                [("OHE", self.OHE, self.transformed_cols_)]
                , remainder="passthrough"
            )
            self.CT.fit(X)
        return self

    def transform(self, X, y=None):
        transformed = self.CT.transform(X)
        self.a_ = transformed
        if self.keep_header:
            return pd.DataFrame(data=transformed, columns=self.header_new_)
        else:
            return transformed
        
class CustomizedScaler(BaseEstimator, TransformerMixin):
    """Performing standardization using sklearn's ``StandardScaler``
    transformer but allowing column names to be retained
    """
    def __init__(self, scaler=StandardScaler(), keep_header=True):
        self.scaler,self.keep_header = scaler, keep_header
    
    def fit(self, X, y=None):
        self.header = list(X.columns)
        self.scaler.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        transformed = self.scaler.transform(X)
        if self.keep_header:
            return pd.DataFrame(data=transformed, columns=self.header)
        else:
            return transformed


# In[ ]:


# Constants
ADD_NN = False # whether to use a feedforward neural network
ALPHAS = [0.1, 0.3, 1, 3, 10]
RS = 713
KFOLDS = 3
test_features = [['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'
                  ,'BsmtFinType1', 'BsmtFinType2', 'BsmtUnfExp']
                 , ['AllBath', 'BsmtHalfBath', 'BsmtFullBath'
                    , 'HalfBath', 'FullBath']
                 , ['GarageCars', 'GarageArea']
                 , ['KitchenAbvGr']
                 , ['PoolArea', 'PoolQC']
                 , ['MiscFeature', 'MiscVal']
                 , ['Utilities', 'Electrical', 'OverallCond', 'LowQualFinSF']
                 , ['Age', 'YrSinceRemod']]

# Functions to get transformer pipeline and estimators
# For estimators, their hyperparams are chosen arbitrarily 
# and not selected for optimization
def get_pipe(additional_transform=[]):
    transform = [("PreI", PreProcessor(old_new, num_to_obj))
                 , ("RI", RelationImputer(related = RELATED))]
    transform += additional_transform
    return Pipeline(transform)

def get_GBR():
    return GradientBoostingRegressor(n_estimators=500
                                     , max_depth=2
                                     , max_features=0.8
                                     , random_state=RS)

def get_ETR():
    return ExtraTreesRegressor(n_estimators=700
                               , max_depth=3
                               , max_features=0.8
                               , random_state=RS)

def get_RFR():
    return RandomForestRegressor(n_estimators=700
                                 , max_depth=3
                                 , max_features=0.8
                                 , max_samples=0.8
                                 , random_state=RS)

def get_XGBR():
    return XGBRegressor(n_estimators=500
                        , learning_rate=0.1
                        , max_depth=3
                        , colsample_bytree=0.8
                        , n_jobs=-1, random_state=RS)

def get_Ridge():
    return RidgeCV(alphas=ALPHAS, normalize=False, store_cv_values=True)

def get_Enet():
    return ElasticNetCV(l1_ratio=0.5
                        , alphas=ALPHAS
                        , normalize=False
                        , max_iter=1000
                        , random_state=RS
                       )

def get_NN(hidden_layers=1, compiled=True):
    NN = Sequential()
    regu = regularizers.l1_l2(l1=0.01, l2=0.01)
    for _ in range(hidden_layers):
        NN.add(layers.Dense(100
                            , activation="relu"
                            , kernel_regularizer=regu
                            , bias_regularizer=regu))
    NN.add(layers.Dense(1))
    if compiled:
        optimizer = optimizers.SGD(0.001)
        NN.compile(optimizer=optimizer, loss="mse", metrics=["MSE"])
    return NN

def get_SVM():
    return LinearSVR(random_state=RS, max_iter=1000)

def get_LGBR():
    return LGBMRegressor(max_depth=3, n_estimators=500
                         , random_state=RS, n_jobs=-1)

estimators = [get_GBR, get_ETR, get_RFR, get_XGBR, get_Ridge
              , get_Enet, get_SVM, get_LGBR]
if ADD_NN:
    estimators.add(get_NN)

# Neural Network's fitting hyperparameters
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5)
NN_kwargs = dict(epochs=100
                 , batch_size=10
                 , validation_split=0.2
                 , verbose=0
                 , callbacks=[early_stop])

# Functions to prepare visualization output
def RMSE(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))
# rmse_scorer = make_scorer(RMSE)

def reload_xy(msg=False):
    df = reload(msg=msg)
    X = df[[c for c in df.columns if c != "SalePrice"]].copy()
    y = df["SalePrice"].copy()
    return X, np.array(np.log1p(y))
X, y_trans = reload_xy()

def new_output(bl):
    cols = ["Est", bl, "Score_mean", "Score_delt(%)"
            , "Score_std", "Time_run", "Time_delt(%)"]
    return pd.DataFrame(columns=cols)

def fit_timer(get_est, X, y, params={}):
    e = get_est()
    t = dt.now()
    e.fit(X, y, **params)
    return timer(t)
        
def scoring(get_est, additional_transform
            , X=X, y=y_trans, scorer=RMSE, splitter=KFold, cv=KFOLDS):
    scores = []
    splitter = splitter(cv, shuffle=True, random_state=713)
    kw = {}
    if get_est.__name__ == 'get_NN':
        kw = NN_kwargs
    for tr, te in splitter.split(X):
        X_tr, X_te = X.iloc[tr, :], X.iloc[te, :]
        y_tr, y_te = y[tr], y[te]
        p = get_pipe(additional_transform)
        e = get_est()
        X_tr = p.fit_transform(X_tr)
        X_te = p.transform(X_te)
        e.fit(X_tr, y_tr, **kw)
        y_pre = e.predict(X_te)
        scores.append(RMSE(y_te, y_pre))
    pipe = get_pipe(additional_transform)
    X_trans = pipe.fit_transform(X)
    t = fit_timer(get_est, X_trans, y_trans, params=kw)
    return scores, t

def new_row(df, tactics, i, get_est, bl, scores, time):
    scr_m, scr_std = np.mean(scores), np.std(scores)
    est = get_est()
    if (i % len(tactics)) == 0:
        scr_d, time_d = np.nan, np.nan
    else:
        last_scr, last_t = df.loc[i-1, ["Score_mean", "Time_run"]]
        scr_d = (scr_m - last_scr) / (last_scr) * 100
        time_d = (t - last_t) / (last_t) * 100
    output.loc[i] = [
        type(est).__name__, bl, scr_m, scr_d, scr_std, time, time_d
    ]
    
def timer(t):
    t = dt.now() - t
    return t.seconds + t.microseconds/1e6

def for_loop(bl, estimators=estimators):
    l = list(product(estimators, bl))
    return tqdm(l, position=0)


# As shown below, the feature engineering suggestions prove effective in improving test score (lower score is better) in all 8 models.. With few dimensions (fewer categorical variables), training time also improves significantly. 

# In[ ]:


# Comparison & Output
output = new_output("Feature_Engineer?")
i = 0
tactics = [False, True]
for get_est, ex in for_loop(tactics):
    add_trans = [("FE", FeatureEngineer(execute=ex))
             , ("CusOHE", CustomizedOHE())
             , ("Stdize", CustomizedScaler(keep_header=False))]
    scores, t = scoring(get_est, add_trans)
    new_row(output, tactics, i, get_est, ex, scores, t)
    i += 1
output.round(3)


# One thing to note is that the tree-based models took approximately 3-6 seconds to run (using CPU only). That is quite concerning given the fact that there are less than 1500 training samples. When we perform model selection and hyperparameter optimization, the running time is going to scale remarkably. It's better if we can somehow reduce the complexity of the models and save that precious training time, and one way is to **continue cutting down on number of features.**<br>
# <br>
# Among the 84 features resulted from our feature engineering decisions, not all of them will be useful. Let's see if there are redundant features we can throw away. We will look at feature importances in the first 4 tree-based models mentioned above. It should be noted that this is not the optimal method (sometimes even misleading), so to be careful we will average the feature's importance scores across 4 models.

# In[ ]:


# Less first re-train our ensemble models
add_trans = [("FE", FeatureEngineer())
             , ("CusOHE", CustomizedOHE())
             , ("Stdize", CustomizedScaler())]
pipe = get_pipe(add_trans)
X_trans = pipe.fit_transform(X, y_trans)
# Function to return original features' names
def col_name(col, ohe_cols=pipe.steps[3][1].transformed_cols_):
    mat = re.match(r"(^x\d{1,2})_(.+)", col)
    if mat:
        return ohe_cols[int(mat.group(1)[1:])]
    else:
        return col
    
get_ests = [get_GBR, get_ETR, get_RFR, get_XGBR]
feat_imp = pd.DataFrame(
    columns = ["OHE_feat", "Features"] 
    + [type(e()).__name__ for e in get_ests] 
    + ["Average"]
)
feat_imp.OHE_feat = list(X_trans.columns)
feat_imp.Features = feat_imp.OHE_feat.apply(lambda x: col_name(x))
for get_est in get_ests:
    est = get_est()
    est.fit(X_trans, y_trans)
    feat_imp[type(est).__name__[:-9]] = est.feature_importances_
feat_imp = feat_imp.groupby(["Features"]).sum().reset_index()
feat_imp["Average"] = feat_imp.iloc[:,[1,2,3,4]].mean(axis=1)
feat_imp = feat_imp.sort_values(by="Average", ascending=False)
feat_imp = feat_imp.reset_index(drop=True)

# Feature importance examination
n = len(feat_imp)
all_columns = list(feat_imp.Features)
weak_cols = list(feat_imp[feat_imp.Average < (1/n)].Features)
insig_cols = list(feat_imp[
    (feat_imp.iloc[:,1:5] > (1/n)).sum(axis=1) == 0
].Features)
zero_cols = list(feat_imp[feat_imp.Average == 0].Features)

print(f"Features with AVERAGE importance score < (1/n_features):    {len(weak_cols)}/{n}")
print(*weak_cols[:5], sep=", ", end=", etc.\n\n")
print(f"Features with ALL importance scores < (1/len(features)):    {len(insig_cols)}/{n}")
print(*insig_cols[:5], sep=", ", end=", etc.\n\n")
print(f"Features with 0 importance scores: {len(zero_cols)}/{n}")
print(*zero_cols, sep=", ")

feat_imp.to_csv("Column_importances.csv", index=False)
a = feat_imp.iloc[:25,:]
ax = a.plot(y=[1,2,3,4], kind="line", figsize=(13,7), rot=90)
a.plot(x="Features", y="Average"
       , kind="bar", ax=ax, color="lightsteelblue"
       , title="Top 25 Most Important Features")


# Next, we will look at the features that we deemed necessary to be tested earlier.

# In[ ]:


print(f"Average importance: {round(1/n,3)}")
for c in test_features:
    print("\n" + "="*20)
    print(feat_imp[feat_imp.Features.isin(c)].round(3))


# We can draw the following conclusions from above:
# - `BsmtUnfExp` is redundant and should not be created.
# - `BsmtUnfSF`, `BsmtFinType2` and `BsmtFinSF2` should be removed.
# - Should drop `BsmtHalfBath`, `BsmtFullBath`, `HalfBath`
# - Keep both garage area features
# - `KitchenAbvGr` can be dropped
# - `PoolQC` should be dropped
# - `MiscFeature` and `MiscVal` should be dropped
# - `Electrical`, `LowQualFinSF` and `Utilities` should be dropped
# - Keep both `Age` and `YrSinceRemod`

# In[ ]:


class ColumnDropper(BaseEstimator, TransformerMixin):
    col_imp = pd.read_csv("/kaggle/working/Column_importances.csv")
    all_cols = list(col_imp.Features)
    n = len(all_cols)
    weak_cols = list(col_imp[col_imp.Average < (1/n)].Features)
    insig_cols = list(col_imp[
        (col_imp.iloc[:,1:5] > (1/n)).sum(axis=1) == 0
    ].Features)
    insig2_cols = list(col_imp[
        (col_imp.iloc[:,1:5] > (1/n/10)).sum(axis=1) == 0
    ].Features)
    zero_cols = list(col_imp[col_imp.Average == 0].Features)
    
    def __init__(self, strategy=None, others=None):
        """ ``strategy``: int, "weak", "insignificant" or "zero".
        Default None.
            ~ int then drop all columns ranked (starting from 0)
                ``cols_drop`` or below
            ~ "weak" then drop all weak_cols.
            ~ "insignificant" then drop all insig_cols.
            ~ "insignificant2"
            ~ "zero" then drop all zero_cols
        
        ``others``: list of other columns to be dropped
        """
        self.strategy = strategy
        self.others = others
        if isinstance(strategy, (int,np.integer)):
            self.columns = self.all_cols[strategy:]
        elif strategy == "weak":
            self.columns = self.weak_cols
        elif strategy == "insignificant":
            self.columns = self.insig_cols
        elif strategy == "insignificant2":
            self.columns = self.insig2_cols            
        elif strategy == "zero":
            self.columns = self.zero_cols
        elif strategy is None:
            self.columns = []
        else:
            raise Exception(f"Wrong 'strategy' input: \"{strategy}\"")
        if others:
            self.columns += others
            self.columns = list(set(self.columns))
    
    def fit(self, X, y=None):
        if X.shape[1] != self.n:
            msg = f"Wrong number of features: expecting {self.n}, got {X.shape[1]}"
            raise Exception(msg)
        missing_cols = [c for c in X.columns if c not in self.all_cols]
        if len(missing_cols) > 0:
            msg = "Columns not existing: " + ", ".join(missing_cols)
            raise Exception(msg)
        else:
            return self
    def transform(self, X, y=None):
        if len(self.columns) > 0:
            X = X.drop(columns=self.columns)
        return X


# Let's test and see if our decisions will improve the scores.

# In[ ]:


cols_to_drop = ["BsmtUnfExp", "BsmtUnfSF", "BsmtFinType2"
                , "BsmtFinSF2", "BsmtHalfBath", "BsmtFullBath"
                , "HalfBath", "KitchenAbvGr", "PoolQC"
                , "MiscFeature", "MiscVal", "Electrical"
                , "LowQualFinSF", "Utilities"]

output = new_output("Drop?")
i = 0
tactics = [None, cols_to_drop]
for get_est, drop in for_loop(tactics):    
    add_trans = [("FE", FeatureEngineer())
                 , ("Drop", ColumnDropper(others=drop))
                 , ("CusOHE", CustomizedOHE())
                 , ("Stdize", CustomizedScaler(keep_header=False))]
    scores, t = scoring(get_est, add_trans)
    d = True if drop else False
    new_row(output, tactics, i, get_est, d, scores, t)
    i += 1
output.round(3)


# As we can see, the mean scores (lower is better) all improve slightly. More importantly, fitting time (lower is better) also shows great improvement. Therefore, we will drop the columns mentioned above.<br>
# <br>
# **Is it possible to drop more columns and shorten running time even further?**<br>
# <br>
# We may try dropping all 0-importance columns, or we can even think about keeping only top 10, top 20 most important features, etc. There is an **abundant** number of options for us to explore (2^`n_features` to be exact), and it's impossible to test all of them.<br>
# <br>
# To find the number of features/which specific columns to drop, I will perform a simple method: add features one by one, from the most to the least important, and observe how the `RMSE` score and training time change.<br>
# <br>
# Note that this might not be the best method to select features as any two features can interact in some way that we don't know. It, however, should serve as a good guess for us to grasp (hopefully) some of the most predictive variables from the dataset.

# In[ ]:


scrs = {}
time = {}
tactics = np.arange(84, 0, -1)
for get_est, tac in for_loop(tactics):
    dropper = ColumnDropper(strategy=tac, others=cols_to_drop)
    add_trans = [("FE", FeatureEngineer())
                 , ("Drop", dropper)
                 , ("CusOHE", CustomizedOHE())
                 , ("Stdize", CustomizedScaler(keep_header=False))]
    scores, t = scoring(get_est, add_trans)
    est_name = type(get_est()).__name__
    if est_name in scrs:
        scrs[est_name].append(np.mean(scores))
        time[est_name].append(t)
    else:
        scrs[est_name] = [np.mean(scores)]
        time[est_name] = [t]


# In[ ]:


PLOT_CHANGE = False
def running_delt(l, transform=PLOT_CHANGE):
    pre = np.nan
    o = []
    for nex in l:
        o.append((nex-pre)/pre)
        pre = nex
    return o
plt.style.use("seaborn-whitegrid")
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(14,10))
est_names = [type(e()).__name__ for e in estimators]

colors = ["b", "g", "r", "c", "m", "olive", "k", "darkorange", "saddlebrown"]
for n, c in zip(est_names, colors):
    ax1.plot(tactics, scrs[n], c, label=n)
    ax2.plot(tactics, time[n], c=c)
ax1.set_ylabel("Mean score")
ax2.set_xlabel("Features being kept")
ax2.set_ylabel("Train time")
ax1.legend()
plt.show()


# Generally, dropping variables always significantly shortens training time, but removing too much information will hurt model's performance. Based on the graphs above:
# - Most models stop improving from after 20 features are added.
# - Our A-players, including `GradientBoostingRegressor`, `XGBRegressor` and `LGBMRegressor` continue to improve as more features are added, up until around the 40-th most important feature. After that, they are mostly noises.
# - Running time increases roughly proportionally with number of features for most models, as expected.

# Given these observations, we will drop the bottom half of the features (meaning that we keep only top 42/84 features).

# In[ ]:


DROP = 42 # Keeping only top ``DROP`` most important features
train = reload(msg=False)
X_train = train[[c for c in train.columns if c != "SalePrice"]].copy()
y_train = train["SalePrice"].copy()
X_test = reload("test.csv", msg=False)
add_trans = [("FE", FeatureEngineer())
    , ("Drop", ColumnDropper(strategy=DROP, others=cols_to_drop))
    , ("CusOHE", CustomizedOHE())]
pipe = get_pipe(add_trans)
X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)

dropped_cols = pipe.steps[3][1].columns
print(f"Columns dropped: {len(dropped_cols)}/84")
print("Examples: " + ", ".join(dropped_cols[:6]) + ", etc.")

submission = reload("sample_submission.csv", dropped_columns=[], msg=False)
est = get_XGBR()
est.fit(X_train, np.log1p(y_train))
submission["SalePrice"] = np.expm1(est.predict(X_test))


# In[ ]:


X_train["SalePrice"] = train["SalePrice"]
submission.to_csv("Submission.csv", index=False)
X_train.to_csv("train_preprocessed.csv", index=False)
X_test.to_csv("test_preprocessed.csv", index=False)

