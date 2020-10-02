#!/usr/bin/env python
# coding: utf-8

# In this notebook, I focus on imputing the `Null`s in the dataset. Different from most of other publicly available notebooks which either ignore features/samples that contain `Null`s or use [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html), I took a little further step and examined the relationship between features.
# <br><br>
# ### Table of Content
# ...<a href='#Loading'>Package & Data Loading</a><br>
# ...<a href='#QuickLook'>I. A Quick Look</a><br>
# ...<a href='#RelationImputer'>II. RelationImputer</a><br>
# ......<a href='#Logic_RelationImputer'>1. Logic</a><br>
# ......<a href='#Example_RelationImputer'>2. Example</a><br>
# ......<a href='#ActualMissing_RelationImputer'>3. Actual Missing Values</a><br>
# ...<a href='#Testing'>III. Testing</a><br>

# # Package & Data Loading <a id='Loading'></a>

# In[ ]:


from copy import deepcopy
from datetime import datetime as dt
from itertools import product
import math
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def reload(df="train.csv", dropped_columns=['Id']): # set dropped_columns = [] if want to keep Id column
    data_path = "../input/house-prices-advanced-regression-techniques/"
    if df == "all_data":
        train = pd.read_csv(data_path + "train.csv")
        test = pd.read_csv(data_path + "test.csv")
        data = train.drop(columns="SalePrice").append(test)
    else:
        data = pd.read_csv(data_path + df)
    print(df + " loaded successfully!")
    return data.reset_index(drop=True).drop(columns=dropped_columns)

train = reload()
test = reload("test.csv", dropped_columns=[])
all_data = reload("all_data")


# # I. A Quick Look <a id='QuickLook'></a>
# Let's first take a glance at the data.

# In[ ]:


print("train shape: {}".format(train.shape))
print("test shape: {}\n\n".format(test.shape))

for df, name in zip([train, test], ["train", "test"]):
    print("There are {} features with Nulls in the {} set.".format(df.isna().any().sum(), name))

null_counts = all_data.isna().sum()
print("\nFeatures with Nulls:")
print(*list(null_counts[null_counts > 0].index), sep=", ")

print("\n" + "="*20 +"\nNull ratio in tran set:\n")
null_cols = train.isna().sum() / len(train)
print(null_cols[null_cols > 0].sort_values())


# Most will choose to remove *FireplaceQu*, *Fence*, *Alley*, *MiscFeature* and *PoolQC* because they all have large amount of missing values. For the rest, one may be quick to point out that their `Null`s actually imply that a sample doesn't have a particular feature (e.g. `Null` *GarageType* means no basement). Therefore, we can use [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) and replace the `Null`s with a constant (e.g. "missing_value" for categorical `Null`s and 0 for numerical ones).<br>
# <br>
# However, that method doesn't apply to all cases. For example, we know that all properties must have some degree of flatness, and therefore `Null` *LandContour* is actually a missing value. Even for the case of *GarageType*, if all other garage-related features (*GarageYrBlt*, *GarageFinish*, *GarageCars*, etc.) are not missing, a `Null` *GarageType* is also likely to be erroneous. Let's take a look at the examples below.

# In[ ]:


garage = ['GarageType', 'GarageYrBlt', 'GarageFinish'
          , 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
index_nonerror = all_data[garage].isin([0, np.nan]).sum(axis=1) >= len(garage)
print("a) Nulls that most likely represent missing properties (non-erroneous)")
all_data.loc[index_nonerror, garage].head()


# In[ ]:


indx_nonerror = all_data[garage].isin([0, np.nan]).sum(axis=1)
indx_nonerror = (indx_nonerror < len(garage)) & (indx_nonerror > 0)
print("b) Nulls that are actually missing data (erroneous)")
all_data.loc[indx_nonerror, garage].head()


# For this dataset, being able to tell which `Null` represents property-missing and which one is value-missing will much benefit imputation process. Here, one may wonder why we don't use Scikit-Learn's [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html). The reason is that `IterativeImputer` doesn't support imputing categorical (classification) and numerical features (regression) at the same time. There is already a [request](https://github.com/scikit-learn/scikit-learn/issues/17087) for such ability, but as of now (scikit-learn 0.23.1) it has not been available (in the same thread someone suggests [this package](https://pypi.org/project/autoimpute/) which is said to be able to perform mixed-dtype imputation, but I personally have not had a chance to test it).<br>
# <br>
# Having said that, I decided to take a (slightly) further step by making an imputer that works like `SimpleImputer` but takes into account features' "nullability".
# # II. RelationImputer <a id='RelationImputer'></a>
# In the following section, I made an imputer called `RelationImputer` that attempts to impute "erroneous" (representing missing values) and "non-erroneous" (implying the lack of a property) `Null`s separately. The logic is described in section 1 below. You can skip this part and go to section 2 where I provide an example of how the imputer works.
# ## 1. Logic <a id='Logic_RelationImputer'></a>
# In `RelationImputer`, the features need to be manually separated into two types:
# - Nullable: features whose `Null`s **may** not be erroneous and, instead, due to the lack of a property (*GarageType*, *TotalBsmtSF*, etc.).
# - Non-nullable: features whose `Null`s **always** represents missing values (*HouseStyle*, *LandContour*, etc.).<br>
# <br>
# (these names are of course unofficial so feel free to drop in any suggestion you have)

# `RelationImputer` imputes a Non-nullable feature in a way that is mostly similar that of `SimpleImputer`. On the other hand, for a Nullable feature, `RelationImputer` will consider other Nullables that are related to the feature (thus the name) to determine if the feature's `Null` is erroneous or not. Specifically:
# - If the number of `Null`s among those related Nullables is less than a specified threshold (which is by default equal to the number of the features themselves) then the `Null` being examined is an actually erroneous and missing data point.
# - Otherwise, the `Null` is considered non-erroneous

# Erroneous `Null`s are imputed similarly to Non-nullable while non-erroneous `Null`s will be labeled differently.

# In[ ]:


class RelationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, related, threshold=None, strategy="mean", fill_value=None
                 , missing_num=np.nan, missing_obj=np.nan
                 , final_num=0, final_obj="missing_value"):
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
            Return indexes of "actual" nulls.
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


# ## 2. Example <a id='Example_RelationImputer'></a>
# Here I provide an example for how `RelationImputer` works.<br>
# <br>
# Below is an example dataset. Columns *c1*, *c2* and *c3* are Nullable and *c4* is Non-nullable. In addition, *c1* and *c2* are related (i.e. they are all about a property of the samples).

# In[ ]:


# Here is an example
test_df = pd.DataFrame({'c1':[np.nan,2,0,0,5,5,7,33]
                        , 'c2':['a','b',np.nan,'d','a',np.nan,'g',np.nan]
                        , 'c3':[np.nan,13,np.nan,64,-999,np.nan,713, np.nan]
                        , 'c4':[99,100,np.nan,102,103,0,105,33]
                       })
print("Original dataframe:")
test_df


# Based on the logic of `RelationImputer`:
# - `Null` *c1* of sample 1, and `Null` *c2* of sample 5 and sample 7 are actual missing values and should be imputed.
# - `Null` *c2* of sample 2 is non-erroneous and should be labeled differently (in this case it's "missing_value" for categorical and 0 for numerical feature). 
# - All `Null`s in *c4* should be missing values and therefore imputed.<br>
# <br>
# Below is the actual imputed result.

# In[ ]:


test_related = [['c1','c2'],['c3']]
test_RI = RelationImputer(test_related)
test_RI.fit(test_df)
print("Impute value of each column:\n{}\n".format(test_RI.statistics_.sort_index()))
print("Transformed dataframe:")
test_RI.transform(test_df)


# The imputer performs as expected!
# ## 3. Actual Missing Values <a id='ActualMissing_RelationImputer'></a>
# Here I use `RelationImputer` to see how many actual `Null`s are there in the train set (by setting impute strategy `strategy`="constant" and imputed value `fill_value`=`np.nan` so all `Null`s stay the same).

# In[ ]:


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
           , ["GarageType", "GarageYrBlt", "GarageFinish" # Garage
              , "GarageCars", "GarageArea", "GarageQual", "GarageCond"]
           , ["MasVnrType", "MasVnrArea"] # Masonry Veneer
           , ["PoolArea", "PoolQC"] # Pool
           , ["Fence"] # Fence
           , ["MiscFeature", "MiscVal"] # Other miscellaneous
            ]
RI = RelationImputer(RELATED, strategy="constant", fill_value=np.nan)
true_nulls = RI.fit_transform(train)
true_nulls = true_nulls.isna().sum(axis=0) / len(train)
true_nulls[true_nulls > 0]


# As we can see, only *LotFrontage* seems to have a lot of actual missing data and possibly should be dropped during feature selection (though it should be noted that for features with significant portion being `Null` that we see earlier such as *PoolQC*, most of their values will be the same - "missing_value" in this case - so eventually they may get dropped anyway). 

# # III. Testing <a id='Testing'></a>
# In this section, I do a quick comparison test between different `SimpleImputer`-like imputing ways. The three imputing way I'm testing are:
# - Simple-imputing (with `SimpleImputer`): all `Null`s are replaced by their features' statistics (mean, median, mode).
# - Relation-imputing (with `RelationImputer`): logic as mentioned above.
# - Mixed (with `SimpleImputer`): This method is kind of a mixed between the two logics above. Specifically, `Null`s in selected columns (that are possible to have "non-error" NULLs") will be replaced with some constants, while the rest will be SimpleImpute-ed.

# ### Some preprocessing:
# Based on [data description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), *MSSubClass* is numerical but actually categorical (without any information regarding order of its values), so it will be handled using one-hot encoding. In addition, time-related features will be converted their respective numerical time-interval values as follows:
# - *YearBuilt*: convert to *Age* (*MoSold*/*YrSold* - *YearBuilt*)
# - *YearRemodAdd*: convert to *YrSinceRemod* (*MoSold*/*YrSold* - *YearRemodAdd*)
# - *GarageYrBlt*: convert to *YrSinceGarageBlt* (*MoSold*/*YrSold* - *GarageYrBlt*)
# - *MoSold*, *YrSold*: convert to categorical features.

# In[ ]:


# Pre-process data before imputation
class PreImputation(BaseEstimator, TransformerMixin):
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

def change_nullables(cols, map_dict):
    return [c if c not in map_dict.keys() else map_dict[c] for c in cols]
DATE_TO_YRS = {"YearBuilt":"Age"
               , "YearRemodAdd":"YrSinceRemod"
               , "GarageYrBlt": "YrSinceGarageBlt"}
RELATED = list(map(lambda x: change_nullables(x, DATE_TO_YRS), RELATED))

NUM_TO_OBJ = ["MSSubClass", "MoSold", "YrSold"]
# CAT_COLS = list(all_data.select_dtypes(object).columns) + NUM_TO_OBJ
CAT_COLS=['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
       'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
NUM_COLS = [col for col in all_data.columns if col not in CAT_COLS]
NUM_COLS = [col 
            if col not in DATE_TO_YRS.keys() 
            else DATE_TO_YRS[col] 
            for col in NUM_COLS]
IMPUTE_STRAT = "most_frequent"
FINAL_NUM = 0
FINAL_OBJ = "None"
L1_RATIO = 0.5
ALPHAS = [0.1, 0.3, 1, 3, 10]


# In[ ]:


# estimators
PreI = PreImputation(DATE_TO_YRS, NUM_TO_OBJ)
Ridge = RidgeCV(alphas=ALPHAS, normalize=True, store_cv_values=True)
Enet = ElasticNetCV(l1_ratio=L1_RATIO
                    , alphas=ALPHAS
                    , normalize=True
                    , max_iter=5000
                    , random_state=713
                   )
GBR = GradientBoostingRegressor(n_estimators=500
                                , min_samples_split=5
                                , min_samples_leaf=2
                                , max_depth=2
                               )
estimators = {"Ridge": Ridge
              , "Elastic Net": Enet
              , "Gradient Boosting": GBR
             }

# Simple-imputing
cat_SI_OHE = Pipeline(
    [("cat_Imputation", SimpleImputer(strategy="most_frequent"))
     , ("cat_OHE", OneHotEncoder(handle_unknown="ignore"))]
    )
num_SI_OHE = Pipeline(
    [("num_Imputation", SimpleImputer(strategy=IMPUTE_STRAT))]
    )
SI_imp_OHE = ColumnTransformer(
    [("cat_SI_OHE", deepcopy(cat_SI_OHE), CAT_COLS)
     , ("num_SI_OHE", deepcopy(num_SI_OHE), NUM_COLS)]
    , remainder="passthrough"
    )
SI_transform = [("PreImputation", deepcopy(PreI))
                , ("Imputation_OHE", deepcopy(SI_imp_OHE))
               ]

# Relation-imputing
RI = RelationImputer(RELATED, strategy=IMPUTE_STRAT
                     , final_num=FINAL_NUM, final_obj=FINAL_OBJ)
RI_OHE = ColumnTransformer(
    [('OHE', OneHotEncoder(handle_unknown="ignore"), CAT_COLS)]
    , remainder="passthrough"
    )
RI_transform = [("PreImputation", deepcopy(PreI))
                , ("Imputation", deepcopy(RI))
                , ("OHE", deepcopy(RI_OHE))
               ]

# Mixed-imputing
# Here we have something that is kind of in-between
# the two logics above - SimpleImputer and RelationImputer
#
# All columns in RELATED list will be imputed with strategy="constant"
# while the rest is imputed with strategy="mean" (numerical) and
# "most_frequent" (categorical)

RELATED_flat = [c for cs in RELATED for c in cs]
NUM_const, CAT_const = [
    [c for c in COLS if c in RELATED_flat] 
    for COLS in [NUM_COLS, CAT_COLS]
]
NUM_imp, CAT_imp = [
    [c for c in COLS if c not in RELATED_flat] 
    for COLS in [NUM_COLS, CAT_COLS]
]
MI_NUM_const = SimpleImputer(strategy="constant", fill_value=0)
MI_NUM_imp = SimpleImputer(strategy="mean")
MI_CAT_const = Pipeline(
    [("MI_CAT_const", SimpleImputer(strategy="constant", fill_value="missing_value"))
     , ("MI_CAT_OHE", OneHotEncoder(handle_unknown="ignore"))]
)
MI_CAT_imp = Pipeline(
    [("MI_CAT_imp", SimpleImputer(strategy="most_frequent"))
     , ("MI_CAT_OHE", OneHotEncoder(handle_unknown="ignore"))]
)
MI_imp_OHE = ColumnTransformer(
    [("NUM_const", deepcopy(MI_NUM_const), NUM_const)
     , ("NUM_imp", deepcopy(MI_NUM_imp), NUM_imp)
     , ("CAT_const", deepcopy(MI_CAT_const), CAT_const)
     , ("CAT_imp", deepcopy(MI_CAT_imp), CAT_imp)]
    , remainder="passthrough"
)
MI_transform = [("PreImputation", deepcopy(PreI))
                , ("Imputation_OHE", deepcopy(MI_imp_OHE))
               ]

transformers = {"SimpleImputer": SI_transform
                , "RelationImputer": RI_transform
                , "Mixed": MI_transform
               }


# In[ ]:


KFOLDS = 3
def RMSE(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))
rmse_scorer = make_scorer(RMSE)
X = train.drop(columns="SalePrice").copy()
y = train.SalePrice.copy()
y = np.log1p(y)

i = 0
output = pd.DataFrame({"Transformer": []
                       , "Estimator": []
                       , "Mean score": []
                       , "Std score": []
                      })
for e, t in tqdm(list(product(estimators, transformers))):
    trans = deepcopy(transformers[t])
    trans.append(("Estimator", estimators[e]))
    pipe = Pipeline(trans)
    scores = cross_val_score(
        pipe, X=X, y=y, scoring=rmse_scorer, cv=KFOLDS, n_jobs=-1
    )
    output.loc[i] = [t, e, np.mean(scores), np.std(scores)]
    i += 1
output


# 
# Actually there is not much difference between the three methods. In my next notebook, we will observe if `RelationImputer`makes a bigger difference when features are more carefully selected.

# In[ ]:


selected_pipeline = Pipeline(
    RI_transform + [("GBR", deepcopy(GBR))]
)
submission = pd.DataFrame()
submission["Id"] = test["Id"]
selected_pipeline.fit(X, y)
submission["SalePrice"]= selected_pipeline.predict(test.drop(columns="Id"))
submission["SalePrice"] = np.expm1(submission.SalePrice)
submission.to_csv("Submission.csv", index=False)
submission.head()

