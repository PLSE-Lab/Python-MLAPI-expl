# References
# https://www.kaggle.com/humananalog/house-prices-advanced-regression-techniques/xgboost-lasso
# https://www.kaggle.com/yadavsarthak/house-prices-advanced-regression-techniques/you-got-this-feature-engineering-and-lasso
# https://www.kaggle.com/juliencs/house-prices-advanced-regression-techniques/a-study-on-regression-applied-to-the-ames-dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew


def main():

    # Get data

    # train = pd.read_csv("train.csv.gz")
    # test = pd.read_csv('test.csv.gz')
    train = pd.read_csv("../input/train.csv.gz")
    test = pd.read_csv('../input/test.csv.gz')
    train_shape = train.shape
    print("train : " + str(train.shape))

    # Check for duplicates
    idsUnique = len(set(train.Id))
    idsTotal = train.shape[0]
    idsDupli = idsTotal - idsUnique
    print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")

    # Drop Id column
    # train.drop("Id", axis = 1, inplace = True)
    np.sum((train['SalePrice'].values % 10) > 0)


    # **Preprocessing**
    train = train[train[:train.shape[0]].GrLivArea < 4000]
    train_shape = train.shape
    
    # There seems to be 2 extreme outliers on the bottom right, really large houses that sold for really cheap. More generally, the author of the dataset recommends removing 'any houses with more than 4000 square feet' from the dataset.
    # Reference : https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

    # Log transform the target for official scoring
    # train.SalePrice = np.log1p(train.SalePrice)
    train.SalePrice = np.log(train.SalePrice)
    y = train.SalePrice

    test = pd.read_csv('test.csv.gz')
    test['SalePrice'] = train['SalePrice'].median()
    train = train.append(test)
    print(train.shape)

    train["YearBuilt"] = pd.to_datetime(train["YearBuilt"],format="%Y")
    train["YearRemodAdd"] = pd.to_datetime(train["YearRemodAdd"],format="%Y")
    train["YrSold"] = pd.to_datetime(train["YrSold"],format="%Y")


    # # Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.
    # # Handle missing values for features where median/mean or most common value doesn't make sense
    # # Alley : data description says NA means "no alley access"
    # train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
    # # BedroomAbvGr : NA most likely means 0
    # train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
    # # BsmtQual etc : data description says NA for basement features is "no basement"
    # train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
    # train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
    # train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
    # train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
    # train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
    # train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
    # train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
    # train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
    # # CentralAir : NA most likely means No
    # train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
    # # Condition : NA most likely means Normal
    # train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
    # train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
    # # EnclosedPorch : NA most likely means no enclosed porch
    # train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
    # # External stuff : NA most likely means average
    # train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
    # train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
    # # Fence : data description says NA means "no fence"
    # train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
    # # FireplaceQu : data description says NA means "no fireplace"
    # train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
    # train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
    # # Functional : data description says NA means typical
    # train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
    # # GarageType etc : data description says NA for garage features is "no garage"
    # train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
    # train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
    # train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
    # train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
    # train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
    # train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
    # # HalfBath : NA most likely means no half baths above grade
    # train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
    # # HeatingQC : NA most likely means typical
    # train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
    # # KitchenAbvGr : NA most likely means 0
    # train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
    # # KitchenQual : NA most likely means typical
    # train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
    # # LotFrontage : NA most likely means no lot frontage
    # train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
    # # LotShape : NA most likely means regular
    # train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
    # # MasVnrType : NA most likely means no veneer
    # train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
    # train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
    # # MiscFeature : data description says NA means "no misc feature"
    # train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
    # train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
    # # OpenPorchSF : NA most likely means no open porch
    # train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
    # # PavedDrive : NA most likely means not paved
    # train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
    # # PoolQC : data description says NA means "no pool"
    # train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
    # train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
    # # SaleCondition : NA most likely means normal sale
    # train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
    # # ScreenPorch : NA most likely means no screen porch
    # train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
    # # TotRmsAbvGrd : NA most likely means 0
    # train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
    # # Utilities : NA most likely means all public utilities
    # train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
    # # WoodDeckSF : NA most likely means no wood deck
    # train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)
    # train[train['Neighborhood'] == 'Greens']
    # Location based on the school location in the district
    train["lat"] = train.Neighborhood.replace({'Blmngtn' : 42.062806,
                                               'Blueste' : 42.009408,
                                                'BrDale' : 42.052500,
                                                'BrkSide': 42.033590,
                                                'ClearCr': 42.025425,
                                                'CollgCr': 42.021051,
                                                'Crawfor': 42.025949,
                                                'Edwards': 42.022800,
                                                'Gilbert': 42.027885,
                                                'GrnHill': 42.000854,
                                                'IDOTRR' : 42.019208,
                                                'Landmrk': 42.044777,
                                                'MeadowV': 41.991866,
                                                'Mitchel': 42.031307,
                                                'NAmes'  : 42.042966,
                                                'NoRidge': 42.050307,
                                                'NPkVill': 42.050207,
                                                'NridgHt': 42.060356,
                                                'NWAmes' : 42.051321,
                                                'OldTown': 42.028863,
                                                'SWISU'  : 42.017578,
                                                'Sawyer' : 42.033611,
                                                'SawyerW': 42.035540,
                                                'Somerst': 42.052191,
                                                'StoneBr': 42.060752,
                                                'Timber' : 41.998132,
                                                'Veenker': 42.040106})

    train["lon"] = train.Neighborhood.replace({'Blmngtn' : -93.639963,
                                               'Blueste' : -93.645543,
                                                'BrDale' : -93.628821,
                                                'BrkSide': -93.627552,
                                                'ClearCr': -93.675741,
                                                'CollgCr': -93.685643,
                                                'Crawfor': -93.620215,
                                                'Edwards': -93.663040,
                                                'Gilbert': -93.615692,
                                                'GrnHill': -93.643377,
                                                'IDOTRR' : -93.623401,
                                                'Landmrk': -93.646239,
                                                'MeadowV': -93.602441,
                                                'Mitchel': -93.626967,
                                                'NAmes'  : -93.613556,
                                                'NoRidge': -93.656045,
                                                'NPkVill': -93.625827,
                                                'NridgHt': -93.657107,
                                                'NWAmes' : -93.633798,
                                                'OldTown': -93.615497,
                                                'SWISU'  : -93.651283,
                                                'Sawyer' : -93.669348,
                                                'SawyerW': -93.685131,
                                                'Somerst': -93.643479,
                                                'StoneBr': -93.628955,
                                                'Timber' : -93.648335,
                                                'Veenker': -93.657032})
    # train["lon"] = preprocessing.scale(train["lon"])
    # train["lat"] = preprocessing.scale(train["lat"])

    # # from https://www.kaggle.com/amitchoudhary/house-prices-advanced-regression-techniques/script-v6
    # # IR2 and IR3 don't appear that often, so just make a distinction
    # # between regular and irregular.
    # train["IsRegularLotShape"] = (train["LotShape"] == "Reg") * 1

    # # The only interesting "misc. feature" is the presence of a shed.

    # # Months with the largest number of deals may be significant.
    # train["HighSeason"] = train["MoSold"].replace(
    #  {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

    # train["NewerDwelling"] = train["MSSubClass"].replace(
    #  {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
    #   90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

    # # Some numerical features are actually really categories
    # train = train.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45",
    #                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75",
    #                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120",
    #                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
    #                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
    #                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
    #                       })

    # # Encode some categorical features as ordered numbers when there is information in the order
    # train = train.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
    #                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    #                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
    #                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
    #                                          "ALQ" : 5, "GLQ" : 6},
    #                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
    #                                          "ALQ" : 5, "GLQ" : 6},
    #                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
    #                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
    #                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
    #                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    #                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
    #                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
    #                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    #                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    #                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    #                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    #                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
    #                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
    #                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
    #                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
    #                       "Street" : {"Grvl" : 1, "Pave" : 2},
    #                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
    #                      )


    # Then we will create new features, in 3 ways :
    #
    #  1. Simplifications of existing features
    #  2. Combinations of existing features
    #  3. Polynomials on the top 10 existing features
######################################

    # # Differentiate numerical features (minus the target) and categorical features
    # categorical_features = train.select_dtypes(include = ["object"]).columns
    # numerical_features = train.select_dtypes(exclude = ["object"]).columns
    # numerical_features = numerical_features.drop("SalePrice")
    # print("Numerical features : " + str(len(numerical_features)))
    # print("Categorical features : " + str(len(categorical_features)))
    # train_num = train[numerical_features]
    # train_cat = train[categorical_features]

    # # Handle remaining missing values for numerical features by using median as replacement
    # print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
    # train_num = train_num.fillna(train_num.median())
    # print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

    # # Log transform of the skewed numerical features to lessen impact of outliers
    # # Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
    # # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
    # skewness = train_num.apply(lambda x: skew(x))
    # skewness = skewness[abs(skewness) > 0.5]
    # print(str(skewness.shape[0]) + " skewed numerical features to log transform")
    # skewed_features = skewness.index
    # train_num[skewed_features] = np.log1p(train_num[skewed_features])

    # # Create dummy features for categorical values via one-hot encoding
    # print("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))
    # train_cat = pd.get_dummies(train_cat)
    # print("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))

    # # Join categorical and numerical features
    # train = pd.concat([train_num, train_cat], axis = 1)
    # print("New number of features : " + str(train.shape[1]))


    test = train[train_shape[0]:]
    train = train[:train_shape[0]]
    y = y[:train_shape[0]]
    
    test.to_csv("k_Ames_houses_test_ll.csv.gz",index=False,compression="gzip")
    train.to_csv("k_Ames_houses_train_ll.csv.gz",index=False,compression="gzip")
    # y.to_csv("k_Ames_houses_y_ll.csv.gz",index=False,compression="gzip")

    # return [train, y, numerical_features, test]
    return [train, y, test]
    
    

main()