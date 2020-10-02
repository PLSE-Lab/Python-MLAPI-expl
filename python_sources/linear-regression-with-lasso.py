import numpy as np # linear algebra
import pandas as pd # data processing, CSV file

from scipy.stats import skew

def create_features(all_data):
    # Handle missing values for features where median/mean or most common value doesn't make sense

    # Alley : data description says NA means "no alley access"
    all_data.loc[:, "Alley"] = all_data.loc[:, "Alley"].fillna("None")
    # BedroomAbvGr : NA most likely means 0
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
    # Fence : data description says NA means "no fence"
    all_data.loc[:, "Fence"] = all_data.loc[:, "Fence"].fillna("No")
    # FireplaceQu : data description says NA means "no fireplace"
    all_data.loc[:, "FireplaceQu"] = all_data.loc[:, "FireplaceQu"].fillna("No")
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
    # LotFrontage : NA most likely means no lot frontage
    all_data.loc[:, "LotFrontage"] = all_data.loc[:, "LotFrontage"].fillna(0)
    # LotShape : NA most likely means regular
    all_data.loc[:, "LotShape"] = all_data.loc[:, "LotShape"].fillna("Reg")
    # MasVnrType : NA most likely means no veneer
    all_data.loc[:, "MasVnrType"] = all_data.loc[:, "MasVnrType"].fillna("None")
    all_data.loc[:, "MasVnrArea"] = all_data.loc[:, "MasVnrArea"].fillna(0)
    # MiscFeature : data description says NA means "no misc feature"
    all_data.loc[:, "MiscFeature"] = all_data.loc[:, "MiscFeature"].fillna("No")
    all_data.loc[:, "MiscVal"] = all_data.loc[:, "MiscVal"].fillna(0)
    # OpenPorchSF : NA most likely means no open porch
    all_data.loc[:, "OpenPorchSF"] = all_data.loc[:, "OpenPorchSF"].fillna(0)
    # PavedDrive : NA most likely means not paved
    all_data.loc[:, "PavedDrive"] = all_data.loc[:, "PavedDrive"].fillna("N")
    # PoolQC : data description says NA means "no pool"
    all_data.loc[:, "PoolQC"] = all_data.loc[:, "PoolQC"].fillna("No")
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
                                           150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                           "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                       7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                          })

    # Encode some categorical features as ordered numbers when there is information in the order
    all_data = all_data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                           "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                           "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                           "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                           "Min2" : 6, "Min1" : 7, "Typ" : 8},
                           "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                           "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                           "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                           "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                           "Street" : {"Grvl" : 1, "Pave" : 2},
                           "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                         )

    # Create new features
    # 1* Simplifications of existing features
    all_data["SimplOverallQual"] = all_data.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                           4 : 2, 5 : 2, 6 : 2, # average
                                                           7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                          })
    all_data["SimplOverallCond"] = all_data.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                           4 : 2, 5 : 2, 6 : 2, # average
                                                           7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                          })
    all_data["SimplPoolQC"] = all_data.PoolQC.replace({1 : 1, 2 : 1, # average
                                                 3 : 2, 4 : 2 # good
                                                })
    all_data["SimplGarageCond"] = all_data.GarageCond.replace({1 : 1, # bad
                                                         2 : 1, 3 : 1, # average
                                                         4 : 2, 5 : 2 # good
                                                        })
    all_data["SimplGarageQual"] = all_data.GarageQual.replace({1 : 1, # bad
                                                         2 : 1, 3 : 1, # average
                                                         4 : 2, 5 : 2 # good
                                                        })
    all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    all_data["SimplFunctional"] = all_data.Functional.replace({1 : 1, 2 : 1, # bad
                                                         3 : 2, 4 : 2, # major
                                                         5 : 3, 6 : 3, 7 : 3, # minor
                                                         8 : 4 # typical
                                                        })
    all_data["SimplKitchenQual"] = all_data.KitchenQual.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    all_data["SimplHeatingQC"] = all_data.HeatingQC.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    all_data["SimplBsmtFinType1"] = all_data.BsmtFinType1.replace({1 : 1, # unfinished
                                                             2 : 1, 3 : 1, # rec room
                                                             4 : 2, 5 : 2, 6 : 2 # living quarters
                                                            })
    all_data["SimplBsmtFinType2"] = all_data.BsmtFinType2.replace({1 : 1, # unfinished
                                                             2 : 1, 3 : 1, # rec room
                                                             4 : 2, 5 : 2, 6 : 2 # living quarters
                                                            })
    all_data["SimplBsmtCond"] = all_data.BsmtCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
    all_data["SimplBsmtQual"] = all_data.BsmtQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
    all_data["SimplExterCond"] = all_data.ExterCond.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    all_data["SimplExterQual"] = all_data.ExterQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    
    # 2* Combinations of existing features
    # Overall quality of the house
    all_data["OverallGrade"] = all_data["OverallQual"] * all_data["OverallCond"]
    # Overall quality of the garage
    all_data["GarageGrade"] = all_data["GarageQual"] * all_data["GarageCond"]
    # Overall quality of the exterior
    all_data["ExterGrade"] = all_data["ExterQual"] * all_data["ExterCond"]
    # Overall kitchen score
    all_data["KitchenScore"] = all_data["KitchenAbvGr"] * all_data["KitchenQual"]
    # Overall fireplace score
    all_data["FireplaceScore"] = all_data["Fireplaces"] * all_data["FireplaceQu"]
    # Overall garage score
    all_data["GarageScore"] = all_data["GarageArea"] * all_data["GarageQual"]
    # Overall pool score
    all_data["PoolScore"] = all_data["PoolArea"] * all_data["PoolQC"]
    # Simplified overall quality of the house
    all_data["SimplOverallGrade"] = all_data["SimplOverallQual"] * all_data["SimplOverallCond"]
    # Simplified overall quality of the exterior
    all_data["SimplExterGrade"] = all_data["SimplExterQual"] * all_data["SimplExterCond"]
    # Simplified overall pool score
    all_data["SimplPoolScore"] = all_data["PoolArea"] * all_data["SimplPoolQC"]
    # Simplified overall garage score
    all_data["SimplGarageScore"] = all_data["GarageArea"] * all_data["SimplGarageQual"]
    # Simplified overall fireplace score
    all_data["SimplFireplaceScore"] = all_data["Fireplaces"] * all_data["SimplFireplaceQu"]
    # Simplified overall kitchen score
    all_data["SimplKitchenScore"] = all_data["KitchenAbvGr"] * all_data["SimplKitchenQual"]
    # Total number of bathrooms
    all_data["TotalBath"] = all_data["BsmtFullBath"] + (0.5 * all_data["BsmtHalfBath"]) + \
    all_data["FullBath"] + (0.5 * all_data["HalfBath"])
    # Total SF for house (incl. basement)
    all_data["AllSF"] = all_data["GrLivArea"] + all_data["TotalBsmtSF"]
    # Total SF for 1st + 2nd floors
    all_data["AllFlrsSF"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    # Total SF for porch
    all_data["AllPorchSF"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + \
    all_data["3SsnPorch"] + all_data["ScreenPorch"]
    # Has masonry veneer or not
    all_data["HasMasVnr"] = all_data.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                                   "Stone" : 1, "None" : 0})
    # House completed before sale or not
    all_data["BoughtOffPlan"] = all_data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                          "Family" : 0, "Normal" : 0, "Partial" : 1})
                                                          
    # Create new features
    # 3* Polynomials on the top 10 existing features
    all_data["OverallQual-s2"] = all_data["OverallQual"] ** 2
    all_data["OverallQual-s3"] = all_data["OverallQual"] ** 3
    all_data["OverallQual-Sq"] = np.sqrt(all_data["OverallQual"])
    all_data["AllSF-2"] = all_data["AllSF"] ** 2
    all_data["AllSF-3"] = all_data["AllSF"] ** 3
    all_data["AllSF-Sq"] = np.sqrt(all_data["AllSF"])
    all_data["AllFlrsSF-2"] = all_data["AllFlrsSF"] ** 2
    all_data["AllFlrsSF-3"] = all_data["AllFlrsSF"] ** 3
    all_data["AllFlrsSF-Sq"] = np.sqrt(all_data["AllFlrsSF"])
    all_data["GrLivArea-2"] = all_data["GrLivArea"] ** 2
    all_data["GrLivArea-3"] = all_data["GrLivArea"] ** 3
    all_data["GrLivArea-Sq"] = np.sqrt(all_data["GrLivArea"])
    all_data["SimplOverallQual-s2"] = all_data["SimplOverallQual"] ** 2
    all_data["SimplOverallQual-s3"] = all_data["SimplOverallQual"] ** 3
    all_data["SimplOverallQual-Sq"] = np.sqrt(all_data["SimplOverallQual"])
    all_data["ExterQual-2"] = all_data["ExterQual"] ** 2
    all_data["ExterQual-3"] = all_data["ExterQual"] ** 3
    all_data["ExterQual-Sq"] = np.sqrt(all_data["ExterQual"])
    all_data["GarageCars-2"] = all_data["GarageCars"] ** 2
    all_data["GarageCars-3"] = all_data["GarageCars"] ** 3
    all_data["GarageCars-Sq"] = np.sqrt(all_data["GarageCars"])
    all_data["TotalBath-2"] = all_data["TotalBath"] ** 2
    all_data["TotalBath-3"] = all_data["TotalBath"] ** 3
    all_data["TotalBath-Sq"] = np.sqrt(all_data["TotalBath"])
    all_data["KitchenQual-2"] = all_data["KitchenQual"] ** 2
    all_data["KitchenQual-3"] = all_data["KitchenQual"] ** 3
    all_data["KitchenQual-Sq"] = np.sqrt(all_data["KitchenQual"])
    all_data["GarageScore-2"] = all_data["GarageScore"] ** 2
    all_data["GarageScore-3"] = all_data["GarageScore"] ** 3
    all_data["GarageScore-Sq"] = np.sqrt(all_data["GarageScore"])

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#create data for computing (e.g. without target and id)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
                      

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#I assume that a skewness of over 0.66 is to enough to log transform
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.66]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

create_features(all_data)

all_data = all_data.fillna(all_data.median())
all_data = pd.get_dummies(all_data)

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

#calculate root mean square error
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
    
# second one is Lasso, we repeat the process
alphas = [2, 1, 0.1, 0.001, 0.0005]
model_lasso = LassoCV(alphas = alphas).fit(X_train, y)

cv_lasso = rmse_cv(model_lasso)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

#final predictions

pred = np.exp(model_lasso.predict(X_test)) - 1
test["SalePrice"] = pred

out = open('submission.csv', "w")
out.write("Id,SalePrice\n")

rows = []
for index, house in test.iterrows():
    rows.append("%d,%d\n"%(house["Id"],house["SalePrice"]))
    
out.writelines(rows)
out.close()
