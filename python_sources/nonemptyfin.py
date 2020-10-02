# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:18:33 2017

@author: karimmejri
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import skew

    
    
def computeSkewness(dataset):
    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
    skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    dataset[skewed_feats] = np.log1p(dataset[skewed_feats])
    return dataset


def normalize(X):
    """Normalize an array, or a dataframe, to have mean 0 and stddev 1."""
    return (X - np.mean(X, axis=0))/(np.std(X, axis=0))

    
    
def normalizeFeatures(dataset):
    dataset = dataset.fillna(dataset.mean())
    
    dataset["LotArea"] = normalize(dataset["LotArea"]) 
    dataset["YearBuilt"] = normalize(dataset["YearBuilt"]) 
    dataset["YearRemodAdd"] = normalize(dataset["YearRemodAdd"]) 
    dataset["PoolArea"] = normalize(dataset["PoolArea"]) 
    dataset["3SsnPorch"] = normalize(dataset["3SsnPorch"]) 
    dataset["YrSold"] = normalize(dataset["YrSold"]) 
    dataset["MoSold"] = normalize(dataset["MoSold"])
    
    dataset["MSSubClass"] = normalize(dataset["MSSubClass"]) 
    dataset["LotFrontage"] = normalize(dataset["LotFrontage"]) 
    dataset["MasVnrArea"] = normalize(dataset["MasVnrArea"]) 
    dataset["BsmtFinSF1"] = normalize(dataset["BsmtFinSF1"]) 
    dataset["BsmtFinSF2"] = normalize(dataset["BsmtFinSF2"]) 
    
    dataset["GrLivArea"] = normalize(dataset["GrLivArea"]) 
    dataset["GarageYrBlt"] = normalize(dataset["GarageYrBlt"]) 
    dataset["GarageCars"] = normalize(dataset["GarageCars"]) 
    dataset["GarageArea"] = normalize(dataset["GarageArea"]) 
    
    dataset["1stFlrSF"] = normalize(dataset["1stFlrSF"]) 
    dataset["2ndFlrSF"] = normalize(dataset["2ndFlrSF"]) 
    dataset["LowQualFinSF"] = normalize(dataset["LowQualFinSF"]) 
    dataset["GrLivArea"] = normalize(dataset["GrLivArea"]) 
    dataset["TotRmsAbvGrd"] = normalize(dataset["TotRmsAbvGrd"]) 
    dataset["GarageYrBlt"] = normalize(dataset["GarageYrBlt"]) 
    
    return dataset
    
    
    
    
def categoricalToNumeric(dataset):
    dataset = dataset.replace({
                           "BsmtCond" : {None: 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "BsmtExposure" : {None : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                           "BsmtFinType1" : {None : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtFinType2" : {None : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtQual" : {None : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                           "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "Functional" : {None : 0, "Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                                           "Min2" : 6, "Min1" : 7, "Typ" : 8},
                           "GarageCond" : {None : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "GarageQual" : {None : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "KitchenQual" : { None : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                           "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                           "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                           #"Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4},
                           "PoolQC" : {None : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                           "Alley" : { None : 0, "Grvl" : 1, "Pave" : 2 },
                           "FireplaceQu" : { None : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                    })
    
    return dataset
    
    
    
def dummiesFeatures(dataset):
    conv = pd.get_dummies(dataset['CentralAir'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['Street'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['RoofStyle'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['SaleType'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['SaleCondition'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['LandContour'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['LotConfig'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['BldgType'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['Fence'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['Foundation'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['MSZoning'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['RoofMatl'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['Exterior1st'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['Exterior2nd'])
    dataset = pd.concat([conv,dataset],axis=1)     
    conv = pd.get_dummies(dataset['MasVnrType'])
    dataset = pd.concat([conv,dataset],axis=1) 
    conv = pd.get_dummies(dataset['HouseStyle'])
    dataset = pd.concat([conv,dataset],axis=1) 
    conv = pd.get_dummies(dataset['Heating'])
    dataset = pd.concat([conv,dataset],axis=1)
    conv = pd.get_dummies(dataset['Electrical'])
    dataset = pd.concat([conv,dataset],axis=1)
    conv = pd.get_dummies(dataset['GarageType'])
    dataset = pd.concat([conv,dataset],axis=1)
    conv = pd.get_dummies(dataset['HouseStyle'])
    dataset = pd.concat([conv,dataset],axis=1)
    conv = pd.get_dummies(dataset['Neighborhood'])
    dataset = pd.concat([conv,dataset],axis=1)

    if "Id" in dataset: 
        dataset.drop(['Id'], inplace = True, axis = 1)

    return dataset
    
    
def addFeatures(dataset):
    
    #Proposed feature: '1stFlrSF' + '2ndFlrSF' to give us combined Floor Square Footage
    dataset['1stFlr_2ndFlr_Sf'] = np.log1p(dataset['1stFlrSF'] + dataset['2ndFlrSF'])
    #1stflr+2ndflr+lowqualsf+GrLivArea = All_Liv_Area
    dataset['All_Liv_SF'] = np.log1p(dataset['1stFlr_2ndFlr_Sf'] + dataset['LowQualFinSF'] + dataset['GrLivArea'])
    
    
    dataset["IsRegularLotShape"] = (dataset["LotShape"] == "Reg") * 1
    # Most properties are level; bin the other possibilities together
    # as "not level".
    dataset["IsLandLevel"] = (dataset["LandContour"] == "Lvl") * 1
    # Most land slopes are gentle; treat the others as "not gentle".
    dataset["IsLandSlopeGentle"] = (dataset["LandSlope"] == "Gtl") * 1
    # Most properties use standard circuit breakers.
    dataset["IsElectricalSBrkr"] = (dataset["Electrical"] == "SBrkr") * 1
    # About 2/3rd have an attached garage.
    dataset["IsGarageDetached"] = (dataset["GarageType"] == "Detchd") * 1
    # Most have a paved drive. Treat dirt/gravel and partial pavement
    # as "not paved".
    dataset["IsPavedDrive"] = (dataset["PavedDrive"] == "Y") * 1
    # If YearRemodAdd != YearBuilt, then a remodeling took place at some point.
    dataset["Remodeled"] = (dataset["YearRemodAdd"] != dataset["YearBuilt"]) * 1    
    # Did a remodeling happen in the year the house was sold?
    dataset["RecentRemodel"] = (dataset["YearRemodAdd"] == dataset["YrSold"]) * 1    
    # Was this house sold in the year it was built?
    dataset["VeryNewHouse"] = (dataset["YearBuilt"] == dataset["YrSold"]) * 1
    dataset["Has2ndFloor"] = (dataset["2ndFlrSF"] == 0) * 1
    dataset["HasMasVnr"] = (dataset["MasVnrArea"] == 0) * 1
    dataset["HasWoodDeck"] = (dataset["WoodDeckSF"] == 0) * 1
    dataset["HasOpenPorch"] = (dataset["OpenPorchSF"] == 0) * 1
    dataset["HasEnclosedPorch"] = (dataset["EnclosedPorch"] == 0) * 1
    dataset["Has3SsnPorch"] = (dataset["3SsnPorch"] == 0) * 1
    dataset["HasScreenPorch"] = (dataset["ScreenPorch"] == 0) * 1
    


    dataset["SimplOverallQual"] = dataset.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                           4 : 2, 5 : 2, 6 : 2, # average
                                                           7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                          })
    dataset["SimplOverallCond"] = dataset.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                           4 : 2, 5 : 2, 6 : 2, # average
                                                           7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                          })
    dataset["SimplPoolQC"] = dataset.PoolQC.replace({1 : 1, 2 : 1, # average
                                                 3 : 2, 4 : 2 # good
                                                })
    dataset["SimplGarageCond"] = dataset.GarageCond.replace({1 : 1, # bad
                                                         2 : 1, 3 : 1, # average
                                                         4 : 2, 5 : 2 # good
                                                        })
    dataset["SimplGarageQual"] = dataset.GarageQual.replace({1 : 1, # bad
                                                         2 : 1, 3 : 1, # average
                                                         4 : 2, 5 : 2 # good
                                                        })
    dataset["SimplFireplaceQu"] = dataset.FireplaceQu.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    dataset["SimplFireplaceQu"] = dataset.FireplaceQu.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    dataset["SimplFunctional"] = dataset.Functional.replace({1 : 1, 2 : 1, # bad
                                                         3 : 2, 4 : 2, # major
                                                         5 : 3, 6 : 3, 7 : 3, # minor
                                                         8 : 4 # typical
                                                        })
    dataset["SimplKitchenQual"] = dataset.KitchenQual.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          })
    dataset["SimplHeatingQC"] = dataset.HeatingQC.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    dataset["BadHeating"] = dataset.HeatingQC.replace({1 : 1, # bad
                                                       2 : 0, 3 : 0, # average
                                                       4 : 0, 5 : 0 # good
                                                      })
    dataset["SimplBsmtFinType1"] = dataset.BsmtFinType1.replace({1 : 1, # unfinished
                                                             2 : 1, 3 : 1, # rec room
                                                             4 : 2, 5 : 2, 6 : 2 # living quarters
                                                            })
    dataset["SimplBsmtFinType2"] = dataset.BsmtFinType2.replace({1 : 1, # unfinished
                                                             2 : 1, 3 : 1, # rec room
                                                             4 : 2, 5 : 2, 6 : 2 # living quarters
                                                            })
    dataset["SimplBsmtCond"] = dataset.BsmtCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
    dataset["SimplBsmtQual"] = dataset.BsmtQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
    dataset["SimplExterCond"] = dataset.ExterCond.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    dataset["SimplExterQual"] = dataset.ExterQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
    return dataset
    
    
    
    
    
    
    
    
    
    
    
    




import sys
from sklearn.linear_model import Ridge,LassoCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin



trainName = '../input/train.csv' #sys.argv[1]
testName =  '../input/test.csv' #sys.argv[2]

import pandas as pd

train = pd.read_csv(trainName, index_col=0)
test = pd.read_csv(testName, index_col=0)

y = train["SalePrice"]
y_train = np.log(train["SalePrice"]+1)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


all_data.drop(['MiscFeature', 'Utilities'], axis=1, inplace=True)


#Feature engeenerig methods
all_data = addFeatures(all_data)
all_data = normalizeFeatures(all_data)
all_data = categoricalToNumeric(all_data)
all_data = dummiesFeatures(all_data)

    
all_data = all_data.select_dtypes([np.number])

x_train = np.array(all_data[:train.shape[0]])
x_test = np.array(all_data[train.shape[0]:])



"""
#LinearRegression MODEL
from sklearn import linear_model
lr = linear_model.LinearRegression(fit_intercept=True)
X, y = train_num.values[:,:-1], train_num.values[:,-1]
ylog = np.log(y)
lr.fit(X, ylog)
"""


"""
#LASSO MODEL 0.128
best_alpha = 0.00098
X, y = train_num.values[:,:-1], train_num.values[:,-1]
ylog = np.log(y)
lr1 = Lasso(alpha=best_alpha, max_iter=50000)
lr1.fit(X, ylog)
print(lr1.score(X, ylog))

#RIDGE MODEL 0.133
X, y = train_num.values[:,:-1], train_num.values[:,-1]
ylog = np.log(y)
lr = Ridge(alpha = 1.0)
lr.fit(X, ylog)
lr.score(X, ylog)

#ElasticNet MODEL 0.129
X, y = train_num.values[:,:-1], train_num.values[:,-1]
ylog = np.log(y)
lr3 = ElasticNet(alpha=0.001)
lr3.fit(X, ylog)
lr3.score(X, ylog)
"""


#xgb REGRESSOR
import xgboost as xgb

X, y = x_train, train["SalePrice"]
ylog = np.log(y)
print("create model..")
lr1 = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7300,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.5,
                 subsample=0.2,
                 seed=42,
                 silent=1)
print("model cerated")
lr1.fit(X, ylog)
print("finish learn")


#LASSO MODEL 0.128
#best_alpha = 0.00098
#best_alpha = 1e-4

X = x_train
ylog = y_train
lr2 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], selection='random', max_iter=15000)
lr2.fit(X, ylog)
print(lr2.score(X, ylog))


"""#Ensemble MODEL 0.129

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append(np.exp(regressor.predict(X).ravel()))

        return np.log1p(np.mean(self.predictions_, axis=0))

        return mproba

lr = EnsembleRegressor([lr1, lr3])
lr.fit(X, ylog)
"""
#Ensemble MODEL 0.129

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append(np.exp(regressor.predict(X).ravel()))

        return np.log1p(np.mean(self.predictions_, axis=0))

        return mproba

lr = EnsembleRegressor([lr1, lr2])
lr.fit(X, ylog)











print("start evaluate")
#predict and create the output file
y_pred_xgb = lr.predict(x_test)
y_pred = np.exp(y_pred_xgb)
print(y_pred)
"""
pred_df = pd.DataFrame(y_pred, index=x_test["Id"].index, columns=["SalePrice"])
pred_df.to_csv('output.csv', header=True, index_label='Id')
"""
SUBMISSION_FILE = '../input/sample_submission.csv'

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = y_pred_xgb
saleprice = np.exp(submission['SalePrice'])-1
submission['SalePrice'] = saleprice
submission.to_csv('xgstacker_starter.sub.csv', index=None)



"""
#cross validation
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
lr.fit(X_train, np.log(y_train))
lr.score(X_train, np.log(y_train))
lr.score(X_test, np.log(y_test))
preds = pd.DataFrame({"SalePrice":lr.predict(test)}, index=test.index)
preds.SalePrice = np.exp(preds.SalePrice)
preds.to_csv("/Users/karimmejri/Desktop/projectFDS/preds3.csv") # this can be submitted to Kaggle!



import matplotlib.pyplot as plt
plt.ion()
train_num_n = (train_num - train_num.mean())/train_num.std()
train_num_n.boxplot(vert=False)
"""


# Any results you write to the current directory are saved as output.