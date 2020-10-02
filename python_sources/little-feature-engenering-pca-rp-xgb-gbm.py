import pandas as pd
import lightgbm as lgm
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn import random_projection
from scipy.special import boxcox1p
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline


#Load train data
def loadTrain():
    data = pd.read_csv('../input/train.csv')
    data = data.drop(['Id'], axis=1)

    #remove rows with large values to make prediction more stable
    data.drop(data[(data["GrLivArea"]>4000)&(data["SalePrice"]<300000)].index,inplace=True)
    yData = data['SalePrice'].values
    xData = data.drop(['SalePrice'], axis=1)

    return xData, yData

#Load test data
def loadTest():
    data = pd.read_csv('../input/test.csv')
    testId = data['Id'].values
    data = data.drop(['Id'], axis=1)
    return data, testId

#Fill missing values(just copycat from another kernel)
def fillMissing(train, test):
    data = pd.concat([train, test]).reset_index(drop=True)
    aa = data.isnull().sum()
    zeroCols = aa[aa>0]
    zeroCols = list(zeroCols.index)

    data["LotAreaCut"] = pd.qcut(data.LotArea,10)
    data['LotFrontage']=data.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
    for col in cols:
        data[col].fillna(0, inplace=True)

    cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", 
            "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
    for col in cols1:
        data[col].fillna("None", inplace=True)

    cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
    for col in cols2:
        data[col].fillna(data[col].mode()[0], inplace=True)

    data.drop(['LotAreaCut'], axis=1, inplace=True)

    return data

#
def categorialToNum(data):
    
    #find cat features automatically
    cats = list(data.select_dtypes(include=["object"]).columns)
    
    #encode using pandas internal functionality
    for cat in cats:
        data[cat] = data[cat].astype('category')
        data[cat] = data[cat].cat.codes

#Add extra features, which tell about houses in the neighborhood
def addNeighborhoodFeatures(data):
    features = ['OverallQual', 'OverallCond', 'YearBuilt', 'ExterQual', '1stFlrSF', 'Fireplaces', 'Fence', 'PoolQC']

    #Just took min, max, mean, median, std and var of it
    #It could be only var without std, I guess
    grr = pd.DataFrame()
    for feature in features:
        gr = data.groupby(['Neighborhood'])[feature].agg(['min', 'max', 'mean', 'median', 'std', 'var'])
        gr.columns = [(feature + '_neigh_' + c) for c in gr.columns]
        grr = pd.concat([grr, gr], axis=1)

    data = data.join(grr, on='Neighborhood')

    return data

#Add another important features
def addFeatures(data):
    
    #Overall square of house
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    
    #Skew
    numbericFeat = data.dtypes[data.dtypes != "object"].index
    skewedFeat = data[numbericFeat].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewedFeat})
    skewness = skewness[abs(skewness) > 0.75]
    skewedFeat = skewness.index
    lam = 0.15
    for feat in skewedFeat:
        data[feat] = boxcox1p(data[feat], lam)

#Encode year as categorial feature(don't give visible result for me, but still make sense)
def yearEncode(data):
    lab=LabelEncoder()
    data["YearBuilt"] = lab.fit_transform(data["YearBuilt"])
    data["YearRemodAdd"] = lab.fit_transform(data["YearRemodAdd"])
    data["GarageYrBlt"] = lab.fit_transform(data["GarageYrBlt"])

#Random projection and PCA with RobustScaler scale
def decompose(train, test):
    train = train.fillna(0)
    test = test.fillna(0).reset_index(drop=True)
    
    #Scale data
    scaler = RobustScaler()
    trainScaled = scaler.fit(train).transform(train)
    testScaled = scaler.transform(test)
    
    #N_COMP equals count of features(could be tuned)
    N_COMP = train.shape[1]
    
    #Random projection
    p = random_projection.SparseRandomProjection(n_components=N_COMP)
    trainScaled0 = p.fit_transform(trainScaled)
    testScaled0 = p.transform(testScaled)
    
    #PCA with 128 params(could be tuned)
    N_COMP = 128
    trainScaled = np.nan_to_num(trainScaled)
    testScaled = np.nan_to_num(testScaled)
    pca = KernelPCA(n_components=N_COMP)
    trainScaled1 = pca.fit_transform(trainScaled)
    testScaled1 = pca.transform(testScaled)

    #Put all together
    trainAdd = pd.DataFrame(trainScaled0)
    testAdd = pd.DataFrame(testScaled0)

    cols = []
    for i in range(0, N_COMP):
        cols += ['rp_' + str(i)]

    trainAdd.columns = cols
    testAdd.columns = cols

    trainAddP = pd.DataFrame(trainScaled1)
    testAddP = pd.DataFrame(testScaled1)

    cols = []
    for i in range(0, N_COMP):
        cols += ['pca_' + str(i)]

    trainAddP.columns = cols
    testAddP.columns = cols

    #fill na
    trainAddP = trainAddP.fillna(0)
    testAddP = testAddP.fillna(0)

    train = pd.concat([train, trainAdd, trainAddP], axis=1)
    test = pd.concat([test, testAdd, testAddP], axis=1)

    return train, test

#I used lightGBM and XGBoost here. I tried stack of another models, but it haven't worked good for me

#Train and test LightGBM
def lgb(train, valid, test):

    params = {
            'objective': 'regression',
            'num_leaves': 32,
            'learning_rate': 0.035, 
            'n_estimators': 720, 
            'metric': 'rmse',
            'is_training_metric': True,
            'max_bin': 55, 
            'bagging_fraction': 0.8,
            'bagging_freq': 5, 
            'feature_fraction': 0.2319
    }

    num_round = 5000
    
    #Train with valid
    model = lgm.train(
            params, 
            train, 
            num_round,
            early_stopping_rounds=300,
            valid_sets=[valid]
    )

    return np.expm1(model.predict(test))

#XGBoost
def xgbs(train, valid, test):
    watchlist = [(train, 'train'), (valid, 'valid')]
    params = {
          "objective": "reg:linear",
          "booster": "gbtree",
          "eval_metric": "rmse",
          "nthread": 4,
          "eta": 0.008,
          "max_depth": 4,
          "min_child_weight": 1.7817,
          "colsample_bytree": 0.4603, 
          "gamma": 0.0468,
          "subsample": 0.5213,
          "colsample_bytree": 0.2,
          "colsample_bylevel": 0.7,
          "alpha": 0,
          "lambda": 16.1035,
          "nrounds": 5000
    }
    modelXgb = xgb.train(params, train, 5000, watchlist, maximize=False, early_stopping_rounds = 300, verbose_eval=100)

    return np.expm1(modelXgb.predict(test))

def main():
    xTrain, yTrain = loadTrain()
    xTest, testId = loadTest()

    #Preparing data
    data = fillMissing(xTrain, xTest)
    categorialToNum(data)
    data = addNeighborhoodFeatures(data)
    addFeatures(data)
    yearEncode(data)
    
    #Separating 
    nTrain = xTrain.shape[0]
    xTrain, xTest = decompose(data[:nTrain], data[nTrain:])

    #Scale and log
    scaler = RobustScaler()
    xTrainS = scaler.fit(xTrain).transform(xTrain)
    yTrainS = np.log(yTrain)
    xTestS = scaler.transform(xTest)
    
    #Separating train and valid
    xDev, xVal, yDev, yVal = train_test_split(xTrainS, yTrainS, test_size = 0.1, random_state = 42)
    
    predLgb = lgb(lgm.Dataset(xDev, yDev), lgm.Dataset(xVal, yVal), xTestS)
    predXgb = xgbs(xgb.DMatrix(xDev, yDev), xgb.DMatrix(xVal, yVal), xgb.DMatrix(xTestS))

    #Put results together with some weights(could be tuned or found using another way)
    resPred = predLgb * 0.4 + predXgb * 0.6
    
    #Save submission
    sub = pd.DataFrame()
    sub['Id'] = testId
    sub['SalePrice'] = resPred
    sub.to_csv('submission.csv', index=False)

main()