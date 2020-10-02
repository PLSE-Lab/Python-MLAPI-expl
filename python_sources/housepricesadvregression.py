#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fileNameTrain = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
fileNameTest  = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'
fileNameSubm  = '/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv'
fileNameDesc  = '/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt'

dfTrain = pd.read_csv(fileNameTrain)
dfTest  = pd.read_csv(fileNameTest)
dfSubm  = pd.read_csv(fileNameSubm)
dfDesc  = open(fileNameDesc,'r')
dfDesc  = dfDesc.read().replace('\t','     ').split('\n')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns; sns.set(style="ticks", color_codes=True)

import lightgbm as lgb
print(lgb.__version__)


# In[ ]:


dfDesc


# In[ ]:


pd.set_option("display.max_rows", None)
pd.set_option("display.max_rows", 100)
dfTrain.describe(include = 'all').transpose()
dfTest.describe(include = 'all').transpose()


# ### Replace NaN fields

# In[ ]:


def checkDataQuanity(df,dataSet):
    for col in df.columns:
        
        colVals = df[col]
        colValsU= colVals.unique()
        colValsType = colValsU.dtype
        print(dataSet,colValsType,'\t', col,3*'\t', end='')
        if colValsType == np.dtype('int64'):
            posNan = np.where(np.isnan(colVals.to_numpy() ) == True)[0]
            #Replace Nan
            df = df.fillna(value={col: np.amin(colVals)})
            if posNan.shape[0]>0:
                print('nbr of NaN positions:',posNan.shape)
            else:
                print('clean')
        elif colValsType == np.dtype('float64'):
            posNan = np.where(np.isnan(colVals.to_numpy() ) == True)[0]
            #Replace Nan
            df = df.fillna(value={col: np.amin(colVals)})
            if posNan.shape[0]>0:
                print('nbr of NaN positions:',posNan.shape)
            else:
                print('clean')

        elif colValsType == np.dtype('object'):
            posNan = colVals.to_list()
            df = df.fillna(value={col: 'NAN'})
            if colVals.isnull().sum()>0:
                print('nbr of NaN positions:',colVals.isnull().sum())
            else:
                print('clean',colVals.unique())
                
    print()
    return df


# In[ ]:


dfTrainClean = checkDataQuanity(dfTrain,'TRAIN')
dfTestClean  = checkDataQuanity(dfTest, 'TEST')


# In[ ]:


checkDataQuanity(dfTrainClean,'TRAINclean')
checkDataQuanity(dfTestClean, 'TESTclean')


# ### Create new Feature

# In[ ]:


for dataset in [dfTrainClean, dfTestClean]:
    dataset['TotalSF']           = dataset['TotalBsmtSF'] +        dataset['1stFlrSF']   + dataset['2ndFlrSF']
    dataset['Total_sqr_footage'] = dataset['BsmtFinSF1']  +        dataset['BsmtFinSF2'] + dataset['1stFlrSF']      +        dataset['2ndFlrSF']
    dataset['Total_Bathrooms']   = dataset['FullBath']    + (0.5 * dataset['HalfBath'])  + dataset['BsmtFullBath']  + (0.5 * dataset['BsmtHalfBath'])
    dataset['Total_porch_sf']    = dataset['OpenPorchSF'] +        dataset['3SsnPorch']  + dataset['EnclosedPorch'] +        dataset['ScreenPorch']     + dataset['WoodDeckSF']
    
    dataset['haspool']      = dataset['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    dataset['has2ndfloor']  = dataset['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    dataset['hasgarage']    = dataset['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    dataset['hasbsmt']      = dataset['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    dataset['hasfireplace'] = dataset['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
plt.plot(dfTrainClean['TotalSF'],dfTrainClean['Total_sqr_footage'],marker='+',linewidth=0);
plt.plot(dfTestClean['TotalSF'],dfTestClean['Total_sqr_footage'],marker='.',linewidth=0);
plt.grid(True)


# In[ ]:


print(dfTrainClean.columns.shape);


# ### Outlier detection

# In[ ]:


dataset, col = dfTrainClean,'1stFlrSF'
q=dataset[col].quantile(0.999)
dfOutlier1 = dataset[dataset[col] > q]
dfOutlier1


# In[ ]:


dataset, col = dfTrainClean,'GrLivArea'
q=dataset[col].quantile(0.999)
dfOutlier2 = dataset[dataset[col] > q]
dfOutlier2


# In[ ]:


dataset, col = dfTrainClean,'TotRmsAbvGrd'
q=dataset[col].quantile(0.9925)
dfOutlier3 = dataset[dataset[col] > q]
dfOutlier3


# ### DROP OUTLIERS!!!

# In[ ]:


dfTrainClean = dfTrainClean.drop([496,1298,523,635])


# #### Plot Training set data 

# In[ ]:


NROWS=30
NCOLS=3
colNames=sorted(dfTrainClean.columns)
fig, axAll = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(22,120))
idx=0
for r in range(NROWS):
    for c in range(NCOLS):
        if idx < len(colNames):
            sns.scatterplot(x=colNames[idx],y = 'SalePrice',data = dfTrainClean,ax=axAll[r,c],edgecolor='blue',  linewidth=0.5,hue="1stFlrSF")
            sns.scatterplot(x=colNames[idx],y = 'SalePrice',data = dfOutlier1,   ax=axAll[r,c],   color='chartreuse')
            sns.scatterplot(x=colNames[idx],y = 'SalePrice',data = dfOutlier2,   ax=axAll[r,c],   color='magenta')
            sns.scatterplot(x=colNames[idx],y = 'SalePrice',data = dfOutlier3,   ax=axAll[r,c],   color='aqua')
            axAll[r][c].grid(True)
        idx=idx+1


# ### Symmetrize numerical values

# In[ ]:


colNames=sorted(dfTrainClean.columns)

for col in colNames:
    colValsUTrain = dfTrainClean[col].unique()
    if colValsUTrain.dtype != np.dtype('object'):
        
        if col not in ('SalePrice','Id'):
            estMedian = dfTrainClean[col].median()
            dfTrainClean[col] = dfTrainClean[col]- estMedian
            dfTestClean[col]  = dfTestClean[col] - estMedian
            
        if col  == 'SalePrice':
            estMedianSalesPrice = dfTrainClean[col].median()
            dfTrainClean[col] = dfTrainClean[col]- estMedianSalesPrice
                   
print('SalesPriceMedia on training set:', estMedianSalesPrice)


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.distplot(dfTrainClean['SalePrice']+estMedianSalesPrice,ax=ax[0]);
ax[0].grid(True)
sns.distplot(dfTrainClean['SalePrice'],ax=ax[1],color='red');
ax[1].grid(True)


# ### Encoding non-numerical values (more or less symmetric around zero)

# In[ ]:


colNames=sorted(dfTrainClean.columns)
for col in colNames:
    colValsUTrain = dfTrainClean[col].unique()
    colValsUTest  = dfTrainClean[col].unique()
    colValsU = np.sort(np.unique(np.concatenate((colValsUTrain,colValsUTest))))
    
    if colValsU.dtype == np.dtype('object'):
        encoderDict = {}
        if len(colValsU)>2:
            symOffset = len(colValsU)//2
        else:
            symOffset = 0.5
        for k,val in enumerate(colValsU):
            encoderDict[val] = k - symOffset
        print(col, '\t\t\t',encoderDict)
    
        dfTrainClean[col] = dfTrainClean[col].map(encoderDict)
        dfTestClean[col]  = dfTestClean[col].map(encoderDict)


# ### Compare Train/Test data distributions

# In[ ]:


NROWS=30
NCOLS=3
colNames=sorted(dfTrainClean.columns)
fig, axAll = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(22,120))
idx=0
for r in range(NROWS):
    for c in range(NCOLS):
        hist, bin_edges = np.histogram(dfTrainClean[colNames[idx]])
        sns.distplot(dfTrainClean[colNames[idx]], bin_edges, kde=False, norm_hist=False, color='red' ,ax=axAll[r][c],label='train')
        if colNames[idx] not in ('SalePrice','Id'):
            sns.distplot(dfTestClean[colNames[idx]],  bin_edges, kde=False, norm_hist=False, color='blue',ax=axAll[r][c],label='test');
        axAll[r][c].grid(True)
        axAll[r][c].legend()
        idx=idx+1


# ### Suppress warnings

# In[ ]:


# naughty - but to suppress LinAlg warnings in KRR
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action='ignore', category=LinAlgWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


# ### GridSearch for parameter tuning - to be done for each model (with selected feature set)

# In[ ]:


XGB = XGBRegressor()

XGB_grid =    {'nthread':[4], 
               'objective':['reg:linear'],
               'learning_rate': [.03, 0.05, .07],
               'max_depth': [5, 6, 7],
               'min_child_weight': [4],
               'silent': [1],
               'subsample': [0.7],
               'colsample_bytree': [0.7],
               'n_estimators': [500]}

n_folds = 5

fearures = dfTrainClean.columns.drop('SalePrice').drop('Id')
X = dfTrainClean[fearures]
y = dfTrainClean['SalePrice']

kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
XGBModel = GridSearchCV(estimator = XGB, param_grid = XGB_grid, cv=kf, scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)
XGBModel.fit(X,y)
XGB_best = XGBModel.best_estimator_
XGBModel.best_params_


# ### Define models and feature sets

# In[ ]:


LRModel = LinearRegression()

KRR = KernelRidge(alpha=0.8, coef0=5, degree=2, gamma=None, kernel='polynomial', kernel_params=None)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, max_iter = 500, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


lgb1Model = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

lgb2Model = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )

gBoost1Model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

gBoost2Model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, 
                                         max_depth=3, max_features='sqrt', 
                                         min_samples_leaf=10, min_samples_split=5, 
                                         loss='huber', random_state =5)

xgBoost1Model = XGBRegressor(objective ='reg:squarederror',
                        n_estimators=1400,
                        learning_rate=0.05,
                        n_jobs=-1)

xgBoost2Model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

xgBoost3Model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=5, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

xgBoost4Model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=5, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.286206,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

xgBoost5Model = XGBRegressor(colsample_bytree=0.7, 
                             learning_rate=0.03, max_depth=6, 
                             min_child_weight=4, n_estimators=500, 
                             subsample=0.7, silent=1, 
                             random_state =7)

features00 = dfTrainClean.columns.drop('SalePrice').drop('Id')
features01 = dfTrainClean.columns.drop('SalePrice').drop('Id').drop('PoolQC').drop('Street').drop('Utilities')

features02 = ['TotalSF','BedroomAbvGr',
                     'YearBuilt',
                     'OverallQual',
                     'OverallCond',
                     'Neighborhood',
                    'GarageCars']

features03 = ['TotalSF',
                 'OverallQual',
                 'GarageCars',
                 #'GarageArea',
                 'YearBuilt',
                 'Fireplaces',
                 'CentralAir',
                 'Neighborhood',
                 'OverallCond',
                 'BedroomAbvGr',
                 'BldgType',
                 'TotRmsAbvGrd',
                 'LandSlope',
                 'LotArea',
                ]

features04 = ['TotalSF',
                 'OverallQual',
                 'GarageCars',
                 'YearBuilt','Fireplaces','CentralAir',
                 'Neighborhood','OverallCond','BedroomAbvGr','BldgType','TotRmsAbvGrd',
                 'LandSlope','LotArea',]

features05 = ['TotalSF',
                 'OverallQual',
                 'GarageCars',
                 'GarageArea',
                 'YearBuilt',
                 'Fireplaces',
                 'CentralAir',
                 'Neighborhood',
                 'OverallCond',
                 'BedroomAbvGr',
                 'BldgType',
                 'TotRmsAbvGrd',
                 'LandSlope',
                 'LotArea',
                ]

features06 = ['LotArea','GrLivArea','TotalSF','BedroomAbvGr','FullBath',
                 'Neighborhood',
                 'GarageCars','GarageType',
                 'OverallQual','KitchenQual','ExterQual',
                 'CentralAir','Fireplaces','Heating',
                 'BsmtExposure',
                 'OverallCond','Condition1','ExterCond','YearBuilt',
                 'PavedDrive','LandSlope',
                 'Foundation',
                 'SaleType','SaleCondition']

features07 = ['LotArea','GrLivArea','TotalSF','BedroomAbvGr','FullBath',
                 'Neighborhood',
                 'GarageCars','GarageType',
                 'OverallQual','KitchenQual','ExterQual',
                 'CentralAir','Fireplaces','Heating',
                 'BsmtExposure',
                 'OverallCond','Condition1','ExterCond','YearBuilt',
                 'PavedDrive','LandSlope',
                 'Foundation',
                 'SaleType','SaleCondition',
                 #    
                 'MSSubClass','MSZoning','LandContour','HouseStyle',
                 'BsmtCond','Electrical','LotConfig','HeatingQC',
                 'RoofStyle','YearRemodAdd',
                 'BsmtQual','BldgType','LotShape','MasVnrType',
                 'BsmtFinType1','TotRmsAbvGrd','Functional']


features08 = ['TotalSF',
                     'OverallQual',
                     'GarageCars',
                     'YearBuilt',
                     'Fireplaces',
                     'CentralAir',
                     'Neighborhood',
                     'OverallCond',
                     'BedroomAbvGr',
                     'BldgType',
                     'TotRmsAbvGrd',
                     'LandSlope',
                     'LotArea',
                     'GrLivArea','FullBath',
                     'GarageType',
                     'KitchenQual','ExterQual',
                     'Heating',
                     'BsmtExposure',
                     'Condition1','ExterCond',
                     'PavedDrive',
                     'Foundation',
                     'SaleType','SaleCondition',
                     #    
                     'MSSubClass','MSZoning','LandContour','HouseStyle',
                     'BsmtCond','Electrical','LotConfig','HeatingQC',
                     'RoofStyle','YearRemodAdd',
                     'BsmtQual','LotShape','MasVnrType',
                     'BsmtFinType1','Functional']

featureSets = [features00,features01,features02,features03,features04,features05,features06,features07,features08]
#featureSets = [features00,features01]
rmses = []
fittedModels = []

for features in featureSets:

    X     = dfTrainClean[features]
    XTest = dfTestClean[features]
    y     = dfTrainClean['SalePrice']
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 1,train_size = 0.75)
    print('SampleSize of Age training set: ',X.shape[0],train_X.shape[0],valid_X.shape[0],train_X.shape[0]/X.shape[0])
    
    LRModel.fit(train_X, train_y);
    print('----LR  - fitting done!')
    
    KRR.fit(train_X, train_y);
    print('----KRR - fitting done!')
    
    lasso.fit(train_X, train_y);
    print('----lasso - fitting done!')
    
    ENet.fit(train_X, train_y);
    print('----ENet - fitting done!')
    
    lgb1Model.fit(train_X, train_y,
                eval_set=[(valid_X, valid_y)],
                eval_metric='l1',
                early_stopping_rounds=5,
                verbose=100)
    print('----LGB1 - fitting done!')
    
    lgb2Model.fit(train_X, train_y,
                eval_set=[(valid_X, valid_y)],
                eval_metric='l1',
                early_stopping_rounds=5,
                verbose=100)
    print('----LGB2 - fitting done!')

    gBoost1Model.fit(train_X, train_y);
    print('----gB1  - fitting done!')

    gBoost2Model.fit(train_X, train_y);
    print('----gB2  - fitting done!')
    
    xgBoost1Model.fit(train_X, train_y, 
                 early_stopping_rounds=15, 
                 eval_set=[(valid_X, valid_y)],
                 verbose=False);
    print('----XGB1 - fitting done!')
    
    xgBoost2Model.fit(train_X, train_y, 
                 early_stopping_rounds=15, 
                 eval_set=[(valid_X, valid_y)],
                 verbose=False);
    print('----XGB2 - fitting done!')

    xgBoost3Model.fit(train_X, train_y, 
                 early_stopping_rounds=15, 
                 eval_set=[(valid_X, valid_y)],
                 verbose=False);
    print('----XGB3 - fitting done!')
    
    xgBoost4Model.fit(train_X, train_y, 
                 early_stopping_rounds=15, 
                 eval_set=[(valid_X, valid_y)],
                 verbose=False);
    print('----XGB4 - fitting done!')
    
    xgBoost5Model.fit(train_X, train_y, 
                 early_stopping_rounds=15, 
                 eval_set=[(valid_X, valid_y)],
                 verbose=False);
    print('----XGB5 - fitting done!')
    
    preds_valLR = LRModel.predict(valid_X)
    preds_valLRFloored = np.clip(preds_valLR+estMedianSalesPrice, dfTrain['SalePrice'].min() ,dfTrain['SalePrice'].max())
    rmseLR = np.sqrt(mean_squared_error(np.log(preds_valLRFloored),  np.log(valid_y + estMedianSalesPrice)))
    print('***  LR RMSE: ',rmseLR)
    
    preds_valKRR = KRR.predict(valid_X)
    preds_valKRRFloored = np.clip(preds_valKRR+estMedianSalesPrice, dfTrain['SalePrice'].min() ,dfTrain['SalePrice'].max())
    rmseKRR = np.sqrt(mean_squared_error(np.log(preds_valKRRFloored),  np.log(valid_y + estMedianSalesPrice)))
    print('***  KRR RMSE: ',rmseKRR)
    
    preds_vallasso = lasso.predict(valid_X)
    preds_vallassoFloored = np.clip(preds_vallasso+estMedianSalesPrice, dfTrain['SalePrice'].min() ,dfTrain['SalePrice'].max())
    rmseLasso = np.sqrt(mean_squared_error(np.log(preds_vallassoFloored),  np.log(valid_y + estMedianSalesPrice)))
    print('***  lasso RMSE: ',rmseLasso)
    
    preds_valENet = ENet.predict(valid_X)
    preds_valENetFloored = np.clip(preds_valENet+estMedianSalesPrice, dfTrain['SalePrice'].min() ,dfTrain['SalePrice'].max())
    rmseENet = np.sqrt(mean_squared_error(np.log(preds_valENetFloored),  np.log(valid_y + estMedianSalesPrice)))
    print('***  ENet RMSE: ',rmseENet)
    
    preds_valLGB1 = lgb1Model.predict(valid_X)
    rmseLGB1 = np.sqrt(mean_squared_error(np.log(preds_valLGB1 + estMedianSalesPrice),  np.log(valid_y + estMedianSalesPrice)))
    print('***  LGB1 RMSE: ',rmseLGB1)
    
    preds_valLGB2 = lgb2Model.predict(valid_X)
    rmseLGB2 = np.sqrt(mean_squared_error(np.log(preds_valLGB2 + estMedianSalesPrice),  np.log(valid_y + estMedianSalesPrice)))
    print('***  LGB2 RMSE: ',rmseLGB2)

    preds_valGB1 = gBoost1Model.predict(valid_X)
    rmseGB1  = np.sqrt(mean_squared_error(np.log(preds_valGB1 + estMedianSalesPrice),  np.log(valid_y + estMedianSalesPrice)))
    print('***   gB1 RMSE: ',rmseGB1)

    preds_valGB2 = gBoost2Model.predict(valid_X)
    rmseGB2  = np.sqrt(mean_squared_error(np.log(preds_valGB2 + estMedianSalesPrice),  np.log(valid_y + estMedianSalesPrice)))
    print('***   gB2 RMSE: ',rmseGB2)

    preds_valXGB1 = xgBoost1Model.predict(valid_X)
    rmseXGB1 = np.sqrt(mean_squared_error(np.log(preds_valXGB1 + estMedianSalesPrice), np.log(valid_y + estMedianSalesPrice)))
    print('*** XGB1 RMSE: ',rmseXGB1)

    preds_valXGB2 = xgBoost2Model.predict(valid_X)
    rmseXGB2 = np.sqrt(mean_squared_error(np.log(preds_valXGB2 + estMedianSalesPrice), np.log(valid_y + estMedianSalesPrice)))
    print('*** XGB2 RMSE: ',rmseXGB2)

    preds_valXGB3 = xgBoost3Model.predict(valid_X)
    rmseXGB3 = np.sqrt(mean_squared_error(np.log(preds_valXGB3 + estMedianSalesPrice), np.log(valid_y + estMedianSalesPrice)))
    print('*** XGB3 RMSE: ',rmseXGB3)

    preds_valXGB4 = xgBoost4Model.predict(valid_X)
    rmseXGB4 = np.sqrt(mean_squared_error(np.log(preds_valXGB4 + estMedianSalesPrice), np.log(valid_y + estMedianSalesPrice)))
    print('*** XGB4 RMSE: ',rmseXGB4)
    
    preds_valXGB5 = xgBoost5Model.predict(valid_X)
    rmseXGB5 = np.sqrt(mean_squared_error(np.log(preds_valXGB5 + estMedianSalesPrice), np.log(valid_y + estMedianSalesPrice)))
    print('*** XGB5 RMSE: ',rmseXGB5)
    
    rmses.append([rmseLR,rmseLasso,rmseENet,rmseKRR, rmseLGB1,rmseLGB2,rmseGB1,rmseGB2,rmseXGB1,rmseXGB2,rmseXGB3,rmseXGB4,rmseXGB5])
    fittedModels.append([LRModel,lasso,ENet,KRR,lgb1Model,lgb2Model,gBoost1Model,gBoost2Model,xgBoost1Model,xgBoost2Model,xgBoost3Model,xgBoost4Model,xgBoost5Model])
    
data = {'model':['LinReg','Lasso','ENet','KRR','LBG1','LBG2','gB1','gB2','XGB1','XGB2','XGB3','XGB4','XGB5']}
for k,rmse in enumerate(rmses):
    data['features_'+str(k)] = rmse

dfValidResults = pd.DataFrame(data)


# ### Result overview

# In[ ]:


dfValidResults.style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')


# In[ ]:


plt.figure(figsize=(12,5));
for k in df.index.to_numpy():
    plt.plot(dfValidResults.iloc[k][1:].to_numpy(), marker='o',linewidth=0.25,label=dfValidResults.iloc[k][0]);
plt.grid(True)
plt.xlabel('featureSet')
plt.ylim(0.1,0.25)
plt.xlim(-1,10)
plt.legend();


# ### Cross validate model with Kfold stratified cross val

# ### Feature importance

# In[ ]:


featureSetNbr = -1  #hardcoded
features = featureSets[featureSetNbr]

featureWeights = {}

for k,model in enumerate(fittedModels[featureSetNbr]):
    try:
        modelName = dfValidResults['model'][k]
        mfip      = model.feature_importances_
        
        featureWeights['weights_'+modelName]=mfip
    except:
        pass
    
dfFeatureImportance = pd.DataFrame(data=featureWeights,index=features).sort_values(['weights_XGB3'], ascending=False)
dfFeatureImportance


# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(20,10));
dfFeatureImportance = dfFeatureImportance.sort_values(['weights_XGB1'], ascending=True)[:20]
dfFeatureImportance[['weights_XGB1','weights_XGB2','weights_XGB3','weights_XGB4','weights_XGB5']].plot.barh(ax=ax);
ax.grid(True)


# In[ ]:


xgb.plot_importance(xgBoost3Model, max_num_features=20,importance_type='gain');
xgb.plot_importance(xgBoost3Model, max_num_features=20,importance_type='weight');
xgb.plot_importance(xgBoost3Model, max_num_features=20,importance_type='cover');
#lgb.plot_importance(lgb2Model, max_num_features=20);


# ### Submission

# In[ ]:


xgBoostTestModel = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=5, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.28571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


features = features08
model    = xgBoost3Model
#model    = xgBoostTestModel

X        = dfTrainClean[features]
XTest    = dfTestClean[features]
y        = dfTrainClean['SalePrice']
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 1,train_size = 0.75)

model.fit(train_X, train_y, 
             early_stopping_rounds=15, 
             eval_set=[(valid_X, valid_y)],
             verbose=False);

preds = model.predict(valid_X)
rmse = np.sqrt(mean_squared_error(np.log(preds + estMedianSalesPrice), np.log(valid_y + estMedianSalesPrice)))
print('model RMSE on validation set: ',rmse)
estimate = model.predict(XTest)
dfOut=pd.DataFrame(dfTest['Id'])
dfOut.insert(1,'SalePrice',estimate + estMedianSalesPrice)
dfOut.to_csv('submission.csv', index=False)


# In[ ]:


model RMSE on validation set:  0.11484437904218533


# In[ ]:


dfOut


# In[ ]:


dfTestClean['SalePrice'] = estimate


# In[ ]:


NROWS=30
NCOLS=3
colNames=sorted(dfTrainClean.columns)
fig, axAll = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(22,120))
idx=0
for r in range(NROWS):
    for c in range(NCOLS):
        if idx < len(colNames):
            sns.scatterplot(x=colNames[idx],y = 'SalePrice',data = dfTrainClean,ax=axAll[r,c],edgecolor='blue',  linewidth=0.5)
            sns.scatterplot(x=colNames[idx],y = 'SalePrice',data = dfTestClean, ax=axAll[r,c],edgecolor='red',   linewidth=0.5)
            axAll[r][c].grid(True)
        idx=idx+1


# In[ ]:




