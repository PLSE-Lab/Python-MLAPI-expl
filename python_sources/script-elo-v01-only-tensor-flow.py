# 1) Import all Library that will be used

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import statsmodels.formula.api as smf

from scipy import stats

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model, svm, gaussian_process
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import lightgbm as lgb

from sklearn import preprocessing
from sklearn import utils

import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# 1) GradientBoostingRegressor Model
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }
GBR = GradientBoostingRegressor()
#GBR = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

# 2) Logistic Regression Model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()

# 3) Aplly Random Forest Model
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100)

# 4) Aplly XGBOOST Model
from xgboost import XGBClassifier
XGB = XGBClassifier()

# 5) Aplly KNeighbors Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# 6) Aplly SVC Model
SVC = SVC(probability=True)

# 7) Aplly Decision Tree Model
DTC = DecisionTreeClassifier()

# 8) Aplly GaussianNB Model
GNB = GaussianNB()

# 9) Aplly Neural Model
NN = MLPClassifier(hidden_layer_sizes=(100,100,50))

# 10) Aplly lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

# 11) Apply Elastic Net
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# 12) Apply Kernel Ridge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# 13) Apply LGBMRegressor
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# 14) Apply LGBMRegressor
from sklearn.linear_model import LinearRegression
LR2 = LinearRegression()

#15) Linear Regression with Tensor Flow
import tensorflow as tf

#
# ! ! ! ! ! ! ! 
#

# Read in the dataframes
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
merch = pd.read_csv('../input/merchants.csv')
ht = pd.read_csv('../input/historical_transactions.csv')
ss = pd.read_csv('../input/sample_submission.csv')

df_train_original = train
df_test_original = test

# Print data shapes
print('Files Load!!!')
print('sample submission shape', ss.shape)

# Train and Test
print('* * * Merchant File * * * ')
print('merchants shape', merch.shape)

# Historical File
print('* * * Historical File * * * ')
print('historical_transactions', ht.shape)

# Train and Test
print('* * * Train and Test File * * * ')
print('Train File - Original: ', train.shape)
print('Test File - Original: ', test.shape)

train = train.fillna(0)
test = test.fillna(0)

train['first1'] = train['first_active_month'].str[0:4]
train['first2'] = train['first_active_month'].str[5:7]
train = train.fillna(0)
train['first1'] = train['first1'].astype(int)
train['first2'] = train['first2'].astype(int)

test['first1'] = test['first_active_month'].str[0:4]
test['first2'] = test['first_active_month'].str[5:7]
test = test.fillna(0)
test['first1'] = test['first1'].astype(int)
test['first2'] = test['first2'].astype(int)

train = train.drop(['first_active_month'], axis=1)
test = test.drop(['first_active_month'], axis=1)
      
# Train and Test
print('Train File - Depois de todas Ops: ', train.shape)
print('Test File - Depois de todas Ops: ', test.shape)

all_data = pd.concat((train.loc[:,'card_id':'first2'],
                      test.loc[:,'card_id':'first2']))

print('Após concatenar os arquivos: ', all_data.shape)

# Replace Nulls by Zero
all_data = all_data.fillna(0)
#
# Historic Data
#

ht = ht.fillna(0)
DfhtGroup = pd.DataFrame()

print('# # # # # HISTORIC DATA FIELDS # # # # #')

print('01 - Field card_id')
ListGroup = ht.groupby(['card_id'])['card_id'].count()
DfhtGroup['card_id'] = ListGroup.index

print('02 - Field pur_mean')
ListGroup = pd.DataFrame(ht.groupby(['card_id'], as_index=False)['purchase_amount'].mean())
DfhtGroup['pur_mean'] = ListGroup['purchase_amount']

print('03 - Field Mon_mean')
ListGroup = pd.DataFrame(ht.groupby(['card_id'], as_index=False)['month_lag'].mean())
DfhtGroup['Mon_mean'] = ListGroup['month_lag']

print('04 - Field Count')
ListGroup = pd.DataFrame(ht.groupby(['card_id'], as_index=False)['authorized_flag'].count())
DfhtGroup['Count'] = ListGroup['authorized_flag']

print('13 - Field merchant_id - will be used in Join')
ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['merchant_id'].agg(lambda x: x.value_counts().index[0]))
DfhtGroup['merchant_id'] = ListGroup['merchant_id']

print('05 - Field City_Mode')
#ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['city_id'].agg(lambda x: x.value_counts().index[0]))
#DfhtGroup['City_Mode'] = ListGroup['city_id']

print('06 - Field Cat1_Mode')
#ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['category_1'].agg(lambda x: x.value_counts().index[0]))
#DfhtGroup['Cat1_Mode'] = ListGroup['category_1']

print('07 - Field Install_Mode')
#ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['installments'].agg(lambda x: x.value_counts().index[0]))
#DfhtGroup['Install_Mode'] = ListGroup['installments']

print('08 - Field Cat3_Mode')
#ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['category_3'].agg(lambda x: x.value_counts().index[0]))
#DfhtGroup['Cat3_Mode'] = ListGroup['category_3']

print('09 - Field Merch_Mode')
#ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['merchant_category_id'].agg(lambda x: x.value_counts().index[0]))
#DfhtGroup['Merch_Mode'] = ListGroup['merchant_category_id']

print('10 - Field Cat2_Mode')
#ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['category_2'].agg(lambda x: x.value_counts().index[0]))
#DfhtGroup['Cat2_Mode'] = ListGroup['category_2']

print('11 - Field State_Mode')
#ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['state_id'].agg(lambda x: x.value_counts().index[0]))
#DfhtGroup['State_Mode'] = ListGroup['state_id']

print('12 - Field Subsec_Mode')
#ListGroup = pd.DataFrame(ht.groupby('card_id', as_index=False)['subsector_id'].agg(lambda x: x.value_counts().index[0]))
#DfhtGroup['Subsec_Mode'] = ListGroup['subsector_id']

print ('Shape Df Main - all_data: ', all_data.shape)
print ('Shape Df - DfhtGroup: ', DfhtGroup.shape)
print ('Shape Df - merch: ', merch.shape)

dfJoin  = pd.DataFrame()
dfJoin  = pd.merge(all_data, DfhtGroup, on='card_id')
dfJoin  = dfJoin.fillna(0)

all_data = dfJoin

print ('All_data after Historic: ', all_data.shape)

merch = merch.fillna(0)
merch2 = merch.drop_duplicates(subset=['merchant_id'], keep=False)

merged = pd.DataFrame()
merged = all_data.merge(merch2, indicator=True, how='outer')
Tot_merged = merged[merged['_merge'] != 'left_only']
Tot_merged = merged[merged['_merge'] != 'right_only']

all_data_sel = all_data

# Fields that will be treated in Get Dummies:
all_data_sel['most_recent_sales_range'] = Tot_merged['most_recent_sales_range']
all_data_sel['most_recent_purchases_range'] = Tot_merged['most_recent_purchases_range']

# Numeric fields - will be changed to str to be treated in Get Dummies
all_data_sel['subsector_id'] = Tot_merged['subsector_id'].astype(str)
all_data_sel['city_id'] = Tot_merged['city_id'].astype(str)
all_data_sel['state_id'] = Tot_merged['state_id'].astype(str)

all_data = all_data_sel

print ('All_data after Join (Merch): ', all_data.shape)

all_data = all_data.drop(['card_id'], axis=1)
all_data = all_data.drop(['merchant_id'], axis=1)

# Get_Dummies categorics in Dummies 
all_data  = all_data.fillna(0)
all_data = pd.get_dummies(all_data)

print('all_data - After all Data Prep: ', all_data.shape)

#Matrix X_train from begining of  matrix (:) till end of df_train.shape[0]
X_train = all_data[:train.shape[0]]

#Matrix X_test after last record of df_train.shape[0]
X_test = all_data[train.shape[0]:]

# DATA CLEANING!!!
print('X_train - Before Cleanning: ', X_train.shape)

# First Field all_data, other = all_data_temp
all_data_temp = X_train

#1) Count: Removing outliers = < 1500:
all_data_temp = all_data_temp[all_data['Count'] < 1500]

#2) Target: Removing outliers < 12 & > -12:
all_data_temp = all_data_temp[all_data['target'] < 12]
all_data_temp = all_data_temp[all_data_temp['target'] > -12]

#3) pur_mean: Removing outliers < 20:
all_data_temp = all_data_temp[all_data_temp['pur_mean'] < 20]

#4) Mon_mean: Removing outliers > -12:
all_data_temp = all_data_temp[all_data_temp['Mon_mean'] > -12]

# Reformat all_data
X_train = all_data_temp

print('X_train - After Cleanning: ', X_train.shape)

# Create y after moves:
y = X_train.target
X_train = X_train.drop(['target'], axis=1)
X_test = X_test.drop(['target'], axis=1)

#
# ! ! ! ! ! ! ! 
#
def RunModel (ModelName, Model, Df_Test_Original, X_train, X_test, y):

    print ('# # # # Prediction for Model:  ', ModelName, '# # # #')
    print ('Shape X_train: ', X_train.shape)
    print ('Shape X_test: ', X_test.shape)
    print ('Shape y: ', y.shape)
    print ('Model: ', Model)
    
    print ('.FIT Model: ', Model)

    Model.fit(X_train, y)

    print ('PREDICT TRAIN: ', Model)

    yhat_Train = Model.predict(X_train)
    
    print ('PREDICT TEST: ', Model)

    yhat_test = Model.predict(X_test)
    
    print ('# # # # Tamanho do Df_Test_Original:', Df_Test_Original.shape)
    print ('# # # # Prediction:', yhat_test.shape, yhat_test)

    Filename = 'Output_' + ModelName + '.csv'
    
    df_Output= pd.DataFrame()
    df_Output['card_id'] = Df_Test_Original['card_id']
    df_Output['target'] = yhat_test
    df_Output.to_csv(Filename, index = False)
    
    return yhat_test

def Bagging (ModelName, Model, Df_Test_Original, X_train, X_test, y):

    from sklearn.ensemble import BaggingClassifier
    n_estimators = [1,2,3,4,5,10,15,20]
    dt_bag_scores = []   
    lr_bag_scores = []
    for ne in n_estimators:
        dt = DecisionTreeClassifier(max_depth=15, random_state=1)
        lr = LogisticRegression(random_state=1)
    
        dt_bag = BaggingClassifier(dt, n_estimators=ne)
        lr_bag = BaggingClassifier(lr, n_estimators=ne)

        dt_bag.fit(X_train, y)
        lr_bag.fit(X_train, y)

        dt_bag_scores.append(eval_auc(dt_bag, X_test, yhat_test))
        lr_bag_scores.append(eval_auc(lr_bag, X_test, yhat_test))

        print(ne, dt_bag_scores[-1], lr_bag_scores[-1])

def GridSearch (ModelName, Model, Df_Test_Original, X_train, X_test, y):
    from sklearn.model_selection import GridSearchCV
    params = {"max_depth": [3,10,None],"max_features": [None, "auto"],"min_samples_leaf": [1, 5, 10],"min_samples_split": [2, 10]         }
    gsdt = GridSearchCV(dt, params, n_jobs=-1, cv=5)
    gsdt.fit(X_train, y)
    display ('Best Parameters!!: ',  gsdt.best_params_)
    display ('Best Score     !!: ',  gsdt.best_score_)
    bdt.get_params()
    params = {"base_estimator__max_depth": [3,10,None],"base_estimator__max_features": [None, "auto"],"base_estimator__min_samples_leaf": [1, 5, 10],"base_estimator__min_samples_split": [2, 10],'bootstrap_features': [False, True],'max_features': [0.5, 1.0],'max_samples': [0.5, 1.0],'n_estimators': [5, 15, 40],         }
    gsbdt = GridSearchCV(bdt, params, n_jobs=3, cv=5)
    gsbdt.fit(X_train, y)
    display ('Best Parameters!!: ',  gsdt.best_params_)
    display ('Best Score     !!: ',  gsdt.best_score_)
#
# ! ! ! ! ! ! ! 
#
def Runlightgbm  (ModelName, Model, Df_Test_Original, X_train, X_test, y):

    print ('# # # # Prediction for Model:  ', ModelName, '# # # #')
    print ('Shape X_train: ', X_train.shape)
    print ('Shape X_test: ', X_test.shape)
    print ('Shape y: ', y.shape)
    print ('Model: ', Model)

    import lightgbm as lgb
    d_train = lgb.Dataset(X_train, label=y)
    
    params ={
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 7,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':int(2),
                'bagging_seed':int(2),
                'drop_seed':int(2)
                }
    
    print ('.FIT Model (Nesse caso, lgb.train: ', Model)

    clf = lgb.train(params, d_train)

    print ('PREDICT TEST: ', Model)

    yhat_test=clf.predict(X_test)
    
    print ('# # # # Tamanho do Df_Test_Original:', Df_Test_Original.shape)
    print ('# # # # Prediction:', yhat_test.shape, yhat_test)

    Filename = 'Output_' + ModelName + '.csv'
    
    df_Output= pd.DataFrame()
    df_Output['card_id'] = Df_Test_Original['card_id']
    df_Output['target'] = yhat_test
    df_Output.to_csv(Filename, index = False)
    
    return yhat_test

#
# ! ! ! ! ! ! ! 
#

# 0) Let's Bagging:
Sel_Model = 'Bagging'
NameM = 'Bagging'
#Bagging(NameM, Sel_Model, df_test_original, X_train, X_test, df_train_original['target'])

# 0) Grid Search:
Sel_Model = 'GridS'
NameM = 'GridS'
#GridSearch(NameM, Sel_Model, df_test_original, X_train, X_test, df_train_original['target'])

#
#
#chose the Model that will run
#
#

# -1) Run Runlightgbm:
Sel_Model = lgb
NameM = 'lgb'
Runlightgbm(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 1) Gradiente Boost Model
Sel_Model = GBR
NameM = 'GBR'
MGBR = RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 2) Linear Regression is not possible beacuse Yhat should be Float. You can try to run if you want to.
Sel_Model = LR
NameM = 'LinearRegress'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 3) Random Forest is not possible beacuse Yhat should be Float. You can try to run if you want to.
Sel_Model = RF 
NameM = 'RanFor'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 4) xgboost Model (Megazord) XD - Didn't run, Kernel died
Sel_Model = XGB
NameM = 'XgBoost'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 5) KNeighbors Model is not possible beacuse Yhat should be Float. You can try to run if you want to.
Sel_Model = knn
NameM = 'KNeighbors'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 6) SVC Model is not possible beacuse Yhat should be Float. You can try to run if you want to.
Sel_Model = SVC
NameM = 'SVC'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 7) Decision Tree Model is not possible beacuse Yhat should be Float. You can try to run if you want to.
Sel_Model = DTC
NameM = 'DecisionTree'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 8) Gaussian is not possible - Unknown label type
Sel_Model = GNB
NameM = 'Gaussian'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 9) Neural Model is not possible - Unknown label type
Sel_Model = NN
NameM = 'NeuralModel'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 10) lasso
Sel_Model = lasso
NameM = 'lasso'
MLASSO = RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 11) Elastic Net
Sel_Model = ENet
NameM = 'ElasticNet'
MENET = RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 12) Kernel Ridge - Error Unknow
Sel_Model = KRR
NameM = 'NeuralModel'
#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 13) LGBMRegressor
Sel_Model = model_lgb
NameM = 'LGBMRegressor'
RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

# 14) Linear Regression
Sel_Model = LR2
NameM = 'LinearRegression'
RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)

#15) Ensemble with best values

ensemble = MGBR*0.70 + MLASSO*0.15 + MENET*0.15

ModelName = 'Ensemble'
Filename = 'Output_' + ModelName + '.csv'
    
df_Output= pd.DataFrame()
df_Output['card_id'] = df_test_original['card_id']
df_Output['target'] = ensemble
df_Output.to_csv(Filename, index = False)
