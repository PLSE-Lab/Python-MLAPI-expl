# data handling
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

# model
from sklearn.linear_model import ElasticNet, BayesianRidge, LassoLarsIC, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import xgboost as xgb
import lightgbm as lgb

# unsupervised learning
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# prevent overfit
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import mean_squared_error

# warning
import warnings
warnings.filterwarnings('ignore')


# data load
tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')
price = tr['price']

tr = tr.drop(columns='price')

def rmse(true, pred, exp):
    if exp==True:
        return np.sqrt(mean_squared_error(np.expm1(true), np.expm1(pred)))
    else:
        return np.sqrt(mean_squared_error(true, pred))
    
def distance_calculate(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    distance = R * c

    return distance


# basic preprocessing
def preprocessing(df, tr, te):
    # basic
    tr_idx = tr['id']
    te_idx = te['id']
    df = df.iloc[:, 1:]
    
    # date
    df['year_month'] = df['date'].apply(lambda x: x[:6])
    df['year'] = df['date'].apply(lambda x: x[:4])
    df.drop(columns=['date'], inplace=True)
    df = df.astype(float)
    
    # rooms
    df.loc[df['bedrooms']>5, 'bedrooms'] = 5
    df.drop(columns='bathrooms', inplace=True)
    
    # grade
    df.loc[df['grade']<4.0, 'grade'] = 4.0
    
    # renovated > year
    df.loc[df['year'] < df['yr_renovated'], 'yr_renovated'] = 0
    
    # zipcode detach
    df['zipcode'] = df['zipcode'].astype(str)
    df['zipcode-3'] = 'z_' + df['zipcode'].str[2:3]
    df['zipcode-4'] = 'z_' + df['zipcode'].str[3:4]
    df['zipcode-5'] = 'z_' + df['zipcode'].str[4:5]
    df['zipcode-34'] = 'z_' + df['zipcode'].str[2:4]
    df['zipcode-45'] = 'z_' + df['zipcode'].str[3:5]
    df['zipcode-35'] = 'z_' + df['zipcode'].str[2:3] + df['zipcode'].str[4:5]
    # labelencoder
    temp = ['zipcode', 'zipcode-3', 'zipcode-4', 'zipcode-5', 'zipcode-34', 'zipcode-45', 'zipcode-35']
    for temp_feature in temp:
        le = LabelEncoder().fit(df[temp_feature])
        df[temp_feature] = le.transform(df[temp_feature])
    
    # skew feature log1p
    continuous_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
    df[continuous_feature] = np.log1p(df[continuous_feature])
    
    # Kmeans1
    kmeans_df = df[['lat', 'long']]
    scaler = StandardScaler().fit(kmeans_df)
    kmeans = KMeans(n_clusters=22, random_state=1028).fit(scaler.transform(kmeans_df))
    df['kmeans_labels_1'] = kmeans.labels_
    
    # Kmeans2
    kmeans_df = df[['lat', 'long']]
    scaler = StandardScaler().fit(kmeans_df)
    kmeans = KMeans(n_clusters=94, random_state=1028).fit(scaler.transform(kmeans_df))
    df['kmeans_labels_2'] = kmeans.labels_
    
    # Kmeans3
    kmeans_df = df[['lat', 'long']]
    kmeans = KMeans(n_clusters=196, random_state=1028).fit(kmeans_df)
    df['kmeans_labels_3'] = kmeans.labels_
    
    # Kmeans4
    kmeans_df = df[['lat', 'long']]
    kmeans = KMeans(n_clusters=159, random_state=1028).fit(kmeans_df)
    df['kmeans_labels_4'] = kmeans.labels_
    
    # Kmeans5
    kmeans_df = df[['lat', 'long']]
    kmeans = KMeans(n_clusters=142, random_state=1028).fit(kmeans_df)
    df['kmeans_labels_5'] = kmeans.labels_
    
    # DBSCAN
    clustering = DBSCAN(eps=0.05, min_samples=5).fit(df[['lat', 'long']])
    df['dbscan'] = clustering.labels_
    
    # distance inner house count
    distance_count = np.zeros(len(df))
    for i in range(len(df)):
        distance_count[i] = np.sum(distance_calculate(df['lat'], df['long'], df['lat'][i], df['long'][i]) < 0.5)
    df['distance_count'] = distance_count
    
    distance_count2 = np.zeros(len(df))
    for i in range(len(df)):
        distance_count2[i] = np.sum(distance_calculate(df['lat'], df['long'], df['lat'][i], df['long'][i]) < 10)
    df['distance_count2'] = distance_count2
    
    distance_count3 = np.zeros(len(df))
    for i in range(len(df)):
        distance_count3[i] = np.sum(distance_calculate(df['lat'], df['long'], df['lat'][i], df['long'][i]) < 3)
    df['distance_count3'] = distance_count3
    
    distance_count4 = np.zeros(len(df))
    for i in range(len(df)):
        distance_count4[i] = np.sum(distance_calculate(df['lat'], df['long'], df['lat'][i], df['long'][i]) < 11)
    df['distance_count4'] = distance_count4
    
    return df

df = pd.concat([tr, te])
df.reset_index(drop=True, inplace=True)
df = preprocessing(df, tr, te)
price = np.log1p(price)    


# index
tr_idx = tr['id']
te_idx = te['id']


# prediction df
oof_df = pd.DataFrame()
pred_df = pd.DataFrame()

df2 = copy.deepcopy(df)

# XGB
tr = df2.loc[tr_idx]
te = df2.loc[te_idx].reset_index(drop=True)

random_state = 1028
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions



random_state = 6767
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




random_state = 33123
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




random_state = 201904
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




random_state = 2018
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




random_state = 2019
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




random_state = 71
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




random_state = 3
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




random_state = 9999
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




random_state = 54321
params={
    'gpu_id':0 ,         
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eta': 0.007,
    'max_depth': 6,
    'min_child_weight': 7,
    'n_estimators' : 60000,
    'subsample': 0.5
}
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_valid = xgb.DMatrix(test_X, label=test_y)
    
    reg = xgb.train(params, xgb_train, num_boost_round=10000000, evals=[(xgb_train, 'train'), (xgb_valid, 'val')], verbose_eval=10000, early_stopping_rounds=10000)
    
    oof[val_idx] = reg.predict(xgb.DMatrix(test_X))
    predictions = reg.predict(xgb.DMatrix(te))    
    print('---'*30)
    
oof_df['random_state'+str(random_state)] = oof
pred_df['random_state'+str(random_state)] = predictions




rmse(price, oof_df.mean(1), 1)


oof = oof_df.mean(1)
pred = pred_df.mean(1)



oof = pd.DataFrame({'oof':oof})


oof.to_csv('xgb_10_ensembles_oof.csv', index=False)

sub = pd.read_csv('../input/sample_submission.csv')
sub['price'] = pred
sub.to_csv('xgb_10_ensembles_pred.csv', index=False)






