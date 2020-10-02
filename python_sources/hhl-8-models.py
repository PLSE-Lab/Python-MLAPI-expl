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


# data load`
tr = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')
te = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')
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



# LGBM
df2 = copy.deepcopy(df)

tr = df2.loc[tr_idx]
te = df2.loc[te_idx].reset_index(drop=True)

random_state=1028
param = {
    "objective" : "regression",
    "metric" : "rmse",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 5,
    "min_data_in_leaf": 5,
    "bagging_freq": 10,
    "learning_rate" : 0.006,
    "bagging_fraction" : 0.464,
    "feature_fraction" : 0.582,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "lambda_l1" : 0.524,
    "lambda_l2" : 0.302,
    "verbosity" : 1,
    "feature_fraction_seed" : random_state,
    "bagging_fraction_seed" : random_state,
    "random_state": random_state
}   
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    lgb_train = lgb.Dataset(train_X, label=train_y)
    lgb_valid = lgb.Dataset(test_X, label=test_y)

    clf = lgb.train(param, lgb_train, 100000, valid_sets = [lgb_train, lgb_valid], early_stopping_rounds = 10000, verbose_eval=10000)

    oof[val_idx] = clf.predict(test_X, num_iteration=clf.best_iteration)
    predictions += clf.predict(te, num_iteration=clf.best_iteration)/splits
    print('---'*30)
    
oof_df['lgbm'] = oof
pred_df['lgbm'] = predictions



# XGB
oof = pd.read_csv('../input/hhl-xgb-10-ensembles/xgb_10_ensembles_oof.csv')
pred = pd.read_csv('../input/hhl-xgb-10-ensembles/xgb_10_ensembles_pred.csv')
oof_df['xgb'] = oof['oof']
pred_df['xgb'] = pred['price']





# GBoost
df2 = copy.deepcopy(df)
tr = df2.loc[tr_idx]
te = df2.loc[te_idx].reset_index(drop=True)

random_state=1028
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    reg_gboost = GradientBoostingRegressor(n_estimators=8000, 
                                            learning_rate=0.01, 
                                            max_depth=5, 
                                            max_features='sqrt', 
                                            min_samples_leaf=15, 
                                            min_samples_split=10, 
                                            loss='huber', 
                                            random_state =random_state).fit(train_X, train_y)
    
    oof[val_idx] = reg_gboost.predict(test_X)
    predictions += reg_gboost.predict(te)/splits
    print('---'*30)

oof_df['gboost'] = oof
pred_df['gboost'] = predictions




# Adaboost
df2 = copy.deepcopy(df)
tr = df2.loc[tr_idx]
te = df2.loc[te_idx].reset_index(drop=True)

random_state=1028
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    reg_adaboost = AdaBoostRegressor(n_estimators=5000, loss='square').fit(train_X, train_y)
    
    oof[val_idx] = reg_adaboost.predict(test_X)
    predictions += reg_adaboost.predict(te)/splits
    print('---'*30)

oof_df['reg_adaboost'] = oof
pred_df['reg_adaboost'] = predictions




# RandomForest
df2 = copy.deepcopy(df)
tr = df2.loc[tr_idx]
te = df2.loc[te_idx].reset_index(drop=True)

random_state=1028
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    reg = RandomForestRegressor(n_estimators=500, random_state=random_state).fit(train_X, train_y)
    
    a = reg.predict(test_X)
    oof[val_idx] = reg.predict(test_X)
    predictions += reg.predict(te)/splits
    print('---'*30)

oof_df['rf'] = oof
pred_df['rf'] = predictions



# Ridge
df2 = copy.deepcopy(df)

continuous_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'lat', 'long', 'distance_count', 'distance_count2', 'distance_count3', 'distance_count4']
categorical_feature = list(set(df2.columns) - set(continuous_feature))

sscaler = StandardScaler().fit(df2[continuous_feature])
df2[continuous_feature] = sscaler.transform(df2[continuous_feature])
df2[categorical_feature] = df2[categorical_feature].astype(object)
df2 = pd.get_dummies(df2)

tr = df2.loc[tr_idx]
te = df2.loc[te_idx].reset_index(drop=True)
coef = pd.DataFrame()

random_state=1028
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

for idx, (trn_idx, val_idx) in enumerate(kf.split(tr, price)):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    reg = BayesianRidge().fit(train_X, train_y)
    
    coef[idx] = reg.coef_
    print(str(idx+1) + '---')
    
high_coef_feature = pd.DataFrame({'feature':tr.columns, 'coef':np.abs(coef.mean(1))}).sort_values('coef', ascending=False).reset_index(drop=True).loc[:640].feature

tr = tr[high_coef_feature]
te = te[high_coef_feature]

random_state=1028
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))
coef = pd.DataFrame()

for idx, (trn_idx, val_idx) in enumerate(kf.split(tr, price)):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    reg = Ridge().fit(train_X, train_y)
    
    oof[val_idx] = reg.predict(test_X)
    predictions += reg.predict(te)/splits
    
    print(str(idx+1) + '---')
oof_df['ridge'] = oof
pred_df['ridge'] = predictions




# SVR
random_state=1028
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    reg_svr = SVR(kernel='linear', C=0.5).fit(train_X, train_y)
    
    oof[val_idx] = reg_svr.predict(test_X)
    predictions += reg_svr.predict(te)/splits
    print('---'*30)

oof_df['reg_svr'] = oof
pred_df['reg_svr'] = predictions



# KNN
random_state=1028
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = tr.loc[trn_idx], price[trn_idx]
    test_X, test_y = tr.loc[val_idx], price[val_idx]
    
    reg_knn = KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='kd_tree').fit(train_X, train_y)

    oof[val_idx] = reg_knn.predict(test_X)
    predictions += reg_knn.predict(te)/splits
    print('---'*30)

oof_df['knn'] = oof
pred_df['knn'] = predictions





oof_df.to_csv('oof_results.csv', index=False)
pred_df.to_csv('pred_results.csv', index=False)
