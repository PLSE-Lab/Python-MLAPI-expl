#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Relative Strength Index
# Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
# Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)>0};
#        PriceDown(t)=-1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)<0};

def rsi(values):
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)


# In[ ]:


def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    #ave = pd.stats.moments.rolling_mean(price,length)
    ave = price.rolling(window = length, center = False).mean()
    #sd = pd.stats.moments.rolling_std(price,length)
    sd = price.rolling(window = length, center = False).std()
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return np.round(ave,3), np.round(upband,3), np.round(dnband,3)


# In[ ]:


def aroon(df, tf=25):
    aroonup = []
    aroondown = []
    x = tf
    while x< len(df):
        aroon_up = ((df['high'][x-tf:x].tolist().index(max(df['high'][x-tf:x])))/float(tf))*100
        aroon_down = ((df['low'][x-tf:x].tolist().index(min(df['low'][x-tf:x])))/float(tf))*100
        aroonup.append(aroon_up)
        aroondown.append(aroon_down)
        x+=1
    return aroonup, aroondown


# In[ ]:


def abands(df):
    #df['AB_Middle_Band'] = pd.rolling_mean(df['Close'], 20)
    df['AB_Middle_Band'] = df['close'].rolling(window = 20, center=False).mean()
    # High * ( 1 + 4 * (High - Low) / (High + Low))
    df['aupband'] = df['high'] * (1 + 4 * (df['high']-df['low'])/(df['high']+df['low']))
    df['AB_Upper_Band'] = df['aupband'].rolling(window=20, center=False).mean()
    # Low *(1 - 4 * (High - Low)/ (High + Low))
    df['adownband'] = df['low'] * (1 - 4 * (df['high']-df['low'])/(df['high']+df['low']))
    df['AB_Lower_Band'] = df['adownband'].rolling(window=20, center=False).mean()


# In[ ]:


def STOK(df, n):
    df['STOK'] = ((df['close'] - df['low'].rolling(window=n, center=False).mean()) / (df['high'].rolling(window=n, center=False).max() - df['low'].rolling(window=n, center=False).min())) * 100
    df['STOD'] = df['STOK'].rolling(window = 3, center=False).mean()


# In[ ]:


def psar(df, iaf = 0.02, maxaf = 0.2):
    length = len(df)
#     dates = (df['Date'])
    high = (df['high'])
    low = (df['low'])
    close = (df['close'])
    psar = df['close'][0:len(df['close'])]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = df['low'][0]
    hp = df['high'][0]
    lp = df['low'][0]
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if df['low'][i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = df['low'][i]
                af = iaf
        else:
            if df['high'][i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = df['high'][i]
                af = iaf
        if not reverse:
            if bull:
                if df['high'][i] > hp:
                    hp = df['high'][i]
                    af = min(af + iaf, maxaf)
                if df['low'][i - 1] < psar[i]:
                    psar[i] = df['low'][i - 1]
                if df['low'][i - 2] < psar[i]:
                    psar[i] = df['low'][i - 2]
            else:
                if df['low'][i] < lp:
                    lp = df['low'][i]
                    af = min(af + iaf, maxaf)
                if df['high'][i - 1] > psar[i]:
                    psar[i] = df['high'][i - 1]
                if df['high'][i - 2] > psar[i]:
                    psar[i] = df['high'][i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    #return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}
    #return psar, psarbear, psarbull
    df['psar'] = psar
    #df['psarbear'] = psarbear
    #df['psarbull'] = psarbull


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
window = 12
# starting_ids = [0, 7275, 14550,21765, 29040]
def preprocess (data_in):
    starting_ids = []
    data = data_in.copy()
    for k in range(5):
        starting_ids.append(data[data.asset == k]['id'].min())
    # data['asset'] = data['asset'].astype('int')
    data['diff'] = data['high'] - data['low']
    data['diff*volume'] = data['diff'] * data['volume']
    data ['vol/trades'] = data.volume / data.trades
#     id_diff = []
#     for index, row in data.iterrows():
#         id_diff.append(row.id - starting_ids[int(row.asset)])
        
#     data['idd_diff'] = id_diff
    #data = pd.get_dummies(data, columns=['asset'])
    data.drop('id',axis = 1, inplace = True)
    data.drop('asset',axis = 1, inplace = True)
    #data.drop('macd_signal',axis = 1, inplace = True)
#     poly = PolynomialFeatures(2)
#     data = poly.fit_transform(data)
#     rolling_12_min  = data.rolling(window=12).min().add_suffix('_12_min')
#     rolling_12_max  = data.rolling(window=12).max().add_suffix('_12_max')
#     rolling_12_mean  = data.rolling(window=12).mean().add_suffix('_12_mean')
    
#     rolling_1_min  = data.rolling(window=1).min().add_suffix('_1_min')
#     #print (rolling_1_min.head().T)
#     rolling_1_max  = data.rolling(window=1).max().add_suffix('_1_max')
#     rolling_1_mean  = data.rolling(window=12).mean().add_suffix('_1_mean')
    
#     data = pd.concat([data,rolling_12_min,rolling_12_max, rolling_12_mean, rolling_1_min,rolling_1_max, rolling_1_mean ], axis = 1)
    data['Momentum_1D'] = (data['close']-data['close'].shift(1)).fillna(0)
    data['RSI_14D'] = data['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)
    data['RSI_1D'] = data['Momentum_1D'].rolling(center=False, window=1).apply(rsi).fillna(0)
    data['RSI_2D'] = data['Momentum_1D'].rolling(center=False, window=2).apply(rsi).fillna(0)
    data['RSI_4D'] = data['Momentum_1D'].rolling(center=False, window=4).apply(rsi).fillna(0)
    data['RSI_8D'] = data['Momentum_1D'].rolling(center=False, window=8).apply(rsi).fillna(0)
    
    data['BB_Middle_Band'], data['BB_Upper_Band'], data['BB_Lower_Band'] = bbands(data['close'], length=20, numsd=1)
    data['BB_Middle_Band'] = data['BB_Middle_Band'].fillna(0)
    data['BB_Upper_Band'] = data['BB_Upper_Band'].fillna(0)
    data['BB_Lower_Band'] = data['BB_Lower_Band'].fillna(0)
    
    listofzeros = [0] * 25
    up, down = aroon(data)
    aroon_list = [x - y for x, y in zip(up,down)]
    if len(aroon_list)==0:
        aroon_list = [0] * data.shape[0]
        data['Aroon_Oscillator'] = aroon_list
    else:
        data['Aroon_Oscillator'] = listofzeros+aroon_list
        
    data["PVT"] = (data['Momentum_1D']/ data['close'].shift(1))*data['volume']
    data["PVT"] = data["PVT"] - data["PVT"].shift(1)
    data["PVT"] = data["PVT"].fillna(0)
    
    abands(data)
    data.fillna(0)
    
    STOK(data, 4)
    data.fillna(0)
    
#     psar(data)
#     data.fillna(0)
    
    data['ROC'] = ((data['close'] - data['close'].shift(12))/(data['close'].shift(12)))*100
    data.fillna(0)
    
#     for stock in range(len(data)):
    data['VWAP'] = np.cumsum(data['volume'] * (data['high'] + data['low'])/2) / np.cumsum(data['volume'])
    data.fillna(0)
    
    scaler = StandardScaler(with_std=False)
    data = scaler.fit_transform(data)
    return data


# In[ ]:


data = pd.read_csv('../input/train.csv')
target = data.pop('y')
data_test = pd.read_csv('../input/test.csv')


# In[ ]:


target_by_asset = []
test_by_asset = []
train_by_asset = []
PredID = []
preds_df = []
scores = []
best_params = []
grid_scores = []
grid_preds_df = []
for k in range(5):
    train_by_asset.append( data[data['asset']==k])
    target_by_asset.append( target[data['asset']==k])
    PredID.append(data_test[data_test['asset']==k]['id'])
    train_by_asset[k] = preprocess(train_by_asset[k])
    test_by_asset.append(data_test[data_test['asset']==k])
    test_by_asset[k] = preprocess(test_by_asset[k])
    
    #print(train_by_asset[k])
    
   # print(train_by_asset,target_by_asset,PredID,test_by_asset)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_by_asset[k], target_by_asset[k], test_size=0.33, random_state=777, shuffle=False) 

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'colsample_bytree': 0.65,
        'learning_rate': 0.005,
        'n_estimators': 40,
        'num_leaves': 16,
        'objective': 'regression',
        'random_state': 501,
        'reg_alpha': 1,
        'reg_lambda': 0,
        'subsample': 0.7,
        'max_depth' : -1,
        'max_bin': 512,
        'subsample_for_bin': 200,
        'subsample_freq': 1,
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 5,
        'scale_pos_weight': 1,
    }
    
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
    
    mdl = lgb.LGBMRegressor(boosting_type= 'gbdt',
          objective = 'regrssion',
          n_jobs = -1, # Updated from 'nthread'
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])
    
    gridParams = {
        'learning_rate': [0.005],
        'n_estimators': [40],
        'num_leaves': [6,8,12,16],
        'boosting_type' : ['gbdt'],
        'objective': ['regression'],
        'random_state' : [501], # Updated from 'seed'
        'colsample_bytree' : [0.65, 0.66],
        'subsample' : [0.7,0.75],
        'reg_alpha' : [0,1,1.2],
        'reg_lambda' : [0,1,1.2,1.4],
    }

    grid = RandomizedSearchCV(mdl, gridParams,
                    verbose=0,
                    cv=4,
                    n_jobs=-1)
    
    grid.fit(train_by_asset[k], target_by_asset[k])

    # Print the best parameters found
    
    print(grid.best_params_)
    print(grid.best_score_)
    best_params.append(grid.best_params_)
    grid_scores.append(grid.best_score_)
    
    grid_pred = grid.predict(test_by_asset[k])
    grid_pred[:12] = np.zeros(12)
    out = pd.DataFrame(PredID[k])
    out['expected'] = grid_pred
    #print(out)
    grid_preds_df.append(out)
    
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    scores.append(mean_squared_error(y_test, y_pred) ** 0.5)
    
    
    preds = gbm.predict(test_by_asset[k], num_iteration=gbm.best_iteration)
    preds[:12] = np.zeros(12)
    out = pd.DataFrame(PredID[k])
    out['expected'] = preds
    #print(out)
    preds_df.append(out)


# In[ ]:


print(scores)
print(np.mean(scores),np.std(scores))


# [0.786891395468127, 0.6973549079908655, 0.7254750063290656, 0.7244248036717815, 0.759813275450325]
# 0.738791877782033 0.031164842726449895
# 
# rollig (wrong?)
# 
# [0.7868569153648659, 0.6965938932773268, 0.7253140229115677, 0.7244046020382962, 0.7591281367246425]
# 0.7384595140633399 0.031281224748745236
# 
# using bands
# 
# [0.7865777258814584, 0.6956495616477514, 0.7261818906168533, 0.7245652742462407, 0.757114859842344]
# 0.7380178624469296 0.031109133441386192
# 
# 
# 

# In[ ]:


print(grid_scores)
print(np.mean(grid_scores),np.std(grid_scores))


# [0.0034013447836041366, -0.007779764770281005, 0.00025671830563126283, -0.008346788122561203, -0.010524651325048163]
# -0.004598628225730994 0.005419594822320587
# 
# rolling (wrong?)
# 
# [0.00474713090944534, -0.006542880441062967, 0.0010371953974309812, -0.007956894859394197, -0.011771988264293598]
# -0.004097487451574888 0.006072317136793158
# 
# rolling
# 
# [-0.0007421783216381947, -0.005759116230089936, 0.0029404735897728, -0.0099340629682664, -0.013260140730498344]
# -0.005351004932144015 0.005894324451129263

# In[ ]:


out_df = pd.DataFrame(np.concatenate(preds_df), columns=['id','expected'])
out_df.id = out_df.id.astype('int')
out_df.to_csv('by_asset_lgbm.csv',index = False)


# In[ ]:


out_df = pd.DataFrame(np.concatenate(grid_preds_df), columns=['id','expected'])
out_df.id = out_df.id.astype('int')
out_df.to_csv('grid_by_asset.csv',index = False)


# In[ ]:


from sklearn.linear_model import ElasticNetCV
regr = ElasticNetCV(cv=5, random_state=0, l1_ratio = [.1, .5, .7, .9, .95, .99, 1])
scores = []
target_by_asset = []
test_by_asset = []
train_by_asset = []
PredID = []
preds_df = []
scores = []
best_params = []
grid_scores = []
grid_preds_df = []
for k in range(5):
    train_by_asset.append( data[data['asset']==k])
    target_by_asset.append( target[data['asset']==k])
    PredID.append(data_test[data_test['asset']==k]['id'])
    train_by_asset[k] = preprocess(train_by_asset[k])
    test_by_asset.append(data_test[data_test['asset']==k])
    test_by_asset[k] = preprocess(test_by_asset[k])
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(train_by_asset[k],0), target_by_asset[k], test_size=0.33, random_state=777, shuffle=False) 
    
    regr.fit(X_train,y_train)
    y_pred = regr.predict (X_test)
#     
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    scores.append(mean_squared_error(y_test, y_pred) ** 0.5)
    
    preds = regr.predict(np.nan_to_num(test_by_asset[k],0))
    out = pd.DataFrame(PredID[k])
    out['expected'] = preds
    #print(out)
    preds_df.append(out)


# In[ ]:


print(scores)
print(np.mean(scores),np.std(scores))


# Elastic without windows
# 
# [0.7843381203753828, 0.6991611658363793, 0.72681852540914, 0.7244268066276227, 0.7605812564714087]
# 0.7390651749439867 0.02989706857530931

# In[ ]:


out_df = pd.DataFrame(np.concatenate(preds_df), columns=['id','expected'])
out_df.id = out_df.id.astype('int')
out_df.to_csv('Elastic_net.csv',index = False)


# In[ ]:




