#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, random, gc, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from kaggle.competitions import nflrush
import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial import distance_matrix
from scipy.stats import cumfreq
from scipy.optimize import nnls
import lightgbm as lgb
from joblib import Parallel, delayed
import multiprocessing
import keras
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.engine.saving import load_model
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.layers import Input, Concatenate
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, Activation, PReLU, Add
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import metrics
from collections import Counter
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree, ConvexHull 
from shapely.ops import polygonize, unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from shapely.geometry import shape, mapping
import math
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


mname = 'zoo_keras'
# path = '/kaggle/input/nfl-big-data-bowl-2020/'
path = './'
build_data = True
search = False
nrep = 10
patience = 21
w_holdout = True
perm = False


# In[ ]:


np.set_printoptions(linewidth=200, precision=6, suppress=True)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.width', 200)


# In[ ]:


# standardize coordinates, angles, yardline so offense is always driving to the right
# https://www.kaggle.com/kernels/scriptcontent/21906255/
def std_cols(df):
    
    # fix inconsistent team abbreviations
    ita = {'ARZ': 'ARI', 'BLT':'BAL', 'CLV':'CLE', 'HST':'HOU'}
    update = ['PossessionTeam', 'FieldPosition']
    for col in update:
        for old, new in ita.items():
            df.loc[df[col] == old,col] = new
                
    df['X'] = df.apply(lambda x: x.X if x.PlayDirection == 'right'                           else 120-x.X, axis=1) 
    
    df['Y'] = df.apply(lambda x: x.Y if x.PlayDirection == 'right'                           else 53.3-x.Y, axis=1) 
    
#     # adjust 2017 Orientation as it differs by 90 degrees from 2018 and 2019
#     df.loc[df.Season == 2017, 'Orientation'] = np.mod(df.Orientation + 90, 360)
#     # set angles so 0 degrees is directly downfield for rusher and range -180 to 180
#     df['Orientation'] = df.apply(lambda x: 180 - np.mod(x.Orientation + 90 \
#                                      if x.PlayDirection == 'right' \
#                                      else x.Orientation - 90, 360), axis=1)
    
    df['Dir'] = df.apply(lambda x: 180 - np.mod(x.Dir + 90                                      if x.PlayDirection == 'right'                                      else x.Dir - 90, 360), axis=1)
    
    df['YardLine'] = df.apply(lambda x: x.YardLine + 10                               if (x.FieldPosition == x.PossessionTeam)                               else 60 + (50-x.YardLine), axis=1)
    
    df.loc[:, 'S'] = 10 * df['Dis']
    
        
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "sns.set_style('darkgrid')\nmpl.rcParams['figure.figsize'] = [15,10]\n\ntrain = pd.read_csv(path + 'train1.csv', dtype={'WindSpeed': 'object'})\nprint(train.shape)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# train0 = train.copy()\ntrain = std_cols(train)\n# train_df = train.copy()\ntrain_df = train\nprint(train_df.shape)')


# In[ ]:


df = train_df
mode = 'train'
verbose = True

# make a copy so as not to alter the original df
df = df.copy()

df['OffenseDefense'] = df.apply(lambda x: "Offense" if ((x.Team == 'home')                                  & (x.PossessionTeam ==                                     x.HomeTeamAbbr)) |                                 ((x.Team == 'away') &                                  (x.PossessionTeam ==                                   x.VisitorTeamAbbr))                                 else "Defense", axis=1)

df['IsRusher'] = df['NflId'] == df['NflIdRusher']

df.loc[df.IsRusher, 'OffenseDefense'] = "Rusher"

keep = ['GameId','PlayId','X','Y','Dir','S','IsRusher','OffenseDefense']

df = df[keep]

if verbose:
    print(df.shape)

# flip defense direction
# df.loc[df.OffenseDefense=='Defense','Dir'] = df.loc[df.OffenseDefense=='Defense','Dir'] + 180

newdir = df.Dir
# newdir = 0.99*df.Dir + 0.01*df.Orientation
df['SX'] = df.S * np.cos(newdir/180*np.pi) 
df['SY'] = df.S * np.sin(newdir/180*np.pi)

bdf = df.loc[df.IsRusher,['PlayId','X','Y','SX','SY']]
bdf.columns = ['PlayId','Ball_X','Ball_Y','Ball_SX','Ball_SY']

# bdf = df.loc[df.IsRusher,['PlayId','X','Y','S','Dir']]
# bdf.columns = ['PlayId','Ball_X','Ball_Y','Ball_S','Ball_Dir']

df = df.merge(bdf, how='left', on='PlayId')

df['XR'] = df.X - df.Ball_X
df['YR'] = df.Y - df.Ball_Y

df['SXR'] = df.SX - df.Ball_SX
df['SYR'] = df.SY - df.Ball_SY 

# df['SXR'] = (df.S - df.Ball_S) * np.cos(df.Dir/180*np.pi) 
# df['SYR'] = (df.S - df.Ball_S) * np.sin(df.Dir/180*np.pi) 


print(df.head(), df.shape)

# return X


# In[ ]:


k = ['PlayId','XR','YR','SX','SY','SXR','SYR']
# b = ['Ball_X','Ball_Y','Ball_SX','Ball_SY']
# o = df.loc[df.OffenseDefense=='Offense',k]
o = df.loc[df.OffenseDefense=='Offense',k]
d = df.loc[df.OffenseDefense=='Defense',k]
o.columns = ['PlayId','XRO','YRO','SXO','SYO','SXRO','SYRO']
d.columns = ['PlayId','XRD','YRD','SXD','SYD','SXRD','SYRD']
print(o.shape, d.shape)


# In[ ]:


od = o.merge(d, how='outer', on='PlayId')
print(od.shape)


# In[ ]:


od.shape[0]/23171


# In[ ]:


od.head()


# In[ ]:


od['XOD'] = od.XRO - od.XRD
od['YOD'] = od.YRO - od.YRD
od['SXOD'] = od.SXO - od.SXD
od['SYOD'] = od.SYO - od.SYD
print(od.head, od.shape)


# In[ ]:


od.drop(['PlayId','XRO','YRO','SXO','SYO','SXRO','SYRO'], axis=1, inplace=True)
print(od.shape, od.shape[0]/110)


# In[ ]:


od.describe()


# In[ ]:


od.head()


# In[ ]:


# cols = od.columns
# scaler = preprocessing.StandardScaler()
# od = scaler.fit_transform(od)
# od = np.nan_to_num(od)
# od = pd.DataFrame(od, columns=cols)
# od.describe()


# In[ ]:


# reshape to 4d: play, off, def, feature
x4 = od.values.reshape(-1,10,11,od.shape[1])
print(x4.shape)


# In[ ]:


# x4 = x4.transpose(0, 2, 1, 3)
# print(x4.shape)


# In[ ]:


x4[0,0,:5]


# In[ ]:


# %%time
# if build_data:
#     X0 = cruncher0(train_df, mode='train', verbose=True)


# In[ ]:


np.isnan(x4).mean()


# In[ ]:


x4 = np.nan_to_num(x4)


# In[ ]:


np.isnan(x4).mean()


# In[ ]:


x4_train = x4.copy()


# In[ ]:


# target
n = len(train_df) // 22
print(n)

y01_train = train_df["Yards"][::22].values.copy()

y0_train = np.zeros(shape=(n, 199))
for i,yard in enumerate(train_df['Yards'][::22]):
    y0_train[i, yard+99:] = np.ones(shape=(1, 100-yard))


# limit target range
y = y01_train.copy()
MIN = -10
MAX = 35
# MIN = -30
# MAX = 50
y[y < MIN] = MIN
y[y > MAX] = MAX
y -= MIN

num_class = MAX - MIN + 1

y_train = np.zeros(shape=(n, num_class))
for i, yard in enumerate(y):
    y_train[i, yard:] = np.ones(shape=(1, num_class-yard))

y1_train = y

# y_train = np.zeros(len(y_train_),dtype=np.float)
# for i in range(len(y_train)):
#     y_train[i] = (y_train_[i])

# scaler = preprocessing.StandardScaler()
# scaler.fit([[y] for y in y_train])
# y_train = np.array([y[0] for y in scaler.transform([[y] for y in y_train])])

print(y0_train.shape, y01_train.shape)
print(y_train.shape, y1_train.shape)


# In[ ]:


# y_true is a vector of scalars and y_pred cdfs
def crps0(y_true, y_pred):
    ans = 0
    for i, y in enumerate(y_true):
        h = np.zeros(199)
        yf = int(np.floor(y))
        h[(yf+99):] = 1.0
                
        ans += mean_squared_error(h, y_pred[i])
        
    return ans / (len(y_true))


# In[ ]:


# enforce monotonicity
def mono(p):
    for pred in p:
        prev = 0
        for i in range(len(pred)):
            if pred[i] < prev:
                pred[i] = prev
            prev = pred[i]
    return p


# In[ ]:


lgb_params = {
#     'device': 'gpu',
    'objective':'regression_l1',
#     'is_unbalance': True,
    'boosting_type':'gbdt',
    'metric': 'l1',
    'n_jobs': -1,
    'learning_rate': 0.01,
    'num_leaves': 2**6,
    'max_depth': 4,
    'tree_learner':'serial',
    'colsample_bytree': 0.7,
#     'subsample_freq': 1,
    'subsample': 0.7,
    'max_bin': 255,
    'verbose': -1,
    'seed': 123,
} 

# parallelize lgb predictor
# def gb_cox(tr_yc, v):
#     cf = cumfreq(tr_yc + v, numbins=199, defaultreallimits=(-99,100))
#     return cf.cumcount / len(tr_yc) 

def gb_cox(tr_yc, v):
    cf = cumfreq(tr_yc + v, numbins=num_class, defaultreallimits=(0,num_class))
    return cf.cumcount / len(tr_yc) 


# In[ ]:


# adjust predictions by modified yardline
def adjusty(p, df, y_true=None, reduced=True):
    n = len(p)
#     p = np.cumsum(p, axis=1)
#     p = np.clip(p, 0, 1)
    if reduced:
        pred = np.zeros((n, 199))        
        pred[:, (99+MIN):(100+MAX)] = p
        pred[:, 100+MAX:] = 1
    else:
        pred = p
    cdf = pred.copy()
    for i in range(0,n):
        r = i*22
        y = df["YardLine"].iloc[r] - 10
        
        if y < 99: cdf[i,:(100-y-1)] = 0
        if y > 1: cdf[i,-(y-1):] = 1
                
        # check for improvement, should never be worse
        if y_true is not None:
            mse_orig = mean_squared_error(y_true[i], pred[i])
            mse_new = mean_squared_error(y_true[i], cdf[i])
            if (mse_new > mse_orig):
                print('adjusty inconsistency', i, df["FieldPosition"].iloc[r],
                      df["PossessionTeam"].iloc[r],
                      df["YardLine"].iloc[r], y, df["Yards"].iloc[r], mse_orig, mse_new)
                print(y_true[i])
                print(pred[i])
                print(cdf[i])
                break
            
    return cdf


# In[ ]:


# adjust predictions by modified yardline
def adjusty2(p, yardline, y_true=None, reduced=True):
    n = len(yardline)
    if reduced:
#         p = np.cumsum(p, axis=1)
#         p = np.clip(p, 0, 1)
        pred = np.zeros((n, 199))        
        pred[:, (99+MIN):(100+MAX)] = p
        pred[:, 100+MAX:] = 1
    else:
        # pred = np.clip(p, 0, 1)
        pred = p
    cdf = pred.copy()
    for i in range(0,n):
        y = yardline[i]
        
        if y < 99: cdf[i,:(100-y-1)] = 0
        if y > 1: cdf[i,-(y-1):] = 1
                
        # check for improvement, should never be worse
        if y_true is not None:
            mse_orig = mean_squared_error(y_true[i], pred[i])
            mse_new = mean_squared_error(y_true[i], cdf[i])
            if (mse_new > mse_orig):
                print('adjusty2 inconsistency', i, y, mse_orig, mse_new)
                print(y_true[i])
                print(pred[i])
                print(cdf[i])
                break
            
    return cdf


# In[ ]:


def permutation_importance(X, y, model, func, better='smaller', nrep=5): 
    perm = {}
    pred = model.predict(X)
    baseline = func(y, pred)
    print('\nPermutation Importance Baseline Score', baseline)
    for i, c in enumerate(X.columns):
        values = X[c].values.copy()
        dtype = X[c].dtype.name
        score = 0.0
        for r in range(nrep):
            X[c] = np.random.permutation(values)
            X[c] = X[c].astype(dtype) 
            pred = model.predict(X)
            score = score + func(y, pred)
        if better=='smaller':
            perm[c] = score/nrep - baseline
        else:
            perm[c] = baseline - score/nrep
        X[c] = values.copy()
        X[c] = X[c].astype(dtype) 
        print(f'{i} {perm[c]:11.8f} {c}')
    
    df = pd.DataFrame.from_dict(perm, orient='index').reset_index()
    df.columns = ['Feature','Perm']
    
    return df


# In[ ]:


# feature list and X_list assumed to be nested lists of same sizes, X_list contains numpy arrays
def permutation_importance_list(feature_list, X_list, y, yardline, model, func, better='smaller', nrep=5): 
    perm = {}
    p = model.predict(X_list)
    p = mono(p)
    pred = adjusty2(p, yardline, y_true=y)
    baseline = func(y, pred)
    print('\npermutation importance baseline score', baseline)
    for feat, X in zip(feature_list, X_list):
        if len(feat) == 0: continue
        for c, f in enumerate(feat):
            values = X[...,c].copy()
            score = 0.0
            for r in range(nrep):
                X[...,c] = np.random.permutation(values)
                p = model.predict(X_list)
                p = mono(p)
                pred = adjusty2(p, yardline, y_true=y)
                score = score + func(y, pred)
            if better=='smaller':
                perm[f] = score/nrep - baseline
            else:
                perm[f] = baseline - score/nrep
            X[...,c] = values.copy()
            print(f'{c} {perm[f]:.7f} {f}')
    
    df = pd.DataFrame.from_dict(perm, orient='index').reset_index()
    df.columns = ['Feature','Perm']
    
    return df


# In[ ]:


keras.backend.clear_session()
import keras.backend as K
def crps(y_true, y_pred):
    loss = K.mean((K.cumsum(y_pred, axis = 1) - y_true)**2)
    return loss


# In[ ]:


# https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400#latest-683614
keras.backend.clear_session()

def build_model(inp1, inp2, inp3, units=128, print_summary=False):
    
    keras.backend.clear_session()
    gc.collect()
    
    # inputs
    inputs = keras.layers.Input(shape=(inp1,inp2,inp3))
    
    # 4D
    x = keras.layers.Conv2D(128,(1,1),activation='relu')(inputs)
    x = keras.layers.Conv2D(160,(1,1),activation='relu')(x)
    x = keras.layers.Conv2D(128,(1,1),activation='relu')(x)
    a = keras.layers.AveragePooling2D(pool_size=(inp1,1))(x)
    a = keras.layers.Lambda(lambda x1 : x1*0.7)(a)
    m = keras.layers.MaxPooling2D(pool_size=(inp1,1))(x)
    m = keras.layers.Lambda(lambda x1 : x1*0.3)(m)
    x = keras.layers.Add()([a,m])
    x = keras.layers.Reshape((inp2,units))(x)
    x = keras.layers.BatchNormalization()(x)

    # 3D
    x = keras.layers.Conv1D(160,(1),activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(96,(1),activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(96,(1),activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    a = keras.layers.AveragePooling1D(pool_size=inp2)(x)
    m = keras.layers.MaxPooling1D(pool_size=inp2)(x)
    x = keras.layers.Average()([a,m])
    x = keras.layers.Flatten()(x)

    # 2D
    x = keras.layers.Dense(96, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Dropout(0.05)(x)  
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(num_class, activation='sigmoid')(x)
#     x = keras.layers.Dense(num_class, activation='softmax')(x)
    
    model = keras.models.Model(inputs = [inputs], outputs = [x])
    
    opt = keras.optimizers.Adam(learning_rate=2e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss='mse')
    
#     model.compile(optimizer='adam', loss='mse')
#     model.compile(optimizer='sgd', loss='mse')
#     model.compile(optimizer='adam', loss=crps)
    
    if print_summary: print(model.summary())
        
    return model


# In[ ]:


df.reset_index(drop=True, inplace=True)
print(df.shape)


# In[ ]:


df1 = df[::22].reset_index(drop=True)
print(df1.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', "if w_holdout:\n\n#     train_df['week'] = train_df.groupby('PossessionTeam')['GameId'].rank(method='dense')\n#     print(train_df['week'].describe())\n#     dfw = train_df[['GameId','week']][::22].copy().reset_index(drop=True)\n    y_pred = np.zeros((df.shape[0], 199)) \n\n#     if perm: nrepw = 1\n#     else: nrepw = 10\n    nrepw = 10\n    nepoch = 100\n    batch_size = 64\n    units = [199] * nrepw\n    max_depths = [5] * nrepw\n\n    ncores = multiprocessing.cpu_count()\n\n    if perm:\n        os.makedirs('imp',exist_ok=True)\n\n    # collect modeling results in these lists\n    models_nn = []\n    models_lgb = []\n    models_lgb_bi = []\n    models_tr_yc = []\n    models_w = []\n    models_h = []\n\n    ecdfs = []\n    escores = []\n    bscores = []\n    bscorew = []\n    bscorea = []\n    vscores = []\n\n    n = len(y1_train)\n    print('nn train shape', x4_train.shape)\n    # print('lgb train shape', X_train1.shape)\n    # nn_features = list(X_train0.columns)\n    # lgb_features = list(X_train1.columns)\n    os.makedirs('imp', exist_ok=True)\n    first = True\n\n    for nfold in [1]:\n\n        # kfold = KFold(n_splits=nfold, shuffle=False)\n\n        # kfold = GroupKFold(n_splits=K)\n        # groups = train_df['GameId'][::22]\n\n        # groups = 10 * train_df['Season'][::22] + train_df['Week'][::22]\n\n        # kfold = StratifiedKFold(n_splits = K, \n        #                             random_state = 231, \n        #                             shuffle = True)    \n\n\n        # full_val_preds = np.zeros((n))\n        full_val_preds = np.zeros((n,199))\n\n        # test_preds = np.zeros((np.shape(X_test)[0],K))\n\n        # for f, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):\n        # for f, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train, groups=groups)):\n        for f in range(nfold):\n#             f_ind = df[~df.week.between(30, 32)].index\n#             outf_ind = df[df.week.between(30, 32)].index\n            f_ind = df1[df1.GameId < 2019110000].index\n            outf_ind = df1[df1.GameId >= 2019110000].index\n            print(len(f_ind), len(outf_ind))\n\n            x4_train_f, x4_val_f = x4_train[f_ind].copy(), x4_train[outf_ind].copy()\n            y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]\n            y1_train_f, y1_val_f = y1_train[f_ind], y1_train[outf_ind]\n            y0_train_f, y0_val_f = y0_train[f_ind], y0_train[outf_ind]\n            y01_train_f, y01_val_f = y01_train[f_ind], y01_train[outf_ind]\n    #         sw_f = sw[f_ind] \n\n            # shuffle data\n            idx = np.arange(len(y_train_f))\n            np.random.shuffle(idx)\n        #     X_train_f = X_train_f[idx]\n            y_train_f = y_train_f[idx]\n            y1_train_f = y1_train_f[idx]\n            y0_train_f = y0_train_f[idx]\n            y01_train_f = y01_train_f[idx]\n            x4_train_f = x4_train_f[idx]\n        #     y_train_f = y_train_f.iloc[idx]\n\n            # track oof prediction for cv scores\n            val_preds = 0\n            vi = np.array([np.array([v*22 + i for i in range(22)]) for v in outf_ind]).flatten()\n            di = train.iloc[vi].copy()\n            di = di.reset_index(drop=True)\n\n            # ecdf, to be ensembled with nn prediction, kind of a cox neural net model\n            nt = len(y1_train_f)\n            nv = len(y1_val_f)\n            cf = cumfreq(y1_train_f, numbins=199, defaultreallimits=(-99,100))\n            ecdf = cf.cumcount / nt\n            ecdfs.append(ecdf)\n            ecdfr = ecdf.repeat(nv).reshape(199,nv).transpose()\n            escore = mean_squared_error(y0_val_f, ecdfr)\n\n            print('')\n            print('*'*10)\n            print(f'Fold {f+1}/{nfold}')\n            print('*'*10)\n\n            print('')\n            print(f'escore {escore:.6f}')\n            escores.append(escore)\n\n            for j in range(nrepw):\n\n                print('')\n                print(f'Rep {j+1}/{nrepw}')\n\n                model= build_model(x4_train.shape[1], x4_train.shape[2], x4_train.shape[3],\n                    print_summary=first)\n                if first: first = False\n\n                es = EarlyStopping(monitor='val_loss', \n                   mode='min',\n                   restore_best_weights=True, \n                   verbose=2, \n                   patience=patience)\n                es.set_model(model)\n                \n                lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,\n                                                       patience=10, verbose=2, mode='min',\n                                                       min_delta=0.00001)\n                \n#                 oc = OneCycleLR(0.0005)\n\n                h = model.fit(x4_train_f, y_train_f, epochs=nepoch,\n                          # sample_weight=sw_f,\n                          # default batch_size is 32\n                          batch_size=batch_size,\n                          callbacks=[es, lr],\n                          # large batch sizes tend to perform poorly\n        #                   batch_size=2**(j+10),\n                          validation_data=(x4_val_f, y_val_f),\n                          verbose=2)\n\n                models_nn.append(model)\n#                 models_w.append(0.5 / len(fold_list) / nfold / nrep)\n                models_h.append(h)\n\n                vp = model.predict(x4_val_f)\n                vp = mono(vp)\n                vp = adjusty(vp, di, y0_val_f)\n                vs = mean_squared_error(y0_val_f,vp)\n                print(f'nn crps {vs:.6f}')\n\n#                 # lgb\n#                 # print('')\n#                 tr_data = lgb.Dataset(X_train_f1, label=y1_train_f)\n#                 vl_data = lgb.Dataset(X_val_f1, label=y1_val_f) \n#                 # vary max_depth with rep\n#                 lgb_params['max_depth'] = max_depths[j]\n#                 lgb_params['seed'] = 123 + j\n#                 clf = lgb.train(lgb_params, tr_data, valid_sets=[tr_data, vl_data],\n#                                 num_boost_round=20000, early_stopping_rounds=100,\n#                                 verbose_eval=0)\n#                 models_lgb.append(clf)\n#                 models_lgb_bi.append(clf.best_iteration)\n\n#                 vpl = clf.predict(X_val_f1, num_iteration=clf.best_iteration)\n#                 # lgb cox model, shift ecdf so its median is at lgb point prediction\n#                 tr_yc = y1_train_f - np.median(y1_train_f)\n#                 models_tr_yc.append(tr_yc)\n#                 cl = Parallel(n_jobs=ncores)(delayed(gb_cox)(tr_yc,v) for v in vpl)\n#                 c = np.concatenate(cl).reshape(-1,num_class)\n#                 c = mono(c)\n#                 c = adjusty(c, di, y0_val_f)\n#                 print(f'lgb crps {mean_squared_error(y0_val_f,c):.6f}')\n\n#                 # nonnegative least squares to estimate ensemble weights\n#                 b = y0_val_f.flatten()\n#                 A = np.zeros((len(b),3))\n#                 A[:,0] = vp.flatten()\n#                 A[:,1] = c.flatten()\n#                 A[:,2] = ecdfr.flatten()\n#                 bestw = nnls(A,b)[0]\n#                 besta = np.matmul(A,bestw).reshape(-1,199)\n#                 besta = mono(besta)\n#                 besta = adjusty(besta, di, y0_val_f, reduced=False)\n#                 bscore = bests = mean_squared_error(y0_val_f, besta)\n\n#                 # print('')        \n#                 print(f'bscore {bests:.6f} {bestw}')\n#                 bscores.append(bests)\n#                 bscorew.append(bestw)\n#                 bscorea.append(besta)\n\n#                 val_preds += besta / nrepw\n\n                bscores.append(vs)\n                val_preds += vp / nrepw\n\n                # test_preds[:,f] += model.predict(proc_X_test_f)[:,0] / nrep\n\n\n                if perm:\n                    ff = str(nfold) + '_' + str(f+1)\n                    \n#                     feature_imp = pd.DataFrame(zip(lgb_features, clf.feature_importance(),\n#                                                    clf.feature_importance(importance_type='gain')),\n#                                                    columns=['Feature','Splits'+ff,'Gain'+ff])\n                    \n#                     perm_imp = permutation_importance(X_val_f1,\n#                                                       y1_val_f, clf,\n#                                                       mean_absolute_error)\n#                     perm_imp.columns = ['Feature','Perm'+ff]\n#                     feature_imp = feature_imp.merge(perm_imp, how='left', on='Feature')\n\n                    yardline = train_df['YardLine'][::22].values - 10\n                    perm_imp = permutation_importance_list([list(od.columns)],\n                                                           [x4_val_f],\n                                                           y0_val_f, yardline[outf_ind],\n                                                           model, mean_squared_error)\n\n#                     perm_imp = permutation_importance(X_val_f0,\n#                                                       y_val_f, model,\n#                                                       mean_squared_error)\n\n                    perm_imp.columns = ['Feature','PermNN'+ff]\n                    perm_imp = perm_imp.sort_values(by='PermNN'+ff, ascending=False).reset_index(drop=True)\n                    print()\n                    print(perm_imp.head(n=50))\n#                     print()\n#                     print(perm_imp.tail(n=50))\n    \n                    # feature_imp = feature_imp.merge(perm_imp, how='left', on='Feature')\n\n#                     feature_imp.sort_values(by='Splits'+ff, inplace=True, ascending=False)\n#                     print('')\n#                     print(feature_imp.head(n=10))\n\n#                     feature_imp.sort_values(by='Gain'+ff, inplace=True, ascending=False)\n#                     print('')\n#                     print(feature_imp.head(n=10))\n\n#                     feature_imp.sort_values(by='Perm'+ff, inplace=True, ascending=False)\n#                     print('')\n#                     print(feature_imp.head(n=15))\n\n#                     feature_imp.sort_values(by='PermNN'+ff, inplace=True, ascending=False)\n#                     print('')\n#                     print(feature_imp.head(n=15))\n\n#                     print(feature_imp.shape)\n\n#                     fname = 'imp/' + mname + '_imp' + ff + '.csv'\n#                     perm_imp.to_csv(fname, index=False)\n#                     print(fname, feature_imp.shape)\n\n                gc.collect()\n\n            val_preds = mono(val_preds)\n            val_preds = adjusty(val_preds, di, y0_val_f, reduced=False)\n            full_val_preds[outf_ind] += val_preds\n            vscore = mean_squared_error(y0_val_f, val_preds)\n            print(f'\\nvscore {vscore:.6f}')\n            vscores.append(vscore)\n\n        #     if f == 0: break\n\n        nfh = int(np.ceil(nfold / 2))\n        nfq = int(np.ceil(nfold / 4))\n\n        print('')\n        print(f'\\nAll bscores {np.array(bscores)}')\n        print('Mean bscores: %.6f' % np.mean(bscores))\n        print('Mean vscores: %.6f' % np.mean(vscores))\n    #         print('Mean vscores last half: %.6f' % np.mean(vscores[-nfh:]))\n    #         print('Mean vscores last quar: %.6f' % np.mean(vscores[-nfq:]))\n    #     print('Mean ecdf weights last half: %.6f' % np.mean(bscorew[-nfh*nrep:]))\n    #     print('Mean ecdf weights last quar: %.6f' % np.mean(bscorew[-nfq*nrep:]))\n#         print(f'\\nAll bscores {np.array(bscores)}')\n        # print(f'\\nAll vscores {np.array(vscores)}')\n#         print(f'\\nAll lgb iters {np.array(models_lgb_bi)}')\n#         print(f'\\nAll ecdf weights {bscorew}')")


# In[ ]:




