# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:12:39 2019

@author: alexy
"""


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import gc

from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import time
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import seaborn as sns
#import imblearn
import itertools
import lightgbm as lgb
import datetime
from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
from sklearn.cluster import KMeans

from sklearn.feature_selection import RFECV, RFE

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from bayes_opt import BayesianOptimization


caminho = "D:/alexys/python/dsa/competicao_04/competicao-dsa-machine-learning-jun-2019/"

valor_target = "target"

def read_files(path,file):
  print("lendo arquivo %s"%(file))
  df= pd.read_csv(caminho + file)
  df["first_active_month"] = pd.to_datetime(df["first_active_month"])
  df["elapsed_time"] = (datetime.date(2018,2,1) - df["first_active_month"].dt.date).dt.days
  
  return df

train = read_files(caminho, "dataset_treino.csv")
test = read_files(caminho, "dataset_teste.csv")

comerciantes = pd.read_csv(caminho+"comerciantes.csv")

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



def read_transaction(path,file):
  
  print("lendo arquivo %s"%(file))
  df = pd.read_csv(caminho + file, parse_dates =["purchase_date"])
  df = reduce_mem_usage(df)
  for c in ["authorized_flag", "category_1"]:
    df[c] = df[c].map({"Y":1, "N":0}).fillna(0)
    
  print("substituindo inf por nan")
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  for col in df.columns:
    if df[col].dtypes in numerics:
      df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    else:
      next
    
  print("ajuste no campo installments")
  df["installments"].replace([-1,999], np.nan, inplace = True)
  
  
  print("ajuste de mes no arquivo %s"%(file))
  df["month_diff"] = (datetime.date.today() - df["purchase_date"].dt.date).dt.days
  df["month_diff"] = df["month_diff"]//30
  df["month_diff"] = df["month_diff"] + df["month_lag"]
  df = pd.get_dummies(df, columns = ["category_2", "category_3"])
  df["purchase_month"] = df["purchase_date"].dt.month
  
  print("filling nulls")
  for c in ["installments", "purchase_amount"]:
      imputer = Imputer(strategy = "median")
      df[c] = imputer.fit_transform(df[c].values.reshape(-1,1))
#  print("ajuste outlier")
#  media = df["purchase_amount"].mean()
#  std = df["purchase_amount"].std()
#  iqr_range = np.quantile(df["purchase_amount"],0.75) - np.quantile(df["purchase_amount"],0.25)
#
# 
#  upper = media + 3*iqr_range
#  lower = media-3*iqr_range
#  f = df["purchase_amount"] < lower
#  df.loc[f,"purchase_amount"] = lower
#
#  f = df["purchase_amount"] > upper
#  df.loc[f,"purchase_amount"] = upper

#  df["purchase_amount"] = np.log1p(df["purchase_amount"])

  return df

transacoes_historicas = read_transaction(caminho, "transacoes_historicas.csv")
novas_transacoes = read_transaction(caminho, "novas_transacoes_comerciantes.csv")




def aggregation(df, aggkey):
  
  print("Inicio agregacao")
  df.loc[:, 'purchase_date'] = pd.DatetimeIndex(df['purchase_date']).\
                                      astype(np.int64) * 1e-9
  
  
  agg_func = {
    'category_1': ['sum', 'mean'],
    'category_2_1.0': ['mean'],
    'category_2_2.0': ['mean'],
    'category_2_3.0': ['mean'],
    'category_2_4.0': ['mean'],
    'category_2_5.0': ['mean'],
    'category_3_A': ['mean'],
    'category_3_B': ['mean'],
    'category_3_C': ['mean'],
    'merchant_id': ['nunique'],
    'merchant_category_id': ['nunique'],
    'state_id': ['nunique'],
    'city_id': ['nunique'],
    'subsector_id': ['nunique'],
    'purchase_amount': ['sum', 'mean', 'max', 'min', 'std', "median"],
    'installments': ['sum', 'mean', 'max', 'min', 'std', "median"],
    'purchase_month': ['sum', 'mean', 'max', 'min', 'std', "median"],
    'purchase_date': [np.ptp, 'min', 'max'],
    'month_lag': ['sum', 'mean', 'max', 'min', 'std', "median"],
    'month_diff': ['mean']
    }
  
  
  print("rodando groupby")
  df_agg = df.groupby(aggkey).agg(agg_func)
  df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
  df_agg.reset_index(inplace = True)
  
  
  df_temp = df.groupby(aggkey).size().reset_index(name = "transaction_count" )
  
  df_agg = df_agg.merge(df_temp, on = aggkey, how = "left" )
  
  
  df_agg["authorized" ] = df_agg["authorized_flag" ].map({1:"auth", 0:"hist"})
  
  print("gerando pivot")
  df_agg = pd.pivot_table(df_agg, index = "card_id", columns = "authorized", fill_value = 0)  
  df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
  df_agg.reset_index(inplace = True)
  
  
  
  return df_agg
  
key = ["card_id", "authorized_flag"]

transacoes_historicas_2 = aggregation(transacoes_historicas, key)
novas_transacoes_2 = aggregation(novas_transacoes, key)
novas_transacoes_2.columns = ["new_" + c if c != "card_id" else c for c in novas_transacoes_2.columns.values]


def agg_per_month(df):
  
  print("agregação por mês")
  agg_func = {"purchase_amount":['sum', 'mean', 'max', 'min', 'std', "median", "count"],\
              "installments":['sum', 'mean', 'max', 'min', 'std', "median", "count"]}
  temp = df.groupby(["card_id", "month_lag"]).agg(agg_func)
  temp.columns =  ["_".join(c).strip() for c in temp.columns.values]
  temp.reset_index(inplace = True)
  
  temp = temp.groupby("card_id").agg(["mean", "std"])
  temp.columns =  ["_".join(c).strip() for c in temp.columns.values]
  temp.reset_index(inplace = True)
  
  return temp

f = transacoes_historicas["authorized_flag"] == 1
month_info = agg_per_month(transacoes_historicas.copy().loc[f,:])
del f
gc.collect()


campos_valores = ["purchase_amount","installments"]
campos_adicionais = ["category_1", "installments", "city_id", "state_id", "subsector_id"]
#campo_adicional = "category_1"
lista_df_info_adicional = []

for campo_adicional in campos_adicionais:
  
  print("agregação campo adicional %s"%(campo_adicional))
  adicionais = novas_transacoes.groupby(["card_id", campo_adicional])[campos_valores].mean()
  adicionais2  = adicionais.groupby("card_id")[campos_valores].agg(['mean', 'max', 'min', 'std', "median"])
  adicionais2.columns = [campo_adicional+ "_" + "_".join(c).strip() for c in adicionais2.columns.values]  
  adicionais2.reset_index(inplace = True)
  lista_df_info_adicional.append(adicionais2)
  del adicionais, adicionais2
  gc.collect()

for i,df in enumerate(lista_df_info_adicional):
  if i == 0:
    df_info_adicionais = df
  else:
    df_info_adicionais = df_info_adicionais.merge(df, how = "left", on = "card_id")

del lista_df_info_adicional
gc.collect()

dftemp = pd.concat([transacoes_historicas, novas_transacoes])

key = ["card_id", "merchant_id", "month_lag", "authorized_flag"]
print("consolidando base por cartao, merchant e mes")
dftemp = dftemp.groupby(key)["purchase_amount"].sum().reset_index()

def calcula_razao(dftemp):
  print("pivotando a tabela por cartao, merchant e mes")
  list_month_lag = np.sort(np.unique(dftemp["month_lag"]))
  dftemp["month_lag"] = dftemp["month_lag"].map(lambda x: str(x))

  dftemp_pivot = pd.pivot_table(dftemp, index = ["card_id", "merchant_id"], columns = ["month_lag"],\
                                values = ["purchase_amount" ], aggfunc = np.sum, fill_value = 0)
  dftemp_pivot.columns = ["_".join(c).strip() for c in dftemp_pivot.columns.values]
  
  def calc_ratio(colm0,colm1):
    ratio = colm0/colm1
    
    if np.isnan(ratio) | np.isinf(ratio):
      if colm0 == 0:
        ratio = 0
      else:
        ratio = 1
    
    return ratio
  
  for count,i in enumerate(list_month_lag):
    month_str = str(i)
    
    if count == 0:
      next
    else:
      month_str_m1 = str(list_month_lag[count-1])
      concat = month_str + "_" + month_str_m1
      campo = "ratio_" + concat
      colm0 = "purchase_amount_" + month_str
      colm1 = "purchase_amount_" + month_str_m1
  #    ratio = map(lambda x,y: calc_ratio(x,y), dftemp_pivot[colm0].values, dftemp_pivot[colm1].values)
  #    dftemp_pivot[campo] = list(ratio)
      dftemp_pivot[campo] = (dftemp_pivot[colm0]/dftemp_pivot[colm1]).replace([np.inf,-np.inf],np.nan)
      f = dftemp_pivot[colm0] = 0
      dftemp_pivot.loc[f,campo] = 0
      dftemp_pivot[campo] = dftemp_pivot[campo].fillna(1)
      
  campos_agg = [c for c in dftemp_pivot.columns.values if "ratio" in c]

  dftemp_pivot_agg = dftemp_pivot.groupby("card_id")[campos_agg].mean().reset_index()
  dftemp_pivot_agg[campos_agg] = dftemp_pivot_agg[campos_agg].fillna(0)

  return dftemp_pivot_agg


#df_temp_pivot_auth = calcula_razao(dftemp.loc[dftemp["authorized_flag"]==1,:])
#
#df_temp_pivot_auth.columns = [c + "_auth"  if c not in "card_id" else c for c in df_temp_pivot_auth.columns.values]
#
#df_temp_pivot_hist = calcula_razao(dftemp.loc[dftemp["authorized_flag"]!=1,:])
#df_temp_pivot_hist.columns = [c + "_hist" if c not in "card_id" else c for c in df_temp_pivot_hist.columns.values ]


df_temp_pivot_agg = calcula_razao(dftemp)
#
#df_temp_pivot_auth = df_temp_pivot_auth.rename(columns = {"card_id_auth":"card_id"})
#df_temp_pivot_hist = df_temp_pivot_auth.rename(columns = {"card_id_hist":"card_id"})
del dftemp
gc.collect()

def merge_infos(dfin):
  dfout = dfin.merge(transacoes_historicas_2, how = "left", on = "card_id").fillna(0)
  dfout = dfout.merge(novas_transacoes_2, how = "left", on = "card_id").fillna(0)
  dfout = dfout.merge(df_info_adicionais, how = "left" , on = "card_id").fillna(0)
  dfout = dfout.merge(month_info, how = "left" , on = "card_id").fillna(0)
  dfout = dfout.merge(df_temp_pivot_agg, how = "left" , on = "card_id").fillna(0)
  dfout = pd.get_dummies(dfout, columns = ["feature_1", "feature_2","feature_3"])
#  dfout = dfout.merge(df_temp_pivot_auth, how = "left" , on = "card_id").fillna(0)
#  dfout = dfout.merge(df_temp_pivot_hist, how = "left" , on = "card_id").fillna(0)
  
  return dfout

train_merge = merge_infos(train)
test_merge = merge_infos(test)

del transacoes_historicas, transacoes_historicas_2, \
novas_transacoes, novas_transacoes_2
gc.collect()


unimportant_features = [
    'category_2_1.0_mean_auth',
    'category_2_2.0_mean_auth',
    'category_2_3.0_mean_auth',
    'category_2_5.0_mean_auth',
    'category_2_3.0_mean_hist',
    'category_2_4.0_mean_hist',
    'category_2_5.0_mean_hist',
    'category_3_A_mean_hist',
    'installments_min_hist',
    'installments_std_hist',
    'month_lag_std_hist',
    'purchase_amount_max_hist',
    'purchase_month_max_hist',
    'purchase_month_min_hist',
    'purchase_month_std_hist',
    'installments_min_mean',
    'new_category_2_1.0_mean_auth',
    'new_category_2_2.0_mean_auth',
    'new_category_2_3.0_mean_auth',
    'new_category_2_5.0_mean_auth',
    'new_city_id_nunique_auth',
    'new_installments_std_auth',
    'new_state_id_nunique_auth',
    'purchase_amount_mean_mean',
    'purchase_amount_median_mean',
]


features = train_merge.columns.values
features = [f for f in features if f not in [valor_target,"card_id", "first_active_month"] + unimportant_features ]

categorical_feats = ['feature_2', 'feature_3']

kfold = model_selection.KFold(n_splits = 5, random_state = 7)
num_boost_round = 10000
label_out = "outxgb_150"

def get_feature_importances(data, features, valor_target, shuffle = False, seed=None):
    
    # Shuffle target if required
    y = data[valor_target].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data[valor_target].copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'num_leaves': 129,
        'min_data_in_leaf': 148, 
        'objective':'regression',
        'max_depth': 9,
        'learning_rate': 0.005,
        "min_child_samples": 24,
        "boosting": "gbdt",
        "feature_fraction": 0.7202,
        "bagging_freq": 1,
        "bagging_fraction": 0.8125 ,
        "bagging_seed": 11,
        "metric": 'rmse',
        "lambda_l1": 0.3468,
        "random_state": 133,
        "verbosity": -1
    }
    
    # Fit the model
    clf = lgb.train(params=lgb_params,
                    train_set=dtrain,
                    num_boost_round=1000)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(clf.predict(data[features]), y)**0.5
    
    return imp_df



# Seed the unexpected randomness of this world
np.random.seed(123)
# Get the actual importance, i.e. without shuffling
actual_imp_df = get_feature_importances(train_merge, features, valor_target ,shuffle=False)


null_imp_df = pd.DataFrame()
nruns = 100
init = time.time()
for i in range(0, nruns):
    imp_df = get_feature_importances(train_merge, features, valor_target ,shuffle=True)
    imp_df["run"] = i + 1
    null_imp_df = pd.concat([null_imp_df,imp_df])
    
    end = time.time()
    dt = (end-init)/60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nruns, dt)
    print(dsp)
    
feature_scores = [] 
for _f in np.unique(actual_imp_df["feature"]):
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    

sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False))
plt.show()

sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False))
plt.tight_layout()


#f = (scores_df["split_score"] > 0) | (scores_df["gain_score"] > 0.1)
f = (scores_df["gain_score"] >= 0.5)

features = scores_df.loc[f,"feature"].to_list()


lgb_dataset = lgb.Dataset(train_merge[features].values, \
                          train_merge[valor_target].values)




inicio = time.time()
print("Inicio CV - LIGHTGBM" )


def hyperopt_obj_lgb(num_leaves, min_data_in_leaf, max_depth, feature_fraction,\
                     bagging_fraction,lambda_l1, colsample_bytree,\
                     reg_alpha, reg_lambda):
  
  params = {
          'num_leaves': int(num_leaves),
          'min_data_in_leaf': int(min_data_in_leaf), 
          'objective':'regression',
          'max_depth': int(max_depth),
          'learning_rate': 0.0075,
          "boosting": "gbdt",
          "feature_fraction": feature_fraction,
          "bagging_freq": 1,
          "bagging_fraction": bagging_fraction ,
          "bagging_seed": 11,
          "metric": 'rmse',
          "lambda_l1": lambda_l1,
          "verbosity": -1,
          "colsample_bytree":colsample_bytree,
          "reg_alpha":reg_alpha,
          "reg_lambda":reg_lambda
      }
  
  cv_lgb = lgb.cv( params,lgb_dataset, \
                  num_boost_round, \
                  folds = kfold,\
                  verbose_eval  = False, \
                  early_stopping_rounds  = 100,
                  metrics= "rmse")
  
  best_score = min(cv_lgb["rmse-mean"])
  loss = best_score
  print(loss)
  return -loss


LGB_BO = BayesianOptimization(hyperopt_obj_lgb, {
    'max_depth': (4, 15),
    'num_leaves': (70, 130),
    'min_data_in_leaf': (10, 200),
    'feature_fraction': (0.7, 1.0),
    'bagging_fraction': (0.7, 1.0),
    'lambda_l1': (0, 6),
    "colsample_bytree": (0.6,1),
    "reg_alpha":(0.7,1),
    "reg_lambda":(0.7,1)    
    })
  
LGB_BO.maximize(init_points=2, n_iter=20, acq='ei', xi=0.0)

best_lgb_parameters = LGB_BO.max
best_lgb_parameters= best_lgb_parameters["params"]
best_lgb_parameters["metric" ] = "rmse"
for k,v in best_lgb_parameters.items():
  if k == "boosting":
    
#    if int(v) == 0:
#      v = "gbdt"
#    elif int(v) == 1:
#      v = "goss"
#    else:
#      v = "dart"
    best_lgb_parameters[k] = v
   
  elif k in ("max_depth", "min_data_in_leaf", "num_leaves"):
    best_lgb_parameters[k] = int(v)

best_lgb_parameters["learning_rate"] = 0.001
fim = time.time()

print("Fim CV - LIGHTGBM" )
dt = fim - inicio
dt = dt/60
print("tempo execução: %.2f"%(dt))

#
#for i in range(0,10):
#  xtrain, xvalid, ytrain,  yvalid = \
#  model_selection.train_test_split(train_merge[features].values, train_merge[valor_target].values, train_size = 0.75, random_state = i)
#  
#  dtrain = lgb.Dataset(xtrain, ytrain)
#  dvalid = lgb.Dataset(xvalid, yvalid)
#  lgbmodel = lgb.train( best_lgb_parameters,dtrain, num_boost_round, dvalid, verbose_eval  = False, early_stopping_rounds  = 100)  
#  prediction = lgbmodel.predict(test_merge[features].values)
#  list_pred.append(prediction)

#test_merge["lgb"] =prediction

#test_merge[label_out] =np.mean(list_pred,axis=0)


#test_merge.loc[:,["card_id", label_out]]\
#.rename(columns = {label_out:"target"}).to_csv(caminho + "xout75" + ".csv", index = False)

list_pred_kfold= []
param = {'num_leaves': 128,
         'min_data_in_leaf': 149, 
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2634,
         "random_state": 133,
          'reg_alpha': 0.9645029980933089,
          'reg_lambda': 0.9475985318211818,
         "verbosity": 2}

for train_idx, test_idx in kfold.split(train_merge[features].values,train_merge[valor_target].values):
  
#  print("Calculando Fold %i" %(i))
  xtrain = train_merge.loc[train_idx, features]
  ytrain = train_merge.loc[train_idx, valor_target]
  xvalid= train_merge.loc[test_idx, features]
  yvalid = train_merge.loc[test_idx, valor_target]
  dtrain = lgb.Dataset(xtrain, ytrain)
  dvalid = lgb.Dataset(xvalid, yvalid)

  lgbmodel = lgb.train( best_lgb_parameters,dtrain, num_boost_round, dvalid, verbose_eval  = False, early_stopping_rounds  = 200)  
  prediction = lgbmodel.predict(test_merge[features].values)
  
  list_pred_kfold.append(prediction)
  error = mean_squared_error(train_merge[valor_target], lgbmodel.predict(train_merge[features].values))
  error = error **0.5
  print("erro = %f"%(error))
  


test_merge["lgb_kfold"] =np.median(list_pred_kfold,axis=0)

#test_merge.loc[:,["card_id", label_out]]\
#.rename(columns = {label_out:"target"}).to_csv(caminho + label_out + ".csv", index = False)
#  
#

test_merge.loc[:,["card_id", "lgb_kfold"]]\
.rename(columns = {"lgb_kfold":"target"}).to_csv(caminho + "lgb_kfold9" + ".csv", index = False)
  