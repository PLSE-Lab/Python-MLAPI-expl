import os
import gc
import time
import datetime
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from math import floor

path = "/home/meiker/deep_learning/project/datathon-belcorp-prueba/"
os.chdir(path)

consu= pd.read_csv('maestro_consultora.csv')
#producto = pd.read_csv('maestro_producto.csv')
camp= pd.read_csv('campana_consultora.csv')
#venta= pd.read_csv('dtt_fvta_cl.csv')

sub = pd.read_csv('predict_submission.csv')

consu.drop(['Unnamed: 0'],axis=1,inplace=True)
#producto.drop(['Unnamed: 0'],axis=1,inplace=True)
camp.drop(['Unnamed: 0'],axis=1,inplace=True)

def addCampana(campana, n):
  """Adds an integer (n) to a campana"""
  campana = int(campana)
  n = int(n)
  mon = campana % 100
  yr = floor(campana / 100)
  newmon = ((mon + n - 1) % 18) + 1
  newyr = yr + floor((mon + n - 1) / 18)
  return str(newyr) + str(newmon).zfill(2)

def diffCampana(campana1, campana2):
  """Computes the difference in months between two campanas"""
  campana1 = int(campana1)
  mon1 = campana1 % 100
  yr1 = floor(campana1 / 100)
  campana2 = int(campana2)
  mon2 = campana2 % 100
  yr2 = floor(campana2 / 100)
  return 12 * (yr1 - yr2) + (mon1 - mon2)


consu['campanaprimerpedido'] = consu.campanaprimerpedido.astype('str').str[0:4].replace({'nan':1994}).astype('int')
camp['tip_camp'] = camp.campana.astype('str').str[4:].astype('int')

tmp1 = pd.merge(camp,consu,on = ['IdConsultora'],how='left')
tmp1 = tmp1[tmp1.campanaprimerpedido.notna()].reset_index(drop=True)

tmp1['dif_ped'] = [diffCampana(tmp1['campanaultimopedido'].values[i],tmp1['campanaprimerpedido'].values[i]) for i in range(0,len(tmp1))]
tmp1['ant_1ped'] = [diffCampana(tmp1['campana'].values[i],tmp1['campanaprimerpedido'].values[i]) for i in range(0,len(tmp1))]
tmp1['and_uped_1'] = [diffCampana(tmp1['campanaultimopedido'].values[i],tmp1['campana'].values[i]) for i in range(0,len(tmp1))]
tmp1['and_uped_2'] = [diffCampana(tmp1['campana'].values[i],tmp1['campanaingreso'].values[i]) for i in range(0,len(tmp1))]
tmp1['and_uped_3'] = [diffCampana(tmp1['campanaultimopedido'].values[i],tmp1['campanaingreso'].values[i]) for i in range(0,len(tmp1))]

def win(start,ventana,camp):
    meses = []
    for i in range(ventana):
        meses.append(int(addCampana(start,i)))
        
    mes = int(addCampana(max(meses),1))   
        
    train = camp[camp.campana.isin(meses)]
    tmp =  camp.loc[camp.campana==mes,['campana','IdConsultora','Flagpasopedido']].rename({'Flagpasopedido':'target'}, axis=1)
    tmp = tmp[['IdConsultora','target']].reset_index(drop=True)
    train = train.merge(tmp ,on = ['IdConsultora'],how='left')
    train['target'] = train.target.fillna(0)
    return(train)

df = pd.DataFrame()
for i in [17,16]:
    tt = win(201807,i,tmp1)
    df = df.append(tt)

test =sub[['idconsultora']].rename({'idconsultora':'IdConsultora'}, axis=1).merge(test ,on = ['IdConsultora'],how='left')

train=train.fillna(-1)
test=test.fillna(-1)
df = train.append(test)

def label_encoder(df, categorical_columns=None): 
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df, categorical_columns

df, le_encoded_cols = label_encoder(df, None)
import catboost
from catboost import Pool

param = {
        'learning_rate': 0.2,
        'bagging_temperature': 0.1, 
        'l2_leaf_reg': 30,
        'depth': 12, 
       # 'max_leaves': 48,
        'max_bin':255,
        'iterations' : 1000,
        'task_type':'GPU',
        'loss_function' : "Logloss",
        'objective':'CrossEntropy',
        'eval_metric' : "AUC",
        'bootstrap_type' : 'Bayesian',
        'random_seed':1337,
        'early_stopping_rounds' : 100,
        'use_best_model': True
}

train_df = df[df.target.notnull()]
test_df = df[df.target.isnull()]
features =  [i for i in train_df.columns if i not in ['IdConsultora','target','codigocanalorigen','campanaultimopedido',
                                                      'campanaprimerpedido','campanaingreso','flagconsultoradigital',
                                                     'flagsupervisor','flagdigital','flagpedidoanulado','sum_pedido',
                                                      'ano_ing','ano_ult','campana']]
target = train_df.target

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=201)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    
    trn_data = Pool(train_df.iloc[trn_idx][features].values,label=target.iloc[trn_idx].values)
    val_data = Pool(train_df.iloc[val_idx][features].values,label=target.iloc[val_idx].values)   
    
    clf = catboost.train(trn_data, param, eval_set= val_data, verbose = 300)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features])
    predictions += clf.predict(test_df[features]) / folds.n_splits
    predictions = np.exp(predictions)/(1 + np.exp(predictions))
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.get_feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
print('#'*20)    
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

summit_final = pd.concat( [test.IdConsultora, pd.DataFrame(predictions,columns = ['flagpasopedido'] )], axis = 1)
sub =summit_final.groupby(['IdConsultora']).mean().reset_index().rename({'IdConsultora':'idconsultora'}, axis=1)

sub.to_csv('cat_gpu.csv',index=False)

