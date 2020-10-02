#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import gc

from sklearn import preprocessing
from itertools import product
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from datetime import datetime, date, time, timedelta


# In[ ]:


def label_var(data,variables_cat):
    lb=[]
    for m in variables_cat:
        l=LabelEncoder()
        lb.append(l.fit(list(data[m].dropna())))
    
    return lb

def label_enc(data,l,categorical_features):
    i=0
    for m in categorical_features:
        data.loc[data[m].notnull(),m]=l[i].transform(data.loc[data[m].notnull(),m])
        i=i+1


# In[ ]:


def rolling_promedio(unidad,var_numerica,data,periodo):
    ent=unidad+var_numerica
    #data.sort_values(ent,inplace=True)
    data.reset_index(drop=True,inplace=True)
    return data[ent].groupby(unidad).rolling(min_periods=3,window=periodo).mean().reset_index(drop=True)[var_numerica]


# In[ ]:


camp = pd.read_csv('campana_consultora.csv',encoding='latin1') 
cons = pd.read_csv('maestro_consultora.csv',encoding='latin1')
prod = pd.read_csv('maestro_producto.csv',encoding='latin1')
trx  = pd.read_csv('dtt_fvta_cl.csv',encoding='latin1')
sub  = pd.read_csv('predict_submission.csv',encoding='latin1') 


# In[ ]:


camp.shape, cons.shape, prod.shape, trx.shape, sub.shape


# In[ ]:


camp['IdConsultora'].nunique()


# In[ ]:


camp['campana'].nunique()


# In[ ]:


camp['campana'].unique()


# In[ ]:


month=camp["campana"].unique().tolist()
month.extend([201907])


# In[ ]:


x={"campana":month,"IdConsultora":camp["IdConsultora"].unique().tolist()}


# In[ ]:


train=pd.DataFrame(list(product(*x.values())), columns=x.keys())


# In[ ]:


train = train.sort_values(by=['IdConsultora','campana'])


# In[ ]:


train.reset_index(drop=True,inplace=True)


# In[ ]:


train=train.merge(camp,on=["campana","IdConsultora"],how="outer")


# In[ ]:


train.loc[train["Flagpasopedido"].isnull(),"Flagpasopedido"]=0


# In[ ]:


trx['descuento'] = trx['descuento']/100
trx['precio_menos_dcto'] = trx['preciocatalogo']-(trx['preciocatalogo']*trx['descuento'])
trx['precio_por_anulado'] = trx['realanulmnneto']/trx['realuuanuladas']
trx['precio_por_devuelto'] = trx['realdevmnneto']/trx['realuudevueltas']
trx['precio_por_faltante'] = trx['realvtamnfaltneto']/trx['realuufaltantes']
trx['neto_por_unidad'] = trx['realvtamnneto']/trx['realuuvendidas']
trx['rt_neto_vta'] = trx['realvtamnneto']/trx['realvtamncatalogo']
trx['cnt_descuento'] = trx['descuento'].apply(lambda x: 1 if x>0 else 0)


# In[ ]:


aggregations = {
    'codigotipooferta':['nunique'],
    'descuento':['min','max','sum','mean','std'],
    'ahorro':['min','max','sum','mean','std'],
    'canalingresoproducto':['nunique'],
    'idproducto':['nunique'],
    'codigopalancapersonalizacion':['nunique'],
    'preciocatalogo':['min','max','sum','mean','std'],
    'grupooferta':['nunique'],
    'realanulmnneto':['min','max','sum','mean','std'],
    'realdevmnneto':['min','max','sum','mean','std'],
    'realuuanuladas':['min','max','sum','mean','std'],
    'realuudevueltas':['min','max','sum','mean','std'],
    'realuufaltantes':['min','max','sum','mean','std'],
    'realuuvendidas':['min','max','sum','mean','std'],
    'realvtamnfaltneto':['min','max','sum','mean','std'],
    'realvtamnneto':['min','max','sum','mean','std'],
    'realvtamncatalogo':['min','max','sum','mean','std'],
    'realvtamnfaltcatalogo':['min','max','sum','mean','std'],
    'precio_menos_dcto':['min','max','sum','mean','std'],
    'precio_por_anulado':['min','max','sum','mean','std'],
    'precio_por_devuelto':['min','max','sum','mean','std'],
    'precio_por_faltante':['min','max','sum','mean','std'],
    'neto_por_unidad':['min','max','sum','mean','std'],
    'rt_neto_vta':['min','max','sum','mean','std'],
    'cnt_descuento':['sum']
}

trx_group=trx.groupby(['campana','idconsultora']).agg(aggregations)
trx_group.columns = ["_trx_agg_".join(x) for x in trx_group.columns.ravel()]
trx_group.reset_index(inplace=True)


# In[ ]:


train.shape, trx_group.shape


# In[ ]:


train = pd.merge(train,trx_group,how='left',left_on=['campana','IdConsultora'],right_on=['campana','idconsultora'])


# In[ ]:


train.shape


# In[ ]:


gc.collect()


# In[ ]:


c = [x for x in train.columns if x not in ['Unnamed: 0','campana','IdConsultora','Flagpasopedido','idconsultora']]


# In[ ]:


len(c)


# In[ ]:


for x in c:
    train.loc[:,x+"_1_lag"]=train.groupby(['IdConsultora'])[x].shift(1)
    train.loc[:,x+"_2_lag"]=train.groupby(['IdConsultora'])[x].shift(2)
    train.loc[:,x+"_3_lag"]=train.groupby(['IdConsultora'])[x].shift(3)
    train.loc[:,x+"_4_lag"]=train.groupby(['IdConsultora'])[x].shift(4)
    train.loc[:,x+"_5_lag"]=train.groupby(['IdConsultora'])[x].shift(5)
    train.loc[:,x+"_6_lag"]=train.groupby(['IdConsultora'])[x].shift(6)
    train.loc[:,x+"_12_lag"]=train.groupby(['IdConsultora'])[x].shift(12)
    train.loc[:,x+"_18_lag"]=train.groupby(['IdConsultora'])[x].shift(18)

gc.collect()


# In[ ]:


train.loc[:,"Flagpasopedido_1_lag"]=train.groupby(['IdConsultora'])["Flagpasopedido"].shift(1)
train.loc[:,"Flagpasopedido_2_lag"]=train.groupby(['IdConsultora'])["Flagpasopedido"].shift(2)
train.loc[:,"Flagpasopedido_3_lag"]=train.groupby(['IdConsultora'])["Flagpasopedido"].shift(3)
train.loc[:,"Flagpasopedido_4_lag"]=train.groupby(['IdConsultora'])["Flagpasopedido"].shift(4)
train.loc[:,"Flagpasopedido_5_lag"]=train.groupby(['IdConsultora'])["Flagpasopedido"].shift(5)
train.loc[:,"Flagpasopedido_6_lag"]=train.groupby(['IdConsultora'])["Flagpasopedido"].shift(6)
train.loc[:,"Flagpasopedido_12_lag"]=train.groupby(['IdConsultora'])["Flagpasopedido"].shift(12)
train.loc[:,"Flagpasopedido_18_lag"]=train.groupby(['IdConsultora'])["Flagpasopedido"].shift(18)

gc.collect()


# In[ ]:


train.shape


# In[ ]:


train_columns = [x for x in train.columns if ('_1_lag' in x or '_2_lag' in x or '_3_lag' in x or '_4_lag' in x or '_5_lag' in x or '_6_lag' in x or '_12_lag' in x or '_18_lag' in x)]


# In[ ]:


len(train_columns)


# In[ ]:


train.shape, cons.shape


# In[ ]:


train=train.merge(cons[['IdConsultora','campanaultimopedido','estadocivil','flagsupervisor',
                        'flagconsultoradigital','flagcorreovalidad','edad','flagcelularvalidado']],
                  how="left",on='IdConsultora')


# In[ ]:


train.shape


# In[ ]:


del trx, trx_group
gc.collect()


# In[ ]:


train['ind'] = train['campanaultimopedido'] - train['campana']


# In[ ]:


train['flag_camp_ult'] = 0
train.loc[train['ind']>=0,'flag_camp_ult']=1


# In[ ]:


gc.collect()


# In[ ]:


train_columns.extend(['estadocivil','flagsupervisor','flagconsultoradigital','flagcorreovalidad','edad','flagcelularvalidado','flag_camp_ult'])


# In[ ]:


len(train_columns)


# In[ ]:


categorical_features = []

for m in train_columns:
    if(train[m].dtypes=='object'):
        categorical_features.append(m)


# In[ ]:


len(categorical_features)


# In[ ]:


l = label_var(train, categorical_features)


# In[ ]:


label_enc(train,l,categorical_features)


# In[ ]:


for df in [train]:
    for m in categorical_features:
        df[m] = df[m].astype(float)


# In[ ]:


train[['descuento_trx_agg_mean_1_lag_3m','ahorro_trx_agg_mean_1_lag_3m','preciocatalogo_trx_agg_mean_1_lag_3m',
       'realanulmnneto_trx_agg_mean_1_lag_3m','realdevmnneto_trx_agg_mean_1_lag_3m','realuuanuladas_trx_agg_mean_1_lag_3m',
       'realuudevueltas_trx_agg_mean_1_lag_3m','realuufaltantes_trx_agg_mean_1_lag_3m','realuuvendidas_trx_agg_mean_1_lag_3m',
       'realvtamnfaltneto_trx_agg_mean_1_lag_3m','realvtamnneto_trx_agg_mean_1_lag_3m','realvtamncatalogo_trx_agg_mean_1_lag_3m',
       'realvtamnfaltcatalogo_trx_agg_mean_1_lag_3m']]=rolling_promedio(["IdConsultora"],['descuento_trx_agg_mean_1_lag','ahorro_trx_agg_mean_1_lag','preciocatalogo_trx_agg_mean_1_lag',
                                                                                          'realanulmnneto_trx_agg_mean_1_lag','realdevmnneto_trx_agg_mean_1_lag','realuuanuladas_trx_agg_mean_1_lag',
                                                                                          'realuudevueltas_trx_agg_mean_1_lag','realuufaltantes_trx_agg_mean_1_lag','realuuvendidas_trx_agg_mean_1_lag',
                                                                                          'realvtamnfaltneto_trx_agg_mean_1_lag','realvtamnneto_trx_agg_mean_1_lag','realvtamncatalogo_trx_agg_mean_1_lag',
                                                                                          'realvtamnfaltcatalogo_trx_agg_mean_1_lag'],train,3)


# In[ ]:


train_columns.extend(['descuento_trx_agg_mean_1_lag_3m','ahorro_trx_agg_mean_1_lag_3m','preciocatalogo_trx_agg_mean_1_lag_3m',
       'realanulmnneto_trx_agg_mean_1_lag_3m','realdevmnneto_trx_agg_mean_1_lag_3m','realuuanuladas_trx_agg_mean_1_lag_3m',
       'realuudevueltas_trx_agg_mean_1_lag_3m','realuufaltantes_trx_agg_mean_1_lag_3m','realuuvendidas_trx_agg_mean_1_lag_3m',
       'realvtamnfaltneto_trx_agg_mean_1_lag_3m','realvtamnneto_trx_agg_mean_1_lag_3m','realvtamncatalogo_trx_agg_mean_1_lag_3m',
       'realvtamnfaltcatalogo_trx_agg_mean_1_lag_3m'])


# In[ ]:


len(train_columns)


# In[ ]:


categorical_index = [train_columns.index(x) for x in categorical_features]


# In[ ]:


train["Flagpasopedido"] = train["Flagpasopedido"].astype(int)


# In[ ]:


train["Flagpasopedido"].value_counts()


# In[ ]:


len(train_columns)


# In[ ]:


len(train)


# In[ ]:


gc.collect()


# In[ ]:


import time

start_time = time.clock()

train_data = xgb.DMatrix(train.loc[train["campana"]<=201905,train_columns], label=train.loc[train["campana"]<=201905,"Flagpasopedido"])
test_data = xgb.DMatrix(train.loc[train["campana"]==201906,train_columns], label=train.loc[train["campana"]==201906,"Flagpasopedido"])


param = {'booster': 'gbtree',
         'tree_method': 'hist',
         'grow_policy': 'depthwise',
         'objective': 'binary:logistic',
         'eta': 0.01,
         'max_depth': 10,
         'colsample_bytree': 0.7,
         'subsample': 0.85,
         'silent': 1,
         'verbose_eval': True,
         'eval_metric': 'auc',
         'nthread' : 4
        }
    
num_round = 5000
model = xgb.train(param,train_data,evals=[(test_data,'test')],num_boost_round=num_round,early_stopping_rounds=50,verbose_eval=20)

print(time.clock() - start_time, "seconds")


# In[ ]:


t=train.loc[train["campana"]==201907,]


# In[ ]:


t.loc[:,"Flagpasopedido"] = model.predict(xgb.DMatrix(t.loc[:,train_columns]), ntree_limit=model.best_ntree_limit)


# In[ ]:


sub.shape


# In[ ]:


sub.head(3)


# In[ ]:


sub=sub[['idconsultora']].merge(t[['IdConsultora','Flagpasopedido']],how="left",left_on=['idconsultora'],right_on=['IdConsultora'])


# In[ ]:


sub=sub.merge(cons[['IdConsultora','campanaultimopedido']],how="left",on='IdConsultora')


# In[ ]:


sub.head(3)


# In[ ]:


sub.loc[sub['campanaultimopedido']==201907,'Flagpasopedido'] = 1


# In[ ]:


sub[['idconsultora','Flagpasopedido']].to_csv('env7_0.976778.csv',index=False)

