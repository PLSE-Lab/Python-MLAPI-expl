#!/usr/bin/env python
# coding: utf-8

# In[ ]:


dataDir = '/kaggle/input/datathon-belcorp-prueba/'


# In[ ]:


import numpy as np
import math
import pandas as pd


# In[ ]:


df_prod = pd.read_csv(dataDir + 'maestro_producto.csv')


# In[ ]:


df_consultora = pd.read_csv(dataDir +  'maestro_consultora.csv')


# In[ ]:


df_camp_consultora = pd.read_csv(dataDir +  'campana_consultora.csv')


# In[ ]:


df_ventas = pd.read_csv(dataDir + 'dtt_fvta_cl.csv')


# In[ ]:


df_consultora.drop('Unnamed: 0',axis=1,inplace=True)
df_camp_consultora.drop('Unnamed: 0',axis=1,inplace=True)
df_prod.drop('Unnamed: 0',axis=1,inplace=True)


# ### Construyendo el target

# In[ ]:


df_historico_camp = df_camp_consultora[['campana','IdConsultora','Flagpasopedido']]


# In[ ]:


pivot_historico = pd.pivot_table(df_historico_camp, values='Flagpasopedido', index='campana',columns=['IdConsultora'])


# In[ ]:


dictConsultCamp = pivot_historico.to_dict()


# In[ ]:


def getTarget(idConsultora,campana):

  if (campana == 201906):
    return np.nan
  
  year = str(campana)[:-2]
  cap = str(campana)[-2:]

  if (cap=='18'):
    nextCampaign = int(str(int(year)+1) + '01')
  else:
    nextCampaign = campana +1

  ## If the following campaign exists
  if not math.isnan(dictConsultCamp[idConsultora][nextCampaign]):
    return dictConsultCamp[idConsultora][nextCampaign]
  else:
    return np.nan


# In[ ]:


df_camp_consultora['target'] = df_camp_consultora['IdConsultora'].astype(object).combine(df_camp_consultora['campana'],func=getTarget)


# ## Mergeando la data

# In[ ]:


## Mergeo la info de la consultora
df_info_consult = df_camp_consultora.merge(df_consultora,how='left',on='IdConsultora')


# In[ ]:


## Mergeo la info relacionado a ventas
df_info_vt = df_ventas.merge(df_prod,how='left',on='idproducto')


# In[ ]:


df_info_vt['campanaConsultora'] = df_info_vt['campana'].astype(str) + '_' + df_info_vt['idconsultora'].astype(str)


# In[ ]:


df_info_vt_summary = df_info_vt.groupby('campanaConsultora').agg({
                                             'codigotipooferta': lambda x: x.mode().iat[0],
                                             'descuento': 'sum',
                                             'ahorro': 'sum',
                                             'preciocatalogo': 'sum',
                                             'realanulmnneto': 'sum',
                                             'realdevmnneto': 'sum',
                                             'realuuanuladas': 'sum',
                                             'realuudevueltas': 'sum',
                                             'realuufaltantes': 'sum',
                                             'realuuvendidas': 'sum',
                                             'realvtamnfaltneto': 'sum',
                                             'realvtamnneto': 'sum',
                                             'realvtamncatalogo': 'sum',
                                             'realvtamnfaltcatalogo': 'sum'
                                             })


# In[ ]:


df_info_vt_summary.reset_index(inplace=True)


# In[ ]:


df_info_consult['campanaConsultora']= df_info_consult['campana'].astype(str)  + '_' + df_info_consult['IdConsultora'].astype(str)


# In[ ]:


df_info_vt_summary.head(1)


# In[ ]:


## Mergeo todo
df_total = df_info_consult.merge(df_info_vt_summary,on='campanaConsultora',how='left')


# ## Limpiando data

# In[ ]:


# missingData = df_total.isnull().sum()
# percentageMissing = missingData / df_total.shape[0]


# In[ ]:


# percentageMissing.to_frame().reset_index().to_csv('percentageMissing.csv')


# In[ ]:


# percentageMissing = pd.read_csv('percentageMissing.csv')


# In[ ]:


# percentageMissing.drop(columns='Unnamed: 0',axis=1,inplace=True)
# percentageMissing.rename(columns={'0':'porc'},inplace=True)


# In[ ]:


## To drop (high missing values)
droppable = ['codigocanalorigen','flagcorreovalidad','codigofactura']


# In[ ]:


# percentageMissing.sort_values(by='porc',ascending=False)[:25]


# In[ ]:


df_total[df_total['campana']==201906]['IdConsultora'].unique().shape


# In[ ]:


df_total.drop(labels=droppable,axis=1,inplace=True)


# In[ ]:


# df_total = df_total[~df_total['IdConsultora'].isnull()]


# In[ ]:


cat_vars = [c for c in df_total if not pd.api.types.is_numeric_dtype(df_total[c])]


# In[ ]:


cat_vars


# In[ ]:


# 1. Convertir las columnas a tipo "category", ignorar la variable dependiente
# cat_vars = [c for c in df_total if not pd.api.types.is_numeric_dtype(df_total[c])]

cat_vars = ['evaluacion_nuevas',
 'segmentacion',
 'geografia',
 'estadocivil']

for n,col in df_total.items():
    if n in cat_vars:
      df_total[n] = df_total[n].astype('category')

# df_total.dtypes


# In[ ]:


for n,col in df_total.items():
    if pd.api.types.is_categorical_dtype(col):
        df_total[n] = col.cat.codes+1

df_total.head(3)


# In[ ]:


imputation_columns = [
                      'codigotipooferta',
                      'campanaultimopedido',
                      'flagsupervisor',
                      'campanaingreso',
'preciocatalogo',
'realvtamnfaltcatalogo',
'flagdigital',
'ahorro',
'realdevmnneto',
'cantidadlogueos',
'descuento',
'realanulmnneto',
'estadocivil',
'realuuanuladas',
'campanaprimerpedido',
'realuudevueltas',
'flagcelularvalidado',
'realuufaltantes',
'edad',
'realuuvendidas',
'flagconsultoradigital',
'realvtamnfaltneto',
'realvtamnneto',
'realvtamncatalogo']
for n,col in df_total.items():
    if n in imputation_columns:
      df_total[n].fillna((df_total[n].median()), inplace=True)


# In[ ]:


percentageMissing = df_total.isnull().mean()


# In[ ]:


# percentageMissing.sort_values(ascending=False)


# In[ ]:


df_predict = pd.read_csv(dataDir + 'predict_submission.csv')


# In[ ]:


df_total[df_total['target'].isnull()]['IdConsultora'].value_counts().sort_values(ascending=False)


# In[ ]:


df_predict['idconsultora'].unique().shape


# In[ ]:


df_total['campanaprimerpedido']


# In[ ]:


df_total.head(1)


# In[ ]:


## Hacef funcion para restar mejor
df_total['campanaprimerpedido'] = df_total['campana'] - df_total['campanaprimerpedido']
df_total['campanaingreso'] = df_total['campana'] - df_total['campanaingreso']
df_total['campanaultimopedido'] = df_total['campana'] - df_total['campanaultimopedido']


# In[ ]:


df_to_predict = df_total[df_total['target'].isnull()]


# In[ ]:


df_to_predict.shape


# In[ ]:


df_to_predict = df_to_predict.sort_values('campana', ascending=False).drop_duplicates('IdConsultora').sort_index()


# In[ ]:


df_predict.rename(columns={'idconsultora':'IdConsultora'},inplace=True)


# In[ ]:





# In[ ]:


df_entry_model = df_predict.merge(df_to_predict,how='left',on='IdConsultora')


# In[ ]:


df_entry_model


# In[ ]:





# ### Predicting

# In[ ]:


df = df_total[~df_total['target'].isnull()]


# In[ ]:


dropToTrain = ['campana','IdConsultora','campanaConsultora','Flagpasopedido']


# In[ ]:


df.drop(columns=dropToTrain,inplace=True)


# In[ ]:





# In[ ]:


Y = df['target']
X = df.drop('target',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:


import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)


# In[ ]:


parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lgb.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# In[ ]:


X = df.drop('target',axis=1)


# In[ ]:


ids = df_entry_model['IdConsultora'].values


# In[ ]:


dropToTrain = ['campana','campanaConsultora','Flagpasopedido']


# In[ ]:


df_entry_model.drop(columns=dropToTrain,inplace=True)


# In[ ]:


df_entry_model.drop('IdConsultora', inplace=True, axis=1)


# In[ ]:


df_entry_model.drop('flagpasopedido',inplace=True,axis=1)


# In[ ]:


df_entry_model.drop('target',inplace=True,axis=1)


# In[ ]:


x = df_entry_model


# In[ ]:


df_entry_model.columns


# In[ ]:


X.columns


# In[ ]:


y = model.predict(x)


# In[ ]:


y


# In[ ]:


output = pd.DataFrame({'idconsultora': ids, 'flagpasopedido': y})
output.to_csv("submission.csv", index=False)

