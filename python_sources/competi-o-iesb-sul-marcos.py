#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


train = pd.read_csv('../input/train.csv', thousands=',')
test = pd.read_csv('../input/test.csv', thousands=',')
sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['nota_mat'].value_counts()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.regiao.nunique(), test.regiao.nunique()


# In[ ]:


train.estado.nunique(), test.estado.nunique()


# In[ ]:


train['regiao'].value_counts()


# In[ ]:


test['regiao'].value_counts()


# In[ ]:


pd.value_counts(train['estado'])


# In[ ]:


pd.value_counts(test['estado'])


# In[ ]:


train.isnull().sum().to_frame('Qtd. Missing').T


# In[ ]:


test.isnull().sum().to_frame('Qtd. Missing').T


# In[ ]:


df = train.append(test, sort=True)


# In[ ]:


df.isnull().sum().to_frame('Qtd. Missing').T


# In[ ]:


df['populacao'] = df['populacao'].str.replace(',','').str.replace('.','').apply(lambda x: x.split('(')[0]).astype(int)
df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('#DIV/0!','').str.rstrip('%')
df['comissionados_por_servidor'] = df['comissionados_por_servidor'].convert_objects(convert_numeric=True)/100
df['codigo_mun'] = df['codigo_mun'].astype(object)
df['porte'] = df['porte'].astype('category').cat.codes
df['regiao'] = df['regiao'].astype('category').cat.codes
df['estado'] = df['estado'].astype('category').cat.codes


# In[ ]:


df['servidores'] = df['comissionados'].fillna(df['comissionados'])
df['densidade_dem'] = df.groupby(['estado', 'porte'])['densidade_dem'].transform(lambda x:x.fillna(x.mean()))
df['perc_pop_econ_ativa'] = df.groupby(['estado', 'porte'])['perc_pop_econ_ativa'].transform(lambda x:x.fillna(x.mean()))
df['hab_p_medico'] = df.groupby(['estado', 'porte'])['hab_p_medico'].transform(lambda x:x.fillna(x.mean()))
df['exp_vida'] = df.groupby(['estado', 'porte'])['exp_vida'].transform(lambda x:x.fillna(x.mean()))
df['exp_anos_estudo'] = df.groupby(['estado', 'porte'])['exp_anos_estudo'].transform(lambda x:x.fillna(x.mean()))
df['comissionados_por_servidor'] = df.groupby(['estado', 'porte'])['comissionados_por_servidor'].transform(lambda x:x.fillna(x.mean()))

df['gasto_pc_educacao'] = df.groupby(['estado'])['gasto_pc_educacao'].transform(lambda x:x.fillna(x.mean()))
df['gasto_pc_saude'] = df.groupby(['estado'])['gasto_pc_saude'].transform(lambda x:x.fillna(x.mean()))
df['participacao_transf_receita'] = df.groupby(['estado'])['participacao_transf_receita'].transform(lambda x:x.fillna(x.mean()))


# In[ ]:


df.isnull().sum().to_frame('Qtd. Missing').T


# In[ ]:


train = df[~df['nota_mat'].isnull()]
train = train.fillna(-1)
test = df[df['nota_mat'].isnull()]
test = test.fillna(-1)


# In[ ]:


train.isnull().sum().to_frame('Qtd. Missing').T


# In[ ]:


from sklearn.model_selection import train_test_split
treino, valid = train_test_split(train, random_state=42)
treino.shape, valid.shape, test.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=25, n_jobs=-1, n_estimators=300)


# In[ ]:


removed_cols = ['regiao','estado','municipio','porte','codigo_mun','densidade_dem',
                'comissionados_por_servidor','nota_mat']
cols = []
for c in treino.columns:
    if c not in removed_cols:
        cols.append(c)
cols

feats = [c for c in treino.columns if c not in removed_cols]


# In[ ]:


feats


# In[ ]:


rfc.fit(treino[feats], treino['nota_mat'])


# In[ ]:


preds = rfc.predict(valid[feats])


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(valid['nota_mat'], preds)


# In[ ]:


test['nota_mat'] = rfc.predict(test[feats])


# In[ ]:


test.shape


# In[ ]:


test[['codigo_mun','nota_mat']].to_csv('modelo.csv', index=False)

