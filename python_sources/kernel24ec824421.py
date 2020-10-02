#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Standard plotly imports
#import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
#import cufflinks as cf
import plotly.figure_factory as ff
# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)

# Preprocessing, modelling and evaluating
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial
import os
import gc
print(os.listdir("../input"))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
df_trans = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
print('train Identity : ' , df_id.shape)
print('train Transaction : ' , df_trans.shape)


# In[ ]:


df_trans['TransactionAmt'] = df_trans['TransactionAmt'].astype(float)
total = len(df_trans)
total_amt = df_trans.groupby(['isFraud'])['TransactionAmt'].sum().sum()


# In[ ]:


def cat_feat_ploting(df, col):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(14,10))
    plt.suptitle(f'{col} Distributions', fontsize=22)

    plt.subplot(221)
    g = sns.countplot(x=col, data=df, order=tmp[col].values)
    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])

    g.set_title(f"{col} Distribution", fontsize=19)
    g.set_xlabel(f"{col} Name", fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    # g.set_ylim(0,500000)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14) 

    plt.subplot(222)
    g1 = sns.countplot(x=col, hue='isFraud', data=df, order=tmp[col].values)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, color='black', order=tmp[col].values, legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title(f"{col} by Target(isFraud)", fontsize=19)
    g1.set_xlabel(f"{col} Name", fontsize=17)
    g1.set_ylabel("Count", fontsize=17)

    plt.subplot(212)
    g3 = sns.boxenplot(x=col, y='TransactionAmt', hue='isFraud', 
                       data=df[df['TransactionAmt'] <= 2000], order=tmp[col].values )
    g3.set_title("Transaction Amount Distribuition by ProductCD and Target", fontsize=20)
    g3.set_xlabel("ProductCD Name", fontsize=17)
    g3.set_ylabel("Transaction Values", fontsize=17)

    plt.subplots_adjust(hspace = 0.4, top = 0.85)

    plt.show()


# In[ ]:


train=pd.merge(df_trans,df_id,how="left",on="TransactionID")
test=pd.merge(df_trans,df_id,how="left",on="TransactionID")


# In[ ]:


for col in ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29']:
    train[col] = train[col].fillna('NaN')
    cat_feat_ploting(train, col)


# In[ ]:


train.loc[train['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'
train.loc[train['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'
train.loc[train['id_30'].str.contains('Mac OS', na=False), 'id_30'] = 'Mac'
train.loc[train['id_30'].str.contains('Android', na=False), 'id_30'] = 'Android'
train['id_30'].fillna("NAN", inplace=True)


# In[ ]:


def ploting_cnt_amt(df, col, lim=2000):
   tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
   tmp = tmp.reset_index()
   tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
   
   plt.figure(figsize=(16,14))    
   plt.suptitle(f'{col} Distributions ', fontsize=24)
   
   plt.subplot(211)
   g = sns.countplot( x=col,  data=df, order=list(tmp[col].values))
   gt = g.twinx()
   gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                      color='black', legend=False, )
   gt.set_ylim(0,tmp['Fraud'].max()*1.1)
   gt.set_ylabel("%Fraud Transactions", fontsize=16)
   g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
   g.set_xlabel(f"{col} Category Names", fontsize=16)
   g.set_ylabel("Count", fontsize=17)
   g.set_xticklabels(g.get_xticklabels(),rotation=45)
   sizes = []
   for p in g.patches:
       height = p.get_height()
       sizes.append(height)
       g.text(p.get_x()+p.get_width()/2.,
               height + 3,
               '{:1.2f}%'.format(height/total*100),
               ha="center",fontsize=12) 
       
   g.set_ylim(0,max(sizes)*1.15)
   
   #########################################################################
   perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum()                / df.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')
   perc_amt = perc_amt.reset_index()
   perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
   amt = df.groupby([col])['TransactionAmt'].sum().reset_index()
   perc_amt = perc_amt.fillna(0)
   plt.subplot(212)
   g1 = sns.barplot(x=col, y='TransactionAmt', 
                      data=amt, 
                      order=list(tmp[col].values))
   g1t = g1.twinx()
   g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, 
                       order=list(tmp[col].values),
                      color='black', legend=False, )
   g1t.set_ylim(0,perc_amt['Fraud'].max()*1.1)
   g1t.set_ylabel("%Fraud Total Amount", fontsize=16)
   g.set_xticklabels(g.get_xticklabels(),rotation=45)
   g1.set_title(f"{col} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)
   g1.set_xlabel(f"{col} Category Names", fontsize=16)
   g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)
   g1.set_xticklabels(g.get_xticklabels(),rotation=45)    
   
   for p in g1.patches:
       height = p.get_height()
       g1.text(p.get_x()+p.get_width()/2.,
               height + 3,
               '{:1.2f}%'.format(height/total_amt*100),
               ha="center",fontsize=12) 
       
   plt.subplots_adjust(hspace=.4, top = 0.9)
   plt.show()
   
ploting_cnt_amt(df_trans, 'addr1')


# In[ ]:


ploting_cnt_amt(train, 'id_30')


# In[ ]:


for f in train.drop('isFraud', axis=1).columns:
    if train[f].dtype=='object' or train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))  


# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt?fbclid=IwAR0KKqf69Y33EeKhEqqgRVUp9FGRcVNie6iD8iQTDWA6XcTCuC1shqup9kE
# 
# in 51
