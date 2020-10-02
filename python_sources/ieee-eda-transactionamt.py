#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
trainSize = 590540
# Any results you write to the current directory are saved as output.


# In[ ]:


'''
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape, test.shape)

del train_transaction, train_identity, test_transaction, test_identity
gc.collect()

df = pd.concat([train, test], axis = 0, sort = False)
del train, test
gc.collect()
'''


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = pd.read_csv("/kaggle/input/ieee-edatime/df.csv")')


# In[ ]:


df.head()


# In[ ]:


sns.distplot(df['TransactionAmt']);


# In[ ]:


sns.distplot(np.log(df['TransactionAmt']), fit=norm);


# In[ ]:


TrAmount = pd.DataFrame(data={'TrAmount': np.log(df['TransactionAmt']), 'isFraud': df['isFraud']})
TrAmount


# In[ ]:


trainFraudGrouped = TrAmount.groupby('isFraud')


# In[ ]:


trainFraudGrouped.var()


# In[ ]:


plt.bar([0.0, 1.0], trainFraudGrouped.mean()['TrAmount'], color='b', alpha=0.7)
plt.show()
plt.bar([0.0, 1.0], trainFraudGrouped.var()['TrAmount'], color='b', alpha=0.7)


# In[ ]:





# In[ ]:


df.columns


# In[ ]:


train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
train_identity.columns


# In[ ]:


del train_identity
gc.collect()


# In[ ]:


s = (df.dtypes == 'object')
obj_cols = list(s[s].index)


# In[ ]:


obj_cols


# In[ ]:


df.columns[:100]


# In[ ]:


df.columns[100:200]


# In[ ]:


df.columns[4:10]


# In[ ]:


df.columns[15:29]


# In[ ]:


df.addr1.describe()


# In[ ]:


df.addr2.describe()


# In[ ]:


df[df.columns[10:12]].isna().sum()


# In[ ]:


C_features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']


# In[ ]:


df[C_features].isna().sum()


# In[ ]:


df[C_features]


# In[ ]:


df = df.drop(columns=['C13'])


# In[ ]:


C_features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14']


# In[ ]:


from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
si = SimpleImputer(missing_values=np.nan, strategy='mean')

df_C = df[C_features]
df = df.drop(columns=C_features)
df_C = si.fit_transform(df_C)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=13, svd_solver='full')
pca.fit(df_C) 


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


df_C = pca.transform(df_C)
df_C = pd.DataFrame(df_C, columns=C_features)
df_C = df_C.drop(columns='C14')


# In[ ]:


C_features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
df_C = scaler.fit_transform(df_C)
df_C = pd.DataFrame(df_C, columns=C_features)


# In[ ]:


df_C


# In[ ]:


df_C.to_csv("df_C.csv")
del df_C, C_features
gc.collect()


# In[ ]:





# In[ ]:


df.columns[46:338]


# In[ ]:


df[['TransactionDT','D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15']]


# In[ ]:


for i in ['D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15']:
    cor_tr = np.corrcoef(df[:trainSize]['TransactionDT']/(3600*24), df[:trainSize][i].fillna(-1))[0,1]
    cor_te = np.corrcoef(df[trainSize:]['TransactionDT']/(3600*24), df[trainSize:][i].fillna(-1))[0,1]
    df[:trainSize].set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 4))
    df[trainSize:].set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+"  || corr_te= "+str(round(cor_te,3)), figsize=(15, 4))
    plt.show()


# In[ ]:


plt.figure(figsize=(12,10))
cor = df[['D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15']].corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


df[['D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15']].isna().sum()/1097231


# In[ ]:


df = df.drop(columns=['D2', 'D5', 'D11'])


# In[ ]:


gc.collect()


# In[ ]:


D_features = ['D1', 'D3', 'D4', 'D10', 'D15']
df_D = df[D_features + ['TransactionDT']]
df = df.drop(columns=D_features)


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(0, 1097231, 3600):\n    df_D.loc[i:min(i+3600, 1097231), D_features] = si.fit_transform(df_D.loc[i:min(i+3600, 1097231), D_features])\n    '''\n    for d in D_features:\n        if df.loc[[i], d].isna().bool():\n            df.loc[[i], d] = np.mean(df.loc[max(0, i - 3600):min(1097231, i+3600), d])\n    '''")


# In[ ]:


gc.collect()


# In[ ]:


df_D[D_features].isna().sum()


# In[ ]:


from sklearn import linear_model
lreg = linear_model.LinearRegression()
for i in D_features:
    lreg.fit(df_D['TransactionDT'].loc[(df_D[i].T != 0)].values.reshape(-1, 1), df_D[i].loc[(df_D[i].T != 0)].values.reshape(-1, 1))
    y_pred = lreg.predict(df_D['TransactionDT'].values.reshape(-1, 1))
    
    cor_tr = np.corrcoef(df_D[:trainSize]['TransactionDT']/(3600*24), df_D[:trainSize][i].fillna(-1))[0,1]
    cor_te = np.corrcoef(df_D[trainSize:]['TransactionDT']/(3600*24), df_D[trainSize:][i].fillna(-1))[0,1]
   
    
    df_D[:].set_index('TransactionDT')[i].plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 4))
    plt.plot(df_D['TransactionDT'], y_pred, color='blue')
    plt.show()
    
    df_D[i] = df_D[i].values.reshape(-1, 1)/y_pred
    df_D[:].set_index('TransactionDT')[i].plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 4), color='red')
    plt.show()
    
    '''
    plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    '''


# In[ ]:


df_D.to_csv("df_D.csv")
del df_D, D_features
gc.collect()


# In[ ]:





# In[ ]:


V_features = df.columns[24:316]


# In[ ]:


df_V = df[V_features]


# In[ ]:


df = df.drop(columns=V_features)
gc.collect()


# In[ ]:


df_V[V_features].isna().sum()


# In[ ]:


print(np.sum(df_V[V_features].isna().sum()/1097231 > 0.08))
df_V_drop = (df_V[V_features].isna().sum()/1097231 > 0.08)
df_V_drop = df_V_drop[df_V_drop].index
df_V = df_V.drop(columns=df_V_drop)


# In[ ]:


V_features = df_V.columns


# In[ ]:


plt.figure(figsize=(12,10))
cor = df_V[V_features].corr()
sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


df_V_1 = df_V.copy()
df_V_1.fillna(-1, inplace=True)


# In[ ]:


df_V[V_features].isna().sum()


# In[ ]:


df_V_1.to_csv("df_V_orig.csv")
del df_V_1
gc.collect()
df_V.fillna(df_V.mean(), inplace=True)
pca = PCA(n_components=86, svd_solver='full')
pca.fit(df_V) 


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


df_V = pca.transform(df_V)


# In[ ]:


df_V = pd.DataFrame(df_V)


# In[ ]:


df_V


# In[ ]:


df_V.to_csv("df_V.csv")
del df_V


# In[ ]:




