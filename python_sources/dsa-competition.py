#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
pd.set_option('max_columns', 200)
pd.options.display.float_format = '{:.2f}'.format
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras import backend as K
import tensorflow as tf
import pickle

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv(r'/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')


# In[ ]:


df_treino = df.copy()
#df_treino = df_treino.dropna(thresh=df_treino.shape[1]/2, axis=0)

# Variaveis numericas
def prepare(df):
    df = df.select_dtypes(['number'])
    #df = df.replace(np.nan, 0)
    return df

df_treino = prepare(df_treino)

def numProcess(df):
    for c in df.columns[df.dtypes != 'object']:
        df[c] = df[c].fillna(df[c].min())
    return df

df_treino = numProcess(df_treino)

# Remove outlier
def rmOutlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.where(df[col] < (Q1 - 1.5 * IQR),  Q1,
                       np.where(df[col] > (Q1 + 1.5 * IQR), Q3, df[col]))

for col in df_treino.columns[2:]:
    rmOutlier(df_treino, col)


# In[ ]:


X = df_treino.iloc[:, 2:].values
y = df_treino.iloc[:, 1].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0,1))
X = scaler.fit_transform(X)


# In[ ]:


SL = 0.05

indexes = list()
import statsmodels.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    results = sm.OLS(y, x).fit()
    maxVar = max(results.pvalues)  
    print('P>value: ', maxVar.astype(float))    
    if maxVar > SL:
        for i in range(numVars):    
            if results.pvalues[i] == maxVar:
                x = np.delete(x, i, 1)
                indexes.append(i)
            else:
                continue
    return x, maxVar


X_new, p = backwardElimination(X, SL)
while p > SL:
    X_new, p = backwardElimination(X_new, SL)
    print(len(X_new[0]))


# In[ ]:


df_treino = df.copy()
#df_treino = df_treino.dropna(thresh=df_treino.shape[1]/2, axis=0)
dummy = ['v24','v31','v66', 'v110', 'v91']

def freqImputer_obj(df):
    for i in range(df.shape[1]):
        df.iloc[:,i] = np.where(df.iloc[:,i].isna(), df.iloc[:,i].value_counts().idxmax(), df.iloc[:,i])
    return df

df_treino = freqImputer_obj(df_treino)
df_treino = pd.get_dummies(df_treino, columns=dummy, drop_first=True)
df_treino = df_treino.iloc[:,126:]

X_dummies = df_treino.values

X_pvalue = np.concatenate((X_new, X_dummies), axis=1)
X_pvalue.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pvalue, y, test_size = 0.15, random_state = 42)


# In[ ]:


'''
neurons = X_pvalue.shape[1]

def criarRede():
    classifier = Sequential()
    classifier.add(Dense(units=neurons, activation='relu', kernel_initializer='glorot_normal', input_dim=neurons))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=neurons, activation='relu', kernel_initializer='glorot_normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    otimizador = keras.optimizers.Adam(lr=1e-3, decay=1e-4, clipvalue=0.5) 
    classifier.compile(optimizer = otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classifier


model = KerasClassifier(build_fn = criarRede,
                             epochs = 10,
                             batch_size = 200)

from numpy.random import seed
seed(42)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)
'''


# In[ ]:


from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization

def criarRede():
    classifier = Sequential()
    classifier.add(Dense(units=512, activation='relu', kernel_initializer='glorot_normal', input_dim=X_train.shape[1]))
    classifier.add(Dropout(0.5))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dropout(0.25))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=32, activation='relu'))
    classifier.add(Dropout(0.25))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=16, activation='relu'))
    classifier.add(Dropout(0.1))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=1, activation='sigmoid'))
    otimizador = keras.optimizers.Adam(lr=1e-3, decay=1e-4, clipvalue=0.5) 
    classifier.compile(optimizer = otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classifier


from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
# Realiza a parada mais cedo quando percebe overfitting
'''
es = EarlyStopping(monitor='binary_accuracy', 
                   mode='min',
                   restore_best_weights=True, 
                   verbose=1, 
                   patience=20)
'''
# Realiza checkpoint durante o treinamento
'''
mc = ModelCheckpoint('best_model.h5',
                     monitor='binary_accuracy',
                     mode='min',
                     save_best_only=True, 
                     verbose=1, 
                     save_weights_only=True)
'''
# Realize o ajuste na Learning Rate durante o treinamento
rl = ReduceLROnPlateau(monitor='binary_accuracy', 
                       factor=0.1, 
                       patience=10, 
                       verbose=1, 
                       epsilon=1e-4, 
                       mode='min')

model = KerasClassifier(build_fn = criarRede)

from numpy.random import seed
seed(42)

# Realiza o fit do modelo
model.fit(X_train, y_train,
          validation_data=[X_test, y_test],
          callbacks=[rl],
          epochs=200, 
          batch_size=200,
          verbose=1,
          shuffle=True)

# Carrega os melhores pesos
#model.load_weights("best_model.h5")

y_pred = model.predict_proba(X_test)


# In[ ]:


# 0.4720 random_uniform
# 0.4717 glorot_normal
from sklearn.metrics import log_loss
log_loss(y_test, y_pred, eps=1e-15)


# In[ ]:


# 0.4811 full+model2
# 0.4706 filtered+model1
# 0.4743 full+model2
# 0.4718 callbalcs


# In[ ]:


df2 = pd.read_csv(r'/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')
df_test = df2.copy()

# Variaveis numericas
def prepare(df):
    df = df.select_dtypes(['number'])
    #df = df.replace(np.nan, 0)
    return df

df_test = prepare(df_test)

# Impute nan
def numProcess(df):
    for c in df.columns[df.dtypes != 'object']:
        df[c] = df[c].fillna(df[c].min())
    return df

df_test = numProcess(df_test)

# Remove outlier
def rmOutlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.where(df[col] < (Q1 - 1.5 * IQR),  Q1,
                       np.where(df[col] > (Q1 + 1.5 * IQR), Q3, df[col]))

for col in df_test.columns[1:]:
 
# =============================================================

df_test = df_test.iloc[:,1:]
for i in indexes:
    df_test = df_test.drop(df_test.columns[i], axis=1)

X_new = df_test.values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0,1))
X_new = scaler.fit_transform(X_new)

# =============================================================
df_test = df2.copy()
dummy = ['v24','v31','v66', 'v110', 'v91']

def freqImputer_obj(df):
    for i in range(df.shape[1]):
        df.iloc[:,i] = np.where(df.iloc[:,i].isna(), df.iloc[:,i].value_counts().idxmax(), df.iloc[:,i])
    return df

df_test = freqImputer_obj(df_test)

df_test = pd.get_dummies(df_test, columns=dummy, drop_first=True)
df_test = df_test.iloc[:,125:]
X_dummies = df_test.values

X_pvalue_test = np.concatenate((X_new, X_dummies), axis=1)

y_pred_test = model.predict_proba(X_pvalue_test)

# =============================================================
df_test = df2.copy()
result = pd.DataFrame()
result['ID'] = df_test['ID']
result['PredictedProb'] = pd.DataFrame(y_pred_test)[1]
result.to_csv(r'submission.csv', index=None)
result.head()


# In[ ]:




