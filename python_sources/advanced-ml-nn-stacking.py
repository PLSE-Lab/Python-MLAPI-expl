#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports-and-useful-functions" data-toc-modified-id="Imports-and-useful-functions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports and useful functions</a></span></li><li><span><a href="#Categorical-columns-features-engineering" data-toc-modified-id="Categorical-columns-features-engineering-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Categorical columns features engineering</a></span><ul class="toc-item"><li><span><a href="#Street" data-toc-modified-id="Street-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Street</a></span></li><li><span><a href="#Utilities" data-toc-modified-id="Utilities-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Utilities</a></span></li><li><span><a href="#CentralAir" data-toc-modified-id="CentralAir-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>CentralAir</a></span></li><li><span><a href="#Alley" data-toc-modified-id="Alley-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Alley</a></span></li><li><span><a href="#LandSlope" data-toc-modified-id="LandSlope-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>LandSlope</a></span></li><li><span><a href="#PavedDrive" data-toc-modified-id="PavedDrive-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>PavedDrive</a></span></li><li><span><a href="#LotShape" data-toc-modified-id="LotShape-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>LotShape</a></span></li><li><span><a href="#LandContour" data-toc-modified-id="LandContour-2.8"><span class="toc-item-num">2.8&nbsp;&nbsp;</span>LandContour</a></span></li><li><span><a href="#ExterQual" data-toc-modified-id="ExterQual-2.9"><span class="toc-item-num">2.9&nbsp;&nbsp;</span>ExterQual</a></span></li><li><span><a href="#ExterCond" data-toc-modified-id="ExterCond-2.10"><span class="toc-item-num">2.10&nbsp;&nbsp;</span>ExterCond</a></span></li><li><span><a href="#KitchenQual" data-toc-modified-id="KitchenQual-2.11"><span class="toc-item-num">2.11&nbsp;&nbsp;</span>KitchenQual</a></span></li><li><span><a href="#GarageFinish" data-toc-modified-id="GarageFinish-2.12"><span class="toc-item-num">2.12&nbsp;&nbsp;</span>GarageFinish</a></span></li><li><span><a href="#GarageCond" data-toc-modified-id="GarageCond-2.13"><span class="toc-item-num">2.13&nbsp;&nbsp;</span>GarageCond</a></span></li><li><span><a href="#GarageQual" data-toc-modified-id="GarageQual-2.14"><span class="toc-item-num">2.14&nbsp;&nbsp;</span>GarageQual</a></span></li><li><span><a href="#PoolQC" data-toc-modified-id="PoolQC-2.15"><span class="toc-item-num">2.15&nbsp;&nbsp;</span>PoolQC</a></span></li><li><span><a href="#BsmtQual" data-toc-modified-id="BsmtQual-2.16"><span class="toc-item-num">2.16&nbsp;&nbsp;</span>BsmtQual</a></span></li><li><span><a href="#BsmtCond" data-toc-modified-id="BsmtCond-2.17"><span class="toc-item-num">2.17&nbsp;&nbsp;</span>BsmtCond</a></span></li><li><span><a href="#BsmtExposure" data-toc-modified-id="BsmtExposure-2.18"><span class="toc-item-num">2.18&nbsp;&nbsp;</span>BsmtExposure</a></span></li><li><span><a href="#HeatingQC" data-toc-modified-id="HeatingQC-2.19"><span class="toc-item-num">2.19&nbsp;&nbsp;</span>HeatingQC</a></span></li><li><span><a href="#FireplaceQu" data-toc-modified-id="FireplaceQu-2.20"><span class="toc-item-num">2.20&nbsp;&nbsp;</span>FireplaceQu</a></span></li><li><span><a href="#SaleCondition" data-toc-modified-id="SaleCondition-2.21"><span class="toc-item-num">2.21&nbsp;&nbsp;</span>SaleCondition</a></span></li><li><span><a href="#BsmtFinType1-and-BsmtFinType2" data-toc-modified-id="BsmtFinType1-and-BsmtFinType2-2.22"><span class="toc-item-num">2.22&nbsp;&nbsp;</span>BsmtFinType1 and BsmtFinType2</a></span></li><li><span><a href="#Functional" data-toc-modified-id="Functional-2.23"><span class="toc-item-num">2.23&nbsp;&nbsp;</span>Functional</a></span></li></ul></li><li><span><a href="#Exploring-the-correlation-between-features" data-toc-modified-id="Exploring-the-correlation-between-features-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Exploring the correlation between features</a></span></li><li><span><a href="#Baseline-model" data-toc-modified-id="Baseline-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Baseline model</a></span><ul class="toc-item"><li><span><a href="#Submitting-to-kaggle" data-toc-modified-id="Submitting-to-kaggle-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Submitting to kaggle</a></span></li></ul></li></ul></div>

# ## Imports and useful functions

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings('ignore')


# In[ ]:


def preprocess_street(x):
    return x.strip()=='Pave'

def preprocess_centralair(x):
    return x.strip()=='Y'

def preprocess_alley(x):
    if pd.isna(x):
        return 0
    if x.strip()=='Grvl':
        return -1
    
    return 1

landslope_map = {'Gtl': 0, 'Mod':1, 'Sev': 2}

paveddrive_map = {'N': 0, 'P': 1, 'Y': 2}

lotshape_map = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}

landcontour_map = {'Low': 0, 'HLS': 1, 'Bnk': 2, 'Lvl': 3}

general_map = {np.nan: -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

garagefinish_map = {np.nan: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

bsmtexposure_map = {np.nan: 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

bsmtfintype_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, np.nan: 0}

functional_map = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}

def preprocess_test(test):
    df = test.copy()
    df['Street'] = df['Street'].apply(preprocess_street)
    df.drop('Utilities', axis=1, inplace=True)
    df['CentralAir'] = df['CentralAir'].apply(preprocess_centralair)
    df['Alley'] = df['Alley'].apply(preprocess_alley)
    df['LandSlope'] = df['LandSlope'].map(landslope_map)
    df['PavedDrive'] = df['PavedDrive'].map(paveddrive_map)
    df['LotShape'] = df['LotShape'].map(lotshape_map)
    df['FireplaceQu'] = df['FireplaceQu'].map(general_map)
    df['HeatingQC'] = df['HeatingQC'].map(general_map)
    df['BsmtCond'] = df['BsmtCond'].map(general_map)
    df['BsmtQual'] = df['BsmtQual'].map(general_map)
    df['BsmtExposure'] = df['BsmtExposure'].map(bsmtexposure_map)
    df['PoolQC'] = df['PoolQC'].map(general_map)
    df['GarageQual'] = df['GarageQual'].map(general_map)
    df['GarageCond'] = df['GarageCond'].map(general_map)
    df['GarageFinish'] = df['GarageFinish'].map(garagefinish_map)
    df['KitchenQual'] = df['KitchenQual'].map(general_map)
    df['ExterCond'] = df['ExterCond'].map(general_map)
    df['ExterQual'] = df['ExterQual'].map(general_map)
    df['LandContour'] = df['LandContour'].map(landcontour_map)
    df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmtfintype_map)
    df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmtfintype_map)
    df['Functional'] = df['Functional'].map(functional_map)
    df = pd.concat([df.drop('SaleCondition', axis=1), pd.get_dummies(df['SaleCondition'], prefix='SaleCondition')], axis=1)
    numerical_cols = []
    for col in df.columns:
        if col!='Id' and df[col].dtype!='object':
            numerical_cols.append(col)
            
    return np.array(df[numerical_cols].fillna(-999))


def evaluate_model(model, X_train, y_train, rkf, y_scaler):
    rmsles = []
    models = []
    for train_idx, val_idx in rkf.split(X_train):
        model.fit(X_train[train_idx], y_train[train_idx])
        y_pred = model.predict(X_train[val_idx])
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1)).clip(0, np.inf)
        y_true = y_scaler.inverse_transform(y_train[val_idx].reshape(-1,1))
        rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
        rmsles.append(rmsle)
        models.append(model)
        
    print("RMSLE: {:.5f} +- {:.5f}".format(np.mean(rmsles), np.std(rmsles)))
    return models


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# ## Categorical columns features engineering

# Since we have a lot of features, we are going to focus primarly on the categorical ones.
# 
# Let's first find those features and how many unique values they have.

# In[ ]:


categocial_features = []
for col in train.columns:
    if train[col].dtype=='object':
        categocial_features.append((col, len(train[col].unique())))
        
sorted(categocial_features, key=lambda k: k[1])


# ### Street

# In[ ]:


train['Street'].value_counts()


# In[ ]:


train['Street'].isna().sum()


# The street feature can be easily encoded as is pave? 1 else 0.

# In[ ]:


train['Street'] = train['Street'].apply(preprocess_street)


# ### Utilities

# In[ ]:


train['Utilities'].value_counts()


# In[ ]:


train['Utilities'].isna().sum()


# This feature won't help our model at all since we have practically just one value, it is better to just drop it. 

# In[ ]:


train.drop('Utilities', axis=1, inplace=True)


# ### CentralAir

# In[ ]:


train['CentralAir'].value_counts()


# In[ ]:


train['CentralAir'].isna().sum()


# This feature can also be easily encoded as 1 or 0.

# In[ ]:


train['CentralAir'] = train['CentralAir'].apply(preprocess_centralair)


# ### Alley

# In[ ]:


train['Alley'].value_counts()


# In[ ]:


train['Alley'].isna().sum()


# In[ ]:


sns.boxplot(x=train['Alley'].fillna('NaN'), y=train['SalePrice'])


# We can see that a house having an alley does not affect the price to much, but given that it has an alley if it is pave than the price is higher, if it is Gtvl than the price tends to be lower.

# In[ ]:


train['Alley'] = train['Alley'].apply(preprocess_alley)


# ### LandSlope

# In[ ]:


train['LandSlope'].value_counts()


# This feature can be considered ordinal since we know the relation: 
# $$
# Gtl<Mod<Sev
# $$
# 
# Let's the preprocess this columns as Gtl=0, Mod=1, Sev=2.

# In[ ]:


train['LandSlope'] = train['LandSlope'].map(landslope_map)


# ### PavedDrive

# In[ ]:


train['PavedDrive'].value_counts()


# Just like LandSlope, we have a ordinal relation in this column. We know from data description that:
# 
# $$
# N<P<Y
# $$
# 
# using this we can map this feature.

# In[ ]:


train['PavedDrive'] = train['PavedDrive'].map(paveddrive_map)


# ### LotShape

# In[ ]:


train['LotShape'].value_counts()


# Also an ordinal feature with the relation:
# 
# $$
# IR3<IR2<IR1<Reg
# $$

# In[ ]:


train['LotShape'] = train['LotShape'].map(lotshape_map)


# ### LandContour

# In[ ]:


train['LandContour'].value_counts()


# Also an ordinal feature with the relation:
# 
# $$
# Low<HLS<Bnk<Lvl
# $$

# In[ ]:


train['LandContour'] = train['LandContour'].map(landcontour_map)


# ### ExterQual

# In[ ]:


train['ExterQual'].value_counts()


# From data description, it is also ordinal with:
# 
# $$
# Po<Fa<TA<Gd<Ex
# $$

# In[ ]:


train['ExterQual'] = train['ExterQual'].map(general_map)


# ### ExterCond

# In[ ]:


train['ExterCond'].value_counts()


# In[ ]:


train['ExterCond'] = train['ExterCond'].map(general_map)


# ### KitchenQual

# From data description, we can use the same map as ExterQual.

# In[ ]:


train['KitchenQual'] = train['KitchenQual'].map(general_map)


# ### GarageFinish

# In[ ]:


train['GarageFinish'].value_counts()


# In[ ]:


train['GarageFinish'].isna().sum()


# In[ ]:


train['GarageFinish'] = train['GarageFinish'].map(garagefinish_map)


# ### GarageCond

# In[ ]:


train['GarageCond'].value_counts()


# In[ ]:


train['GarageCond'] = train['GarageCond'].map(general_map)


# ### GarageQual

# In[ ]:


train['GarageQual'].value_counts()


# In[ ]:


train['GarageQual'] = train['GarageQual'].map(general_map)


# ### PoolQC

# In[ ]:


train['PoolQC'].value_counts()


# In[ ]:


train['PoolQC'] = train['PoolQC'].map(general_map)


# ### BsmtQual

# In[ ]:


train['BsmtQual'].value_counts()


# In[ ]:


train['BsmtQual'] = train['BsmtQual'].map(general_map)


# ### BsmtCond

# In[ ]:


train['BsmtCond'].value_counts()


# In[ ]:


train['BsmtCond'] = train['BsmtCond'].map(general_map)


# ### BsmtExposure

# In[ ]:


train['BsmtExposure'].value_counts()


# In[ ]:


train['BsmtExposure'] = train['BsmtExposure'].map(bsmtexposure_map)


# ### HeatingQC

# In[ ]:


train['HeatingQC'].value_counts()


# In[ ]:


train['HeatingQC'] = train['HeatingQC'].map(general_map)


# ### FireplaceQu

# In[ ]:


train['FireplaceQu'].value_counts()


# In[ ]:


train['FireplaceQu'] = train['FireplaceQu'].map(general_map)


# ### SaleCondition

# In[ ]:


train['SaleCondition'].value_counts()


# In[ ]:


train['SaleCondition'].isna().sum()


# In[ ]:


sns.boxplot(y='SalePrice', x='SaleCondition', data=train)


# Let's one-hot encode this feature

# In[ ]:


train = pd.concat([train.drop('SaleCondition', axis=1), pd.get_dummies(train['SaleCondition'], prefix='SaleCondition')], axis=1)


# ### BsmtFinType1 and BsmtFinType2

# In[ ]:


train['BsmtFinType1'] = train['BsmtFinType1'].map(bsmtfintype_map)
train['BsmtFinType2'] = train['BsmtFinType2'].map(bsmtfintype_map)


# ### Functional

# In[ ]:


train['Functional'] = train['Functional'].map(functional_map)


# ## Exploring the correlation between features

# In[ ]:


import scipy.cluster.hierarchy as spc

corr = train.corr().values

pdist = spc.distance.pdist(corr)
linkage = spc.linkage(pdist, method='complete')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')

col_to_cluster = {}

for i,col in enumerate(train.corr().columns):
    
    col_to_cluster[col] = idx[i]


# In[ ]:


fig, ax = plt.subplots(figsize=[20,20])
sns.heatmap(train[sorted(col_to_cluster, key=lambda k: col_to_cluster[k])].corr(),
            annot=True, cbar=False, cmap='Blues', fmt='.1f')


# ## Baseline model

# In[ ]:


numerical_cols = []
for col in train.columns:
    if col!='Id' and train[col].dtype!='object' and col!='SalePrice':
        numerical_cols.append(col)


# In[ ]:


X_df = train[numerical_cols].fillna(-999)


# In[ ]:


X_train = np.array(X_df)
X_test = preprocess_test(test)


# In[ ]:


X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)


# In[ ]:


y_train = np.array(train['SalePrice'], ndmin=2).reshape(-1,1)


# In[ ]:


y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)


# In[ ]:


rkf = RepeatedKFold(n_splits=6, n_repeats=5)


# In[ ]:


def get_model():
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='adam', loss='msle')
    return nn_model


# In[ ]:


reg_models = {
    'RF': RandomForestRegressor(n_estimators=250),
    'LGB': LGBMRegressor(n_estimators=200),
    'XGB': XGBRegressor(n_estimators=200, objective='reg:squarederror'),
    'ADA': AdaBoostRegressor(n_estimators=250),
    'KNN': KNeighborsRegressor(n_neighbors=7)
}


# In[ ]:


all_models = []
for model in reg_models:
    print(model)
    models = evaluate_model(reg_models[model], X_train, y_train, rkf, y_scaler)
    all_models += models


# In[ ]:


rmsles = []
nn_models = []
for train_idx, val_idx in rkf.split(X_train):
    nn_model = get_model()
    nn_model.fit(X_train[train_idx], y_train[train_idx], epochs=30, verbose=0)
    y_pred = nn_model.predict(X_train[val_idx])
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1)).clip(0, np.inf)
    y_true = y_scaler.inverse_transform(y_train[val_idx].reshape(-1,1))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    rmsles.append(rmsle)
    nn_models.append(nn_model)
    
print("RMSLE: {:.5f} +- {:.5f}".format(np.mean(rmsles), np.std(rmsles)))


# In[ ]:


all_models += nn_models


# In[ ]:


X_train_predictions = np.zeros(shape=(X_train.shape[0], len(all_models)))
for i, model in enumerate(all_models):
    X_train_predictions[:, i] = model.predict(X_train).reshape(-1)
    
X_test_predictions = np.zeros(shape=(X_test.shape[0], len(all_models)))
for i, model in enumerate(all_models):
    X_test_predictions[:, i] = model.predict(X_test).reshape(-1)


# In[ ]:


models = []
for train_idx, val_idx in rkf.split(X_train_predictions):
    meta_regressor = LinearRegression()
    meta_regressor.fit(X_train_predictions[train_idx], y_train[train_idx])
    y_pred = meta_regressor.predict(X_train_predictions[val_idx])
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1))
    y_true = y_scaler.inverse_transform(y_train[val_idx].reshape(-1,1))
    print("rmsle = {}".format(np.sqrt(mean_squared_log_error(y_true, y_pred))))
    models.append(meta_regressor)


# ### Submitting to kaggle

# In[ ]:


y_pred = np.array([model.predict(X_test_predictions) for model in models]).mean(axis=0).reshape(-1,1)


# In[ ]:


y_pred = y_scaler.inverse_transform(y_pred)


# In[ ]:


sample['SalePrice'] = y_pred


# In[ ]:


sample.to_csv('sample_submission.csv', index=False)

