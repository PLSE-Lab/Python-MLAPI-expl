#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from scipy.stats import skew, norm, boxcox_normmax
from scipy import stats
from scipy.special import boxcox1p 
# models
from xgboost import XGBRegressor
import warnings

# Ignore useless warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Avoid runtime error messages
pd.set_option('display.float_format', lambda x:'%f'%x)

# make notebook's output stable across runs
np.random.seed(42)

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print("Numero de linhas e colunas no train_df : ",train_df.shape)
print("Numero de linhas e colunas no teste_df : ",test_df.shape)


# <b>Analise do arquivo train_df

# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


train_df.dtypes.head(30)


# In[ ]:


#column_names = train_df.columns


# In[ ]:


#verifica valores nulos
train_df.isnull().sum().sort_values(ascending=False)


# In[ ]:


train_df['SalePrice'].describe()


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % train_df['SalePrice'].skew())
print("Kurtosis: %f" % train_df['SalePrice'].kurt())


# Campo SalePrice apresenta Skewness > 1.0, logo o campo precisa ser normalizado (log transformation)

# In[ ]:


train_df.describe(include='O') #categorical features


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train_df['SalePrice'].values, bins=50, kde=False, color="blue")
plt.title("Histogram of SalePrice")
plt.xlabel('SalePrice', fontsize=12)


# In[ ]:


train_df.hist(bins=50, figsize=(20,15))
plt.tight_layout(pad=0.4)


# In[ ]:


#correlation matrix
corrmat = train_df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1.0, square=True, cmap="Blues");


# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


# As 10 features que mais se correlacionam com SalePrice
corr_matrix = train_df.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)[:10]


# In[ ]:


#histogram and normal probability plot
sns.distplot(train_df['SalePrice'], fit=norm);
fig = plt.figure()
#res = stats.probplot(train_df['SalePrice'], plot=plt)


# <b>Tirar Log para normalizar os campos SalePrice e GrLivArea

# In[ ]:


#applying log transformation
train_df['SalePrice'] = np.log(train_df['SalePrice'])


# In[ ]:


#data transformation
train_df['GrLivArea'] = np.log(train_df['GrLivArea'])
test_df['GrLivArea'] = np.log(test_df['GrLivArea'])


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(train_df['SalePrice'], fit=norm);
fig = plt.figure()
#res = stats.probplot(train_df['SalePrice'], plot=plt)


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(train_df['GrLivArea'], fit=norm);
fig = plt.figure()
#res = stats.probplot(train_df['GrLivArea'], plot=plt)


# In[ ]:


train_cp = train_df.copy()
test_cp = test_df.copy()
test_id = test_df["Id"]


# <b>Concatena treino e teste

# In[ ]:


train_cp = train_cp.drop(['SalePrice'], axis=1)


# In[ ]:


all_df = pd.concat([train_cp,test_cp])
print(all_df.shape)
all_df.head()


# In[ ]:


conta_nulo2 = all_df.isnull().sum().sort_values(ascending=False)
conta_nulo2.head(40)


# In[ ]:


all_df.shape 


# <b>Feature Engineering

# In[ ]:


#all_df['temGaragem'] = all_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


all_df['temLareira'] = all_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


all_df = all_df.drop(['Fireplaces'], axis=1) 


# In[ ]:


all_df['GarageCars'] = all_df['GarageCars'].fillna(2.0)
all_df['GarageArea'] = all_df['GarageArea'].fillna(480.0)
all_df['TotalBsmtSF'] = all_df['TotalBsmtSF'].fillna(989.5)


# In[ ]:


all_df_x = pd.DataFrame()


# In[ ]:


all_df_x['Id'] = all_df['Id']
all_df_x['OverallQual'] = all_df['OverallQual']
all_df_x['GrLivArea'] = all_df['GrLivArea']
all_df_x['GarageCars'] = all_df['GarageCars']
all_df_x['GarageArea'] = all_df['GarageArea']
all_df_x['TotalBsmtSF'] = all_df['TotalBsmtSF']
all_df_x['1stFlrSF'] = all_df['1stFlrSF']
all_df_x['FullBath'] = all_df['FullBath']
all_df_x['TotRmsAbvGrd'] = all_df['TotRmsAbvGrd']
all_df_x['YearBuilt'] = all_df['YearBuilt']
all_df_x['Fence'] = all_df['Fence']
all_df_x['temLareira'] = all_df['temLareira']


# In[ ]:


# Get_Dummies para transformar categoricos em Numericos
all_dummy_df = pd.get_dummies(all_df_x)


# In[ ]:


all_dummy_df.shape


# In[ ]:


all_dummy_df.head()


# In[ ]:


train_dummy_df = all_dummy_df[:1460]


# In[ ]:


train_dummy_df.shape


# In[ ]:


train_dummy_df.head()


# In[ ]:


train_df_2 = train_dummy_df
train_df_2["SalePrice"] = train_df["SalePrice"]


# In[ ]:


train_df_2.to_csv("train_hprice.csv", index=False)


# In[ ]:


test_dummy_df = all_dummy_df[1460:]


# In[ ]:


test_dummy_df.shape


# In[ ]:


test_dummy_df.to_csv("test_hprice.csv", index=False)


# In[ ]:


train_dummy_df = train_dummy_df.drop(['Id'], axis=1)
test_dummy_df = test_dummy_df.drop(['Id'], axis=1)


# In[ ]:


train_dummy_df = train_dummy_df.drop(['SalePrice'], axis=1)


# In[ ]:


all_dummy_df = all_dummy_df.drop(['Id'], axis=1)


# In[ ]:


#train_X, test_X, train_y, test_y = train_test_split(train_dummy_df.as_matrix(), train_df['SalePrice'].as_matrix(), test_size=0.25)

train_X = train_dummy_df
test_X = test_dummy_df
train_y = train_df['SalePrice']

print(train_X.shape, test_X.shape)


# In[ ]:


#test_y.shape #array sem colunas


# In[ ]:


#Training a Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fitting the training data to our model
regressor.fit(train_X,train_y)

# Predicting price for the Validation set
y_pred = regressor.predict(test_X)

# Scoring the model
from sklearn.metrics import r2_score, mean_squared_error

# R2 score closer to 1 is a good model
#print(f"R2 score: {r2_score(test_y, y_pred)}")

# MSE score closer to zero is a good model
#print(f"MSE score: {mean_squared_error(test_y, y_pred)}")

#R2 score: 0.8405922641242736
#MSE score: 0.02760294922349491
#An R2 score closer to 1.0 means the model is a better one. Likewise, an MSE score closer to 0 means the model is good.

print(y_pred)


# In[ ]:


y_pred = np.exp(y_pred)


# In[ ]:


y_pred


# In[ ]:


test_id = pd.DataFrame({"Id":test_id.values})


# In[ ]:


sub_df = pd.DataFrame({"Id":test_id["Id"].values})
sub_df["SalePrice"] = y_pred
print(sub_df)


# In[ ]:


sub_df.to_csv("linRegression.csv", index=False)

