#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# 
# # FE & EDA with 3D and abnormals filtering of Used Cars Dataset

# ## An example of effective filtering of anomal values in prediction problems
# 
# **[Used Cars Dataset](https://www.kaggle.com/austinreese/craigslist-carstrucks-data)**
# 
# **Graphs were constructed that showed the presence of anomalies for a number of features**
# 
# **It is suggested to look for the anomalies of the smallest and largest values with the help of quantiles and select filters of values**

# Technology for feature importance building from the my kernels:
# * [NFL feature importance - xgb, lgbm, linreg](https://www.kaggle.com/vbmokin/nfl-feature-importance-xgb-lgbm-linreg)
# * [Feature importance - xgb, lgbm, logreg, linreg](https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg)

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download dataset](#2)
# 1. [Preparing to analysis](#3)
# 1. [EDA](#4)
# 1. [FE: tuning models and building the feature importance diagrams](#5)
#     -  [LGBM](#5.1)
#     -  [XGB](#5.2)
#     -  [Linear Regression](#5.3)
# 1. [Comparison of the all feature importance diagrams](#6)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
import matplotlib.style as style

import lightgbm as lgbm
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score

pd.set_option('max_columns',100)

import warnings
warnings.filterwarnings("ignore")


# ## 2. Download dataset <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


train = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/craigslistVehicles.csv')
train.head(5)


# In[ ]:


train.info()


# ## 3. Preparing to analysis <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to the kernel https://www.kaggle.com/arsnlsnk/used-cars-price-prediction-15-models-5-features
drop_columns = ['url', 'city', 'city_url', 'title_status', 'VIN', 'size', 'image_url', 'desc', 'lat','long', 'paint_color']
train = train.drop(columns = drop_columns)
train = train.dropna()
train.head(5)


# In[ ]:


# from the my kernel: https://www.kaggle.com/vbmokin/automatic-selection-from-20-classifier-models
# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train.columns.values.tolist()
for col in features:
    if train[col].dtype in numerics: continue
    categorical_columns.append(col)
# Encoding categorical features
for col in categorical_columns:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))


# In[ ]:


# Thanks to : https://www.kaggle.com/aantonova/some-new-risk-and-clusters-features
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


# In[ ]:


train = reduce_mem_usage(train)


# In[ ]:


train.info()


# ## 4. EDA <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


train.describe()


# In[ ]:


#Thanks to https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing
def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    
    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
    
plotting_3_chart(train, 'price')


# In[ ]:


#Thanks to https://towardsdatascience.com/an-easy-introduction-to-3d-plotting-with-matplotlib-801561999725

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection="3d")

z_points = train['price']
x_points = train['odometer']
y_points = train['year']
ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');

ax.set_xlabel('odometer')
ax.set_ylabel('year')
ax.set_zlabel('price')

plt.show()


# In[ ]:


y = np.array(train.price)
plt.subplot(131)
plt.plot(range(len(y)),y,'.');plt.ylabel('price');plt.xlabel('index');
plt.subplot(132)
sns.boxplot(y=train.price)


# Check percentiles 5%, 10% and 90%, 95%

# In[ ]:


train_stat = train.describe(percentiles = [.05,.1, .9,.95])
train_stat


# ## Selection the conditions for effective filter of abnormal values

# In[ ]:


train_stat.loc['max',:]-train_stat.loc['95%',:]


# In[ ]:


train_stat.loc['95%',:]-train_stat.loc['90%',:]


# In[ ]:


(train_stat.loc['max',:]-train_stat.loc['95%',:])/(train_stat.loc['95%',:]-train_stat.loc['90%',:])


# In[ ]:


train_stat.loc['10%',:]-train_stat.loc['5%',:]


# In[ ]:


train_stat.loc['5%',:]-train_stat.loc['min',:]


# In[ ]:


(train_stat.loc['5%',:]-train_stat.loc['min',:])/(train_stat.loc['10%',:]-train_stat.loc['5%',:])


# I suggest to make filters only for features that were numeric in the basic dateset:
# price, year, odometer
# 
# There are problems:
# 
# **- smallest values: year**
# 
# **- largest values: price, odometer**
# 
# 
# I suggest adding a **filter by low values of price** because free cars and cars which cost as scrap require other models
# 
# Set the filter by the quantile values of 10% and 95% for abnormal smallest and largest values, respectively

# In[ ]:


train_stat.loc[['10%','90%','95%'],:]


# In[ ]:


train.info()


# In[ ]:


def abnormal_filter(df, threshold_first, threshold_second):
    # Abnormal values filter for DataFrame df:
    # threshold_first (5%-min or max-95%)
    # threshold_second (second diff., times)
    df_describe = df.describe([.05, .1, .9, .95])
    cols = df_describe.columns.tolist()
    i = 0
    abnorm = 0
    for col in cols:
        i += 1
        # abnormal smallest
        P10_5 = df_describe.loc['10%',col]-df_describe.loc['5%',col]
        P_max_min = df_describe.loc['max',col]-df_describe.loc['min',col]
        if P10_5 != 0:
            if (df_describe.loc['5%',col]-df_describe.loc['min',col])/P10_5 > threshold_second:
                #abnormal smallest filter
                df = df[(df[col] >= df_describe.loc['5%',col])]
                print('1: ', i, col, df_describe.loc['min',col],df_describe.loc['5%',col],df_describe.loc['10%',col])
                abnorm += 1
        else:
            if P_max_min > 0:
                if (df_describe.loc['5%',col]-df_describe.loc['min',col])/P_max_min > threshold_first:
                    # abnormal smallest filter
                    df = df[(df[col] >= df_describe.loc['5%',col])]
                    print('2: ', i, col, df_describe.loc['min',col],df_describe.loc['5%',col],df_describe.loc['max',col])
                    abnorm += 1

        
        # abnormal biggest
        P95_90 = df_describe.loc['95%',col]-df_describe.loc['90%',col]
        if P95_90 != 0:
            if (df_describe.loc['max',col]-df_describe.loc['95%',col])/P95_90 > threshold_second:
                #abnormal biggest filter
                df = df[(df[col] <= df_describe.loc['95%',col])]
                print('3: ', i, col, df_describe.loc['90%',col],df_describe.loc['95%',col],df_describe.loc['max',col])
                abnorm += 1
        else:
            if P_max_min > 0:
                if ((df_describe.loc['max',col]-df_describe.loc['95%',col])/P_max_min > threshold_first) & (df_describe.loc['95%',col] > 0):
                    # abnormal biggest filter
                    df = df[(df[col] <= df_describe.loc['95%',col])]
                    print('4: ', i, col, df_describe.loc['min',col],df_describe.loc['95%',col],df_describe.loc['max',col])
                    abnorm += 1
    print('Number of abnormal values =', abnorm)
    return df


# In[ ]:


train = abnormal_filter(train, 0.5, 3)
train.info()


# In[ ]:


# Add filter: train['price'] >= 1700
train = train[train['price'] >= 1700]
train.info()


# In[ ]:


# Manual filter: price (upper (90%) and lower (10%)), year (lower - 10%) and odometer (upper - 90%)
# train = train[((train['price'] >= 1700) & (train['price'] < 31500) & (train['year'] >= 2001) & (train['odometer'] < 217000))]
# train.info()


# In[ ]:


plotting_3_chart(train, 'price')


# In[ ]:


#Thanks to https://towardsdatascience.com/an-easy-introduction-to-3d-plotting-with-matplotlib-801561999725

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection="3d")

z_points = train['price']
x_points = train['odometer']
y_points = train['year']
ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');

ax.set_xlabel('odometer')
ax.set_ylabel('year')
ax.set_zlabel('price')

plt.show()


# In[ ]:


target = train['price']
del train['price']


# ## 5. FE: tuning models and building the feature importance diagrams<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 5.1 LGBM <a class="anchor" id="5.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


X = train
z = target


# In[ ]:


#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)
train_set = lgbm.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgbm.Dataset(Xval, Zval, silent=False)


# In[ ]:


params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,        
    }

modelL = lgbm.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)


# In[ ]:


r2_score(Zval, modelL.predict(Xval))


# In[ ]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgbm.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


feature_score = pd.DataFrame(train.columns, columns = ['feature']) 
feature_score['score_lgb'] = modelL.feature_importance()


# ### 5.2 XGB<a class="anchor" id="5.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#%% split training set to validation set 
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xval   , label=Zval)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]


# In[ ]:


parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:squarederror',
         'eta'      :0.3,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10}
modelx = xgb.train(parms, data_tr, num_boost_round=200, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=10)

print('score = %1.5f, n_boost_round =%d.'%(modelx.best_score,modelx.best_iteration))


# In[ ]:


r2_score(Zval, modelx.predict(data_cv))


# In[ ]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(modelx,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


feature_score['score_xgb'] = feature_score['feature'].map(modelx.get_score(importance_type='weight'))
feature_score


# ### 5.3 Linear Regression <a class="anchor" id="5.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Standardization for regression model
train = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(train),
    columns=train.columns,
    index=train.index
)


# In[ ]:


# Linear Regression

linreg = LinearRegression()
linreg.fit(train, target)
coeff_linreg = pd.DataFrame(train.columns.delete(0))
coeff_linreg.columns = ['feature']
coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)
coeff_linreg.sort_values(by='score_linreg', ascending=False)


# In[ ]:


coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()
feature_score = pd.merge(feature_score, coeff_linreg, on='feature')
feature_score = feature_score.fillna(0)
feature_score = feature_score.set_index('feature')
feature_score


# ### 6. Comparison of the all feature importance diagrams <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#Thanks to https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
# MinMax scale all importances
feature_score = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(feature_score),
    columns=feature_score.columns,
    index=feature_score.index
)

# Create mean column
feature_score['mean'] = feature_score.mean(axis=1)

# Plot the feature importances
feature_score.sort_values('mean', ascending=False).plot(kind='bar', figsize=(20, 7))


# In[ ]:


feature_score.sort_values('mean', ascending=False)


# In[ ]:


# Create total column with different weights
feature_score['total'] = 0.5*feature_score['score_lgb'] + 0.35*feature_score['score_xgb'] + 0.15*feature_score['score_linreg']

# Plot the feature importances
feature_score.sort_values('total', ascending=False).plot(kind='bar', figsize=(20, 7))


# In[ ]:


feature_score.sort_values('total', ascending=False)

