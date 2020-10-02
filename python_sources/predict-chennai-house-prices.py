#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,VotingClassifier
from sklearn.linear_model import SGDRegressor, LinearRegression, BayesianRidge, Lasso, HuberRegressor, ElasticNetCV
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
sns.set(color_codes=True)
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import xgboost as xgb
import lightgbm as lgb
np.random.seed(25)


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train['N_BEDROOM'].value_counts()


# In[ ]:


train.dtypes


# In[ ]:


train.isnull().sum(axis=0)


# In[ ]:


train = train.fillna({"N_BEDROOM": train["N_BEDROOM"].mean(), "INT_SQFT": train["INT_SQFT"].mean(),
                     "N_BATHROOM": train["N_BATHROOM"].mean(), "QS_OVERALL": train["QS_OVERALL"].mean()})
test = test.fillna({"N_BEDROOM": test["N_BEDROOM"].mean(), "INT_SQFT": train["INT_SQFT"].mean(),
                     "N_BATHROOM": test["N_BATHROOM"].mean(), "QS_OVERALL": test["QS_OVERALL"].mean()})


# In[ ]:


# train['N_BEDROOM'] = train['N_BEDROOM'].astype(int)
# test['N_BEDROOM'] = test['N_BEDROOM'].astype(int)

# train['N_BATHROOM'] = train['N_BATHROOM'].astype(int)
# test['N_BATHROOM'] = test['N_BATHROOM'].astype(int)


# # Feature Engineering

# In[ ]:


train['diff_reg_comm'] = train['REG_FEE'] - train['COMMIS']
test['diff_reg_comm'] = test['REG_FEE'] - test['COMMIS']


# In[ ]:


def get_month(x):
    txt = str(x)
    y = txt.split('-')
    if len(y) != 3:
        return 'None'
    return int(y[1])

def get_year(x):
    txt = str(x)
    y = txt.split('-')
    if len(y) != 3:
        return 'None'
    return int(y[2])


# In[ ]:


train['MONTH'] = train['DATE_SALE'].map(lambda x: get_month(x))
train['YEAR'] = train['DATE_SALE'].map(lambda x: get_year(x))

test['MONTH'] = test['DATE_SALE'].map(lambda x: get_month(x))
test['YEAR'] = test['DATE_SALE'].map(lambda x: get_year(x))


# In[ ]:


train['DATE_SALE'] = pd.to_datetime(train['DATE_SALE'].astype(str), format='%d-%m-%Y')
train['DATE_BUILD'] = pd.to_datetime(train['DATE_BUILD'].astype(str), format='%d-%m-%Y')

test['DATE_SALE'] = pd.to_datetime(test['DATE_SALE'].astype(str), format='%d-%m-%Y')
test['DATE_BUILD'] = pd.to_datetime(test['DATE_BUILD'].astype(str), format='%d-%m-%Y')


# In[ ]:


train['HOUSE_LIFETIME'] = train['DATE_SALE'] .subtract( train['DATE_BUILD'])
test['HOUSE_LIFETIME'] = test['DATE_SALE'] .subtract( test['DATE_BUILD'])

train['HOUSE_LIFETIME'] = train['HOUSE_LIFETIME'] .divide( np.timedelta64(1, 'D'))
test['HOUSE_LIFETIME'] = test['HOUSE_LIFETIME'] .divide( np.timedelta64(1, 'D'))


# In[ ]:


sns.distplot(train['SALES_PRICE'] , kde=False);


# In[ ]:


var = 'SALES_PRICE'
data = pd.concat([train['HOUSE_LIFETIME'], train[var]], axis=1)
data.plot.scatter(x=var, y='HOUSE_LIFETIME');        # ylim=(0,2500)


# In[ ]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


feature_names = [x for x in train.columns if x not in ['PRT_ID','SALES_PRICE','DATE_SALE','DATE_BUILD']]
target = train['SALES_PRICE']


# In[ ]:


# One-hot encoding
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
train["AREA"] = lb_make.fit_transform(train["AREA"])
test["AREA"] = lb_make.fit_transform(test["AREA"])

train["SALE_COND"] = lb_make.fit_transform(train["SALE_COND"])
test["SALE_COND"] = lb_make.fit_transform(test["SALE_COND"])

train["PARK_FACIL"] = lb_make.fit_transform(train["PARK_FACIL"])
test["PARK_FACIL"] = lb_make.fit_transform(test["PARK_FACIL"])

train["BUILDTYPE"] = lb_make.fit_transform(train["BUILDTYPE"])
test["BUILDTYPE"] = lb_make.fit_transform(test["BUILDTYPE"])

train["UTILITY_AVAIL"] = lb_make.fit_transform(train["UTILITY_AVAIL"])
test["UTILITY_AVAIL"] = lb_make.fit_transform(test["UTILITY_AVAIL"])

train["STREET"] = lb_make.fit_transform(train["STREET"])
test["STREET"] = lb_make.fit_transform(test["STREET"])

train["MZZONE"] = lb_make.fit_transform(train["MZZONE"])
test["MZZONE"] = lb_make.fit_transform(test["MZZONE"])


# In[ ]:


train[feature_names].dtypes


# In[ ]:


np.arange(len(feature_names))


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 21))
train[feature_names] = scaler.fit_transform(train[feature_names])
test[feature_names] = scaler.fit_transform(test[feature_names])


# In[ ]:


np.random.seed(25)
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
import codecs


# In[ ]:


X_tr, X_val, y_tr, y_val = train_test_split(train[feature_names], target)


# In[ ]:


X_tr.shape


# In[ ]:


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


# In[ ]:


seed = 7
np.random.seed(seed)

# Model
model = Sequential()
model.add(Dense(200, input_dim=22, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss=root_mean_squared_error, optimizer=optimizers.Adadelta())
#model.compile( optimizer=opt,loss=root_mean_squared_error)


# In[ ]:


model.fit(X_tr,y_tr,validation_data=(X_val,y_val),nb_epoch=100,batch_size=10,verbose=2)


# In[ ]:


pred = model.predict(test[feature_names])
pred[:10]


# In[ ]:


## make submission
sub = pd.DataFrame()
sub['PRT_ID'] = test['PRT_ID']
sub['SALES_PRICE'] = pred
sub.to_csv('house_prices.csv', index=False)


# In[ ]:




