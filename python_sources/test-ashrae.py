#!/usr/bin/env python
# coding: utf-8

# Based on this notebook: https://www.kaggle.com/hmendonca/starter-eda-and-feature-selection-ashrae3

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, gc
import random
import datetime

from tqdm import tqdm_notebook as tqdm

# matplotlib and seaborn for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing

import lightgbm as lgb


# In[ ]:


path = '../input/ashrae-energy-prediction'
# Input data files are available in the "../input/" directory.
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Load data and display samples:

# In[ ]:


get_ipython().run_cell_magic('time', '', "# unimportant features (see importance below)\nunimportant_cols = ['wind_direction', 'wind_speed', 'sea_level_pressure']\ntarget = 'meter_reading'\n\ndef load_data(source='train', path=path):\n    ''' load and merge all tables '''\n    assert source in ['train', 'test']\n    \n    building = pd.read_csv(f'{path}/building_metadata.csv', dtype={'building_id':np.uint16, 'site_id':np.uint8})\n    weather  = pd.read_csv(f'{path}/weather_{source}.csv', parse_dates=['timestamp'],\n                                                           dtype={'site_id':np.uint8, 'air_temperature':np.float16,\n                                                                  'cloud_coverage':np.float16, 'dew_temperature':np.float16,\n                                                                  'precip_depth_1_hr':np.float16},\n                                                           usecols=lambda c: c not in unimportant_cols)\n    df = pd.read_csv(f'{path}/{source}.csv', dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])\n    df = df.merge(building, on='building_id', how='left')\n    df = df.merge(weather, on=['site_id', 'timestamp'], how='left')\n    return df\n\n# load and display some samples\ntrain = load_data('train')\ntrain.sample(7)")


# ### Test data:

# In[ ]:


test = load_data('test')
test.sample(7)


# ### Data preprocessing:

# In[ ]:


# the counts above expose the missing data (Should we drop or refill the missing data?)
print("Ratio of available data (not NAN's):")
data_ratios = train.count()/len(train)
data_ratios


# In[ ]:


class ASHRAE3Preprocessor(object):
    @classmethod
    def fit(cls, df, data_ratios=data_ratios):
        cls.avgs = df.loc[:,data_ratios < 1.0].mean()
        cls.pu_le = LabelEncoder()
        cls.pu_le.fit(df["primary_use"])

    @classmethod
    def transform(cls, df):
        df = df.fillna(cls.avgs) # refill NAN with averages
        df['primary_use'] = np.uint8(cls.pu_le.transform(df['primary_use']))  # encode labels

        # expand datetime into its components
        df['hour'] = np.uint8(df['timestamp'].dt.hour)
        df['day'] = np.uint8(df['timestamp'].dt.day)
        df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
        df['month'] = np.uint8(df['timestamp'].dt.month)
        df['year'] = np.uint8(df['timestamp'].dt.year-2000)
        
        # parse and cast columns to a smaller type
        df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)
        df['log_square_feet'] = np.float16(np.log(df['log_square_feet']))
        df['year_built'] = np.uint8(df['year_built']-1900)
        df['floor_count'] = np.uint8(df['floor_count'])
        
        # remove redundant columns
        for col in df.columns:
            if col in ['timestamp', 'row_id']:
                del df[col]
    
        # extract target column
        if 'meter_reading' in df.columns:
            df['meter_reading'] = np.log1p(df['meter_reading']).astype(np.float32) # comp metric uses log errors

        return df
        
ASHRAE3Preprocessor.fit(train)


# In[ ]:


train = ASHRAE3Preprocessor.transform(train)
train.sample(7)


# In[ ]:


train.dtypes


# ### Feature ranked correlation

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(figsize=(16,8))\n# use a ranked correlation to catch nonlinearities\ncorr = train[[col for col in train.columns if col != 'year']].sample(100100).corr(method='spearman')\n_ = sns.heatmap(corr, annot=True,\n                xticklabels=corr.columns.values,\n                yticklabels=corr.columns.values)")


# In[ ]:


# force the model to use the weather data instead of dates, to avoid overfitting to the past history
features = [col for col in train.columns if col not in [target, 'year', 'month', 'day']]


# ### Train-validation partition:

# In[ ]:


# Shuffle:
n = train.shape[0]
ix = np.random.permutation(n)


# In[ ]:


# Training data:
sep = 15000000
tr_x, tr_y = train[features].iloc[ix[:sep]], train[target][ix[:sep]]
va_x, va_y = train[features].iloc[ix[sep:]], train[target][ix[sep:]]


# In[ ]:


xtr = tr_x.values
ytr = tr_y.values
xval = va_x.values
yval = va_y.values


# In[ ]:


print(xtr.shape)
print(ytr.shape)
print(xval.shape)
print(yval.shape)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtr = scaler.fit_transform(xtr)
xval = scaler.transform(xval)


# In[ ]:


print(xtr.mean())
print(xtr.std())
print(xtr.shape)

print(xval.mean())
print(xval.std())
print(xval.shape)


# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


from keras import backend as K

def rmlse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred + 1.0) - K.log(y_true + 1.0))))
    #return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


# In[ ]:


model = keras.Sequential(name="regresion lineal")

# model.add(keras.layers.Dense(1, activation="linear", input_shape=(13,)))
model.add(keras.layers.Dense(50, input_shape=(13,)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(500, activation="selu"))
model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(1, activation="linear"))

model.compile(optimizer=keras.optimizers.Adam(1.e-2), loss='mse', metrics=[rmlse])

model.summary()


# In[ ]:


history = model.fit(xtr, 
                    ytr, 
                    epochs=10, 
                    validation_data=(xval, yval),
                    batch_size=8000)


# In[ ]:


loss_val, acc_val = model.evaluate(xval, yval)
print("Loss on val set = %f" % (loss_val))
print("Accuracy on val set = %f" % (acc_val))


# In[ ]:


val_preds = model.predict(xval)
print(val_preds)


# In[ ]:


val_preds.shape


# In[ ]:


test = ASHRAE3Preprocessor.transform(test)
test.sample(7)


# In[ ]:


tst_x = test[features].iloc[:]


# In[ ]:


xtst = tst_x.values
print(xtst.shape)


# In[ ]:


xtst = scaler.transform(xtst)


# In[ ]:


print(xtst.mean())
print(xtst.std())
print(xtst.shape)


# In[ ]:


tst_preds = model.predict(xtst)
print(tst_preds.shape)


# In[ ]:


submission = pd.read_csv(f'{path}/sample_submission.csv')
submission['meter_reading'] = np.clip(tst_preds, a_min=0, a_max=None) # clip min at zero


# In[ ]:


submission.head(9)


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




