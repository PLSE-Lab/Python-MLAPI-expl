#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import describe
pd.options.display.max_columns = 12
pd.options.display.max_rows = 24


# In[ ]:


# disable warnings in Anaconda
import warnings

warnings.simplefilter('ignore')


# In[ ]:


# plots inisde jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns
sns.set()


# In[ ]:


# use svg for all plots within inline backend
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[ ]:


# increase default plot size
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4


# In[ ]:


import pandas as pd
pd.options.display.max_columns = 12
pd.options.display.max_rows = 24

# disable warnings in Anaconda
import warnings

warnings.simplefilter('ignore')

# plots inisde jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

# use svg for all plots within inline backend
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# increase default plot size
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4


# # Load data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.head()


# ## Convert strings to dates

# In[ ]:


df_train['date'] = pd.to_datetime(df_train['date'])
df_train.index = pd.DatetimeIndex(df_train['date'])
df_train.drop('date', axis=1, inplace=True)


# In[ ]:


df_train.info()


# ## Sales for each storeitem

# In[ ]:


from itertools import product, starmap


def storeitems():
    return product(range(1,51), range(1,11))


def storeitems_column_names():
    return list(starmap(lambda i,s: f'item_{i}_store_{s}_sales', storeitems()))


def sales_by_storeitem(df):
    ret = pd.DataFrame(index=df.index.unique())
    for i, s in storeitems():
        ret[f'item_{i}_store_{s}_sales'] = df[(df['item'] == i) & (df['store'] == s)]['sales'].values
    return ret


# In[ ]:


df_train = sales_by_storeitem(df_train)


# In[ ]:


df_train.info()


# ## Test data
# Loading test data will allow  us to have correct scaling on entire set, and perform all necessary transformation on combined data.

# In[ ]:


# load data
df_test = pd.read_csv('../input/test.csv')
df_test.head()


# In[ ]:


# strings to dates
df_test['date'] = pd.to_datetime(df_test['date'])
df_test.index = pd.DatetimeIndex(df_test['date'])
df_test.drop('date', axis=1, inplace=True)
df_test.info()


# In[ ]:


# mock sales to use same transformations as in df_train
df_test['sales'] = np.zeros(df_test.shape[0])
df_test = sales_by_storeitem(df_test)
df_test.info()


# ## Combine test and train datasets

# In[ ]:


# make sure all column names are the same and in the same order
col_names = list(zip(df_test.columns, df_train.columns))
for cn in col_names:
    assert cn[0] == cn[1]


# In[ ]:


df_test['is_test'] = np.repeat(True, df_test.shape[0])
df_train['is_test'] = np.repeat(False, df_train.shape[0])
df_total = pd.concat([df_train, df_test])
df_total.info()


# ## 1hot encoding weekdays and months

# In[ ]:


weekday_df = pd.get_dummies(df_total.index.weekday, prefix='weekday')
weekday_df.index = df_total.index
weekday_df.head()


# In[ ]:


month_df = pd.get_dummies(df_total.index.month, prefix='month')
month_df.index =  df_total.index
month_df.head()


# In[ ]:


df_total = pd.concat([weekday_df, month_df, df_total], axis=1)
df_total.info()


# In[ ]:


assert df_total.isna().any().any() == False


# ## Shift sales 
# Doing this on combined dataset allows us to have first value for test set already calculated.

# In[ ]:


def shift_series(series, days):
    return series.transform(lambda x: x.shift(days))


def shift_series_in_df(df, series_names=[], days_delta=90):
    """
    Shift columns in df with names in series_names by days_delta.
    
    Negative days_delta will prepend future values to current date,
    positive days_delta wil prepend past values to current date.
    """
    ret = pd.DataFrame(index=df.index.copy())
    str_sgn = 'future' if np.sign(days_delta) < 0 else 'past'
    for sn in series_names:
        ret[f'{sn}_{str_sgn}_{np.abs(days_delta)}'] = shift_series(df[sn], days_delta)
    return ret

    
def stack_shifted_sales(df, days_delta=90):
    names = storeitems_column_names()
    dfs = [df.copy()]
    abs_range = range(1, days_delta+1) if days_delta > 0 else range(days_delta, 0)
    for day_offset in abs_range:
        delta = -day_offset
        shifted = shift_series_in_df(df, series_names=names, days_delta=delta)
        dfs.append(shifted)
    return pd.concat(dfs, axis=1, copy=False)


# In[ ]:


df_total = stack_shifted_sales(df_total, days_delta=-1)


# In[ ]:


df_total = df_total.dropna()  # this should ONLY remove 1st row
df_total.info()


# In[ ]:


# make sure stacked and standard sales columns appear in the same order:
sales_cols = [col for col in df_total.columns if '_sales' in col and '_sales_' not in col]
stacked_sales_cols = [col for col in df_total.columns if '_sales_' in col]
other_cols = [col for col in df_total.columns if col not in set(sales_cols) and col not in set(stacked_sales_cols)]

sales_cols = sorted(sales_cols)
stacked_sales_cols = sorted(stacked_sales_cols)

new_cols = other_cols + stacked_sales_cols + sales_cols


# In[ ]:


df_total = df_total.reindex(columns=new_cols)


# In[ ]:


df_total.head()


# In[ ]:


df_total.tail()


# In[ ]:


df_total.describe()


# In[ ]:


assert df_total.isna().any().any() == False


# ## Scaling
# With combined datasets and shifted sales, we can now correctly min-max scale all data.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[ ]:


cols_to_scale = [col for col in df_total.columns if 'weekday' not in col and 'month' not in col]


# In[ ]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_cols = scaler.fit_transform(df_total[cols_to_scale])
df_total[cols_to_scale] = scaled_cols
df_total.head()


# In[ ]:


df_total.describe()


# ## Split back to train and test set
# `df_total` will be still available - it will be necessary to reverse scaling on output data (sales predictions from the model)

# In[ ]:


df_train = df_total[df_total['is_test'] == False].drop('is_test', axis=1)
df_test = df_total[df_total['is_test'] == True].drop('is_test', axis=1)


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# ## Split to X and y (for training)

# In[ ]:


X_cols_stacked = [col for col in df_train.columns if '_past_' in col]
X_cols_caldata = [col for col in df_train.columns if 'weekday_' in col or 'month_' in col]
X_cols = X_cols_stacked + X_cols_caldata

X = df_train[X_cols]


# In[ ]:


X_colset = set(X_cols)
y_cols = [col for col in df_train.columns if col not in X_colset]

y = df_train[y_cols]


# In[ ]:


X.info()


# In[ ]:


y.info()


# ## Shape for Keras LSTM

# In[ ]:


# split values to train and test, use np arrays to allow reshaping
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)


# In[ ]:


# reshape inputs to be 3d, as in: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
X_train_vals = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_valid_vals = X_valid.values.reshape((X_valid.shape[0], 1, X_valid.shape[1]))


# ## Training LSTM
# Using features for all storeitems (stacked sales from previous day) to predict sales for one storeitem (sales for current day).
# 
# 500 features, 1826 samples

# In[ ]:


from keras.models import Sequential, Model
from keras.layers import *


# In[ ]:


# model alternative 2 - double with conv1ds TODO
# https://arxiv.org/pdf/1709.05206.pdf
def build_model():
    inputs = Input(shape=(X_train_vals.shape[1], X_train_vals.shape[2]))
    # top pipeline
    top_lstm = LSTM(500, return_sequences=True)(inputs)
    top_dense = Dense(500, activation='relu')(top_lstm)
    # bottom pipeline
    bottom_dense = Dense(500)(inputs)
    bottom_conv1 = Conv1D(
        500, 
        kernel_size=1,
        input_shape=(X_train_vals.shape[1], X_train_vals.shape[2])
    )(bottom_dense)
    bottom_conv2 = Conv1D(
        1000,
        kernel_size=50,
        padding='same',
        activation='relu'
    )(bottom_conv1)
    bottom_conv3 = Conv1D(
        500,
        kernel_size=10,
        padding='same',
        activation='relu'
    )(bottom_conv2)
    bottom_pooling = AvgPool1D(
        pool_size=10, 
        padding='same'
    )(bottom_conv3)
#     bottom_reshape = Reshape(
#         target_shape=[500]
#     )(bottom_conv3)
    # concat output
    final_concat = Concatenate()([top_dense, bottom_pooling])
    final_lstm = LSTM(1000, dropout=0.2)(final_concat)
    final_dense = Dense(500)(final_lstm)
    # compile and return
    model = Model(inputs=inputs, outputs=final_dense)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mape'])
    return model

model = build_model()


# In[ ]:


history = model.fit(
    X_train_vals, 
    y_train.values, 
    epochs=130, 
    batch_size=70,
    validation_data=(X_valid_vals, y_valid.values),
    verbose=2,
    shuffle=False
)


# In[ ]:


# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# ## Calculate SMAPE for the model
# SMAPE is calculated on a validation set for the model

# In[ ]:


X_valid.shape


# In[ ]:


y_valid.shape


# In[ ]:


def model_eval(model, X_test, y_test, log_all=False):
    """
    Model must have #predict method.
    X_test, y_test - instances of pd.DataFrame (normal, not reshaped for LSTM !!!)
    
    Note that this function assumes that sales columns for previous values appear 
    in the same order as sales columns for current values.
    """
    # prepare data
    sales_x_cols = [col for col in X_test.columns if 'sales' in col]
    sales_x_idxs = [X_test.columns.get_loc(col) for col in sales_x_cols]
    sales_y_cols = [col for col in y_test.columns if 'sales' in col]
    sales_y_idxs = [y_test.columns.get_loc(col) for col in sales_y_cols]
    n_samples = y_test.shape[0]
    y_pred = np.zeros(y_test.shape)
    # iterate
    x_next = X_test.iloc[0].values
    for i in range(0, n_samples):
        if log_all:
            print('[x]', x_next)
        x_arr = np.array([x_next])
        x_arr = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])
        y_pred[i] = model.predict(x_arr)[0]
        try:
            x_next = X_test.iloc[i+1].values
            x_next[sales_x_idxs] = y_pred[i][sales_y_idxs]
        except IndexError:
            pass  # this happens on last iteration, and x_next does not matter anymore
    return y_pred, y_test.values

def vector_smape(y_pred, y_real):
    nom = np.abs(y_pred-y_real)
    denom = (np.abs(y_pred) + np.abs(y_real)) / 2
    results = nom / denom
    return 100*np.mean(results)  # in percent, same as at kaggle


# Prepare data for calculating SMAPE scores:

# In[ ]:


X_valid, y_valid = X_valid.head(90), y_valid.head(90)


# In[ ]:


y_pred, y_real = model_eval(model, X_valid, y_valid)


# In[ ]:


def unscale(y_arr, scaler, template_df, toint=False):
    """
    Unscale array y_arr of model predictions, based on a scaler fitted 
    to template_df.
    """
    tmp = template_df.copy()
    tmp[y_cols] = pd.DataFrame(y_arr, index=tmp.index)
    tmp[cols_to_scale] = scaler.inverse_transform(tmp[cols_to_scale])
    if toint:
        return tmp[y_cols].astype(int)
    return tmp[y_cols]


# In[ ]:


template_df = pd.concat([X_valid, y_valid], axis=1)
template_df['is_test'] = np.repeat(True, template_df.shape[0])

pred = unscale(y_pred, scaler, template_df, toint=True)
real = unscale(y_real, scaler, template_df, toint=True)


# Calculate SMAPE for each of the items

# In[ ]:


smapes = [vector_smape(pred[col], real[col]) for col in pred.columns]


# In[ ]:


sns.distplot(smapes)


# In[ ]:


describe(smapes)


# ## Plot sample prediction

# In[ ]:


store, item = np.random.randint(1,11), np.random.randint(1,51)
random_storeitem_col = f'item_{item}_store_{store}_sales'


# In[ ]:


plot_lengths = [7, 30, 60, 365]

for pl in plot_lengths:
    plt.plot(pred[random_storeitem_col].values[:pl], label='predicted')
    plt.plot(real[random_storeitem_col].values[:pl], label='real')
    plt.legend()
    plt.show()


# ## Exporting to Kaggle
# Train model on test set (already loaded and formatted) and save results in Kaggle format.

# In[ ]:


# make sure 1st row has correctly stacked sales
df_test[stacked_sales_cols].head(2)


# In[ ]:


# split to X and y
X_test, y_test = df_test[X_cols], df_test[y_cols]


# In[ ]:


# y_test is basically blank, but allows us to use the same function
y_test_pred, _ = model_eval(model, X_test, y_test)


# In[ ]:


test_template_df = pd.concat([X_test, y_test], axis=1)
test_template_df['is_test'] = np.repeat(True, test_template_df.shape[0])

test_pred = unscale(y_test_pred, scaler, test_template_df, toint=True)


# In[ ]:


test_pred.head()


# In[ ]:


plt.plot(test_pred['item_1_store_1_sales'].values)
plt.show()


# In[ ]:


result = np.zeros(45000, dtype=np.int)
for i, s in storeitems():
    slice_start_idx = 90*10*(i-1) + 90*(s-1)
    slice_end_idx = slice_start_idx + 90
    col_name = f'item_{i}_store_{s}_sales'
    result[slice_start_idx:slice_end_idx] = test_pred[col_name].values
result = pd.DataFrame(result, columns=['sales'])
result.index.name = 'id'
result.head()


# In[ ]:


result.to_csv('basic_lstm.csv')

