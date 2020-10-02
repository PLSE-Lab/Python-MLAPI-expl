#!/usr/bin/env python
# coding: utf-8

# # M5 Forecasting - LSTM w/ Custom Generator
# 
# This notebook shows LSTM training/prediction with a custom data generator for Keras LSTM model. The model uses sequences of sales and prices of {w_size} days with categorical features being used with embeddings to predict next one day sales on each item.
# For the submission, it makes prediction with 28 days loop where each one day prediction is used for an input for the next days' prediction in the loop. 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
import gc

from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.layers import Input, Flatten, Concatenate, BatchNormalization, Embedding
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.compat.v1.keras.layers import CuDNNLSTM


# ## Configuration

# In[ ]:


DATA_DIR = '/kaggle/input/m5-forecasting-accuracy/'

DEBUG = False # turning on/off degugging mode
CV = False # turning on/off cross validation

if DEBUG:
    rows = 100
    w_size = 15
    batch_size=32
    epochs = 3
    span_lst = [7]
else:
    rows = None
    w_size = 30 # LSTM window size
    batch_size=512
    epochs = 35
    span_lst = [7, 30, 90] # moving avarage time wiondows


# ## Training Dataset
# 
# ### Sales Dataset
# * Transposing (items, days) dimension into (days, items)
# * Only last 360 days are used for now to save run time

# In[ ]:


d_dtypes = {}
for i in range(1914):
    d_dtypes[f'd_{i}'] = np.int32
    
sales = pd.read_csv(DATA_DIR + 'sales_train_validation.csv',
                    dtype=d_dtypes, nrows=rows)

# categories are used for categorical model input
categories = sales[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
sales['id'] = sales['id'].apply(lambda x: x[:-11])
ids = sales['id'].values

if DEBUG:
    sales = sales.iloc[:, -150:].T.reset_index()
else:
    # Only last 360 days are used to save run time
    sales = sales.iloc[:, -360:].T.reset_index()
    
sales.columns = ['d'] + list(ids)


# ### Calendar Dataset

# In[ ]:


calendar = pd.read_csv(DATA_DIR + 'calendar.csv',
                       dtype={'wm_yr_wk': np.int32, 'wday': np.int32, 
                              'month': np.int32, 'year': np.int32, 
                              'snap_CA': np.int32, 'snap_TX': np.int32,
                              'snap_WI': np.int32})

# subsetting by starting date in sales
calendar = calendar[calendar.d.apply(lambda x: int(x[2:])) >= int(sales.d[0][2:])]


# ### Price Dataset
# * Transposing the long format into (weeks, items) dimension
# * Then it is merged to calendar data to make it daily format

# In[ ]:


prices = pd.read_csv(DATA_DIR + 'sell_prices.csv',
                          dtype={'wm_yr_wk': np.int32, 
                                 'sell_price': np.float32})
prices = prices.loc[prices.wm_yr_wk >=                     calendar[calendar.d == sales.d[0]]['wm_yr_wk'].values[0]]

prices['id'] = prices.apply(lambda x: x.item_id + '_' + x.store_id, axis=1)
prices = prices.pivot(index='wm_yr_wk', columns='id', values='sell_price')

prices = calendar[['d','wm_yr_wk']].merge(prices, how='inner', on=['wm_yr_wk'])
prices.drop('wm_yr_wk', axis=1, inplace=True)
prices = prices.loc[:, list(sales.columns)]

calendar.drop(['date','wm_yr_wk', 'weekday', 'd'], axis=1, inplace=True)


# ## Preprocessing
# 
# Both sales and prices are log scaled and then standardized by global average and std.

# In[ ]:


sales_log = np.log(sales.iloc[:, 1:].values + 1)
sales_mean = np.mean(sales_log)
sales_std = np.std(sales_log)
sales.iloc[:, 1:] = (sales_log - sales_mean) / sales_std

prices_log = np.log(prices.iloc[:, 1:].values)
prices_mean = np.mean(prices_log)
prices_std = np.std(prices_log)
prices.iloc[:, 1:] = (prices_log - prices_mean) / prices_std

sales.fillna(0, inplace=True)
prices.fillna(0, inplace=True)


# Categorical features are label encoded to be used for embeddings.

# In[ ]:


cat_ft1 = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
cat_ft2 = ['wday','month', 'year', 'event_name_1', 'event_type_1',
           'event_name_2', 'event_type_2']

category_counts = {}
state_le = None

def LabelEncoding(df, cat_ft):
    
    for col in cat_ft:
        le = LabelEncoder()
        df.loc[:, col] = df[col].astype(str)
        df.loc[:, col] = le.fit_transform(df[col])
        category_counts[col] = len(list(le.classes_))

    return df

categories = LabelEncoding(categories, cat_ft1)
calendar = LabelEncoding(calendar, cat_ft2)


# ## Defining SequenceGenerator
# 
# Custom data generator is used to create input sequences and scalers during the model training.
# 
# * Sales:
#   
#   Raw sales values along with its moving averages (7days, 30days, 90days) are used for model input sequences.
#   
#   input shape: (days, items) -> output shape: (items, days, features)
#   
# * Prices:
#   
#   Raw price sequence values are used.
#   
#   input shape: (days, items) -> output shape: (items, days, feature)
#   
# * Categories:
#   
#   item_id, dept_id, cat_id, store_id, state_id are used as a single values for each row. These are fed into embedding layer.
#   
#   input shape: (items, features) -> output shape: (items, 1) * features
#   
# * Calendar:
# 
#   wday, month, year, event_name_1, event_type_1, event_name_2, event_type_2 are used as a single values for each row. These are going to embedding layer. Snap indicator corresponding to the item's state are also fetched and it will be directly put into dense layer.
#   
#   input shape: (days, features) -> output shape: (items, 1) * features

# In[ ]:


def moving_average(a, n):
    
    if a.shape[0] >= n:
        ret = np.cumsum(a, axis=0)
        ret[n:, :] = ret[n:, :] - ret[:-n, :]
        ret[:n-1, :] = np.zeros((n-1, ret.shape[1]))
        return ret / n
    else:
        return np.zeros((a.shape[0], a.shape[1]))
    
class SequenceGenerator:
    
    def __init__(self, inputs, spans=[7], window=30, batch_size=32, infer=False):
        self.sales = inputs[0]
        self.prices = inputs[1]
        self.categories = inputs[2]
        self.calendar = inputs[3]
        self.spans = spans
        self.window = window
        self.infer = infer
        self.num_items = self.sales.shape[1]
        
        if self.infer:
            self.batch_size = self.num_items
            self.num_days = self.sales.shape[0] - self.window + 1
            self.steps_per_day = 1 
            self.steps = 1
        else:
            self.batch_size = batch_size
            self.num_days = self.sales.shape[0] - self.window
            self.steps_per_day = self.num_items // self.batch_size + 1
            self.steps = self.steps_per_day * self.num_days

    def generate(self):
        
        ## for inference, it starts from the the last starting date (no slides in days)
        ## for training/validation, it starts from day 0
        start_day = self.num_days - 1 if self.infer else 0
            
        while True:            
            
            for day in range(start_day, self.num_days):
                    
                s = self.sales[day:day+self.window, :].reshape(1, self.window, -1)
                p = self.prices[day:day+self.window, :]                    .reshape(1, self.window, -1)

                X = np.concatenate((s,p),axis=0)

                for span in self.spans:
                    
                    span_ = day if day < span else span

                    ma = moving_average(self.sales[day-span_:day+self.window, :]
                                        , n=span)[span_:, :]\
                        .reshape(1, self.window, -1)

                    X = np.concatenate((X, ma),axis=0)

                ## transposing (features, days, items) into (items, days, features)
                X = np.transpose(X, (2,1,0)) 

                if not self.infer:
                    y = self.sales[day+self.window, :].reshape(-1,1) 
                    
                for i in range(self.steps_per_day):
                    
                    ## if the batch go over the maxium item number, 
                    ## the batch_size will be truncated
                    if (i+1)*self.batch_size > self.num_items:
                        end = self.num_items
                    else:
                        end = (i+1)*self.batch_size
                    
                    ## categories has (items, features) shape
                    ## only relevant item rows are fetched
                    cat = self.categories[i*self.batch_size:end, :]
                    state_id = cat[:, -1]
                    # reshaping into (features, items, 1)
                    cat = cat.T.reshape(cat.shape[1], cat.shape[0], 1)
                    
                    ## calender values are taken at prediction target date
                    calen = self.calendar[day+self.window,:7].reshape(1,-1)
                    calen = np.repeat(calen, end-i*self.batch_size, axis=0)
                    calen = calen.T.reshape(calen.shape[1], calen.shape[0], 1)
                    
                    ## snap values are taken at prediction target date
                    snap = self.calendar[day+self.window,7:].reshape(1,-1)
                    snap = np.repeat(snap, end-i*self.batch_size, axis=0)
                    # taking only relevant state's snap values for each row
                    snap = snap[np.arange(len(snap)), state_id].reshape(-1,1)
                    
                    if self.infer:
                        yield [X[i*self.batch_size: end]] + [j for j in cat]                                + [j for j in calen] + [snap]
                    else:
                        yield [X[i*self.batch_size: end]] + [j for j in cat]                               + [j for j in calen] + [snap],                              y[i*self.batch_size: end]


# ## Defining Model
# 
# * Sales and price sequences are put into LSTM
# * Categorical features are going into Embedding layer
# * Snap indicator is put directly into Dense layer

# In[ ]:


def define_model(lstm_w_size, lstm_n_fts):
    
    ## Categorical embedding
    cat_inputs = []
    for cat in cat_ft1+cat_ft2:
        cat_inputs.append(Input(shape=[1], name=cat))
        
    cat_embeddings = []
    for i, cat in enumerate(cat_ft1+cat_ft2):
        cat_embeddings.append(Embedding(category_counts[cat], 
                                        min(50, int(category_counts[cat]+1/ 2)), 
                                        name = cat + "_embed")(cat_inputs[i]))

    cat_output = Concatenate()([Flatten()(cat_emb)                                           for cat_emb in cat_embeddings])
    cat_output = Dropout(.7)(cat_output)
    
    # snap input
    snap_input = Input(shape=[1])

    ## LSTM
    lstm_input = Input(shape=(lstm_w_size, lstm_n_fts))
    lstm_output = CuDNNLSTM(32)(lstm_input)
    
    concat = Concatenate()([
        lstm_output,
        cat_output,
        snap_input
    ])
        
    dense_output = Dense(10, activation='relu')(concat)
    out = Dense(1)(dense_output)
    model = Model(inputs=[lstm_input] + cat_inputs + [snap_input],
                  outputs=out)

    model.compile(optimizer='adam', loss='mse')
    
    return model


# ## Model Training - Cross Validation

# In[ ]:


def model_training(inputs, cv, w_size=30, batch_size=32, epochs=10,
                   early_stopping=10, plt_iter=True):

    val_scores=[]
    train_evals=[]
    valid_evals=[]
    best_epoch=[]

    for idx, (train_index, val_index) in enumerate(cv.split(inputs[0])):
        
        if idx >= 2: # skipping the first 2 fold to save run time

            #print("###### fold %d ######" % (idx+1))
            sales_train, sales_val = inputs[0][train_index, :],                                     inputs[0][val_index, :]
            prices_train, prices_val = inputs[1][train_index, :],                                       inputs[1][val_index, :]
            calendar_train, calendar_val = inputs[3][train_index, :],                                           inputs[3][val_index, :]
            inputs_train = [sales_train, prices_train, inputs[2], calendar_train]
            inputs_val = [sales_val, prices_val, inputs[2], calendar_val]

            train_gen = SequenceGenerator(inputs_train, spans=span_lst,
                                          window=w_size, batch_size=batch_size)
            val_gen = SequenceGenerator(inputs_val, spans=span_lst, window=w_size,
                                        batch_size=batch_size)

            model = define_model(w_size, 2+len(span_lst))
            early_stop = EarlyStopping(patience=early_stopping,
                                       verbose=True,
                                       restore_best_weights=True)

            hist = model.fit_generator(train_gen.generate(),
                      validation_data=val_gen.generate(),
                      epochs=epochs,
                      steps_per_epoch=train_gen.steps, 
                      validation_steps=val_gen.steps, 
                      callbacks=[early_stop],
                      verbose=0)

            val_scores.append(np.min(hist.history['val_loss']))
            train_evals.append(hist.history['loss'])
            valid_evals.append(hist.history['val_loss'])

            best_epoch.append(np.argmin(hist.history['val_loss']) + 1)
    
    print('### CV scores by fold ###')
    for i in range(2, cv.get_n_splits(sales)):
        print(f'fold {i+1}: {val_scores[i-2]:.4f} at epoch {best_epoch[i-2]}')
    print('CV mean score: {0:.4f}, std: {1:.4f}'          .format(np.mean(val_scores), np.std(val_scores)))
    
    if plt_iter:
        
        fig, axs = plt.subplots(1, 2, figsize=(11,4))
        
        for i, ax in enumerate(axs.flatten()):
            if i < cv.get_n_splits(sales):
                ax.plot(train_evals[i], label='training')
                ax.plot(valid_evals[i], label='validation')
                ax.set(xlabel='epoch', ylabel='loss')
                ax.set_title(f'fold {i+1+2}', fontsize=12)
                ax.legend(loc='upper right', prop={'size': 9})
                          
        fig.tight_layout()
        plt.show()

    return best_epoch


# In[ ]:


# %%time

sales = sales.iloc[:,1:].values
prices = prices.iloc[:,1:].values
categories = categories.values
calendar = calendar.values
inputs = [sales, prices, categories, calendar]

if CV:
    cv = TimeSeriesSplit(n_splits=4)
    best_epoch = model_training(inputs, cv, w_size=w_size, 
                                batch_size=batch_size, 
                                epochs=epochs, early_stopping=5, 
                                plt_iter=True)


# ## Model Training without CV split
# This model will be used to make predictions for the submission file

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_gen = SequenceGenerator(inputs, spans=span_lst, window=w_size,\n                              batch_size=batch_size)\nmodel = define_model(w_size, 2+len(span_lst))\nhist = model.fit_generator(train_gen.generate(),\n                           epochs=best_epoch[-1] if CV else epochs,\n                           steps_per_epoch=train_gen.steps,\n                           verbose=0)')


# ## Predictions for Submission
# * Predicting day by day by looping through 28 days
# * Each prediction will be used for the model input for the next day prediction

# In[ ]:


# subsetting len-w_size-90: as we need the first 90 days 
# prior to the LSTM window to calculate 90 days moving avarage
sales_test = sales[sales.shape[0]-w_size-90:, :]
prices_test = prices[sales.shape[0]-w_size-90:, :]
calendar_test = calendar[sales.shape[0]-w_size-90:, :]
test_inputs = [sales_test, prices_test, categories, calendar_test]

for i in range(28):
    
    test_gen = SequenceGenerator(test_inputs, spans=span_lst, 
                                 window=w_size, infer=True)
    test_iter = test_gen.generate()
    X = next(test_iter)
    y_pred = model.predict(X)

    # appending predicted sales to the input and shifting it by 1
    sales_test = np.append(sales_test, y_pred.reshape(1,-1), axis=0)[1:, :]
    prices_test = prices_test[1:, :]
    calendar_test = calendar_test[1:, :]
    test_inputs = [sales_test, prices_test, categories, calendar_test]


# scaling predicted values back into original values

# In[ ]:


sales_test = np.exp((sales_test * sales_std) + sales_mean) - 1
sales_test = np.maximum(sales_test, np.zeros(sales_test.shape))


# ## Making Submission
# We are only creating "validation" (corresponding to the Public leaderboard) submission. The submission for "evaluation" (corresponding to the Private leaderboard) is out-of-scope for now.

# In[ ]:


submission = pd.read_csv(DATA_DIR + 'sample_submission.csv', nrows=rows)

if DEBUG:
    submission.iloc[:100, 1:] = sales_test[-28:, :].T
else:
    submission.iloc[:len(submission)//2, 1:] = sales_test[-28:, :].T
    
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:




