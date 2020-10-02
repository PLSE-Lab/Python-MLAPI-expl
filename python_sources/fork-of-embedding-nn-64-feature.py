#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#an attempt to use embedding for tabular data
# thanks to https://www.kaggle.com/artgor/nn-baseline, https://www.kaggle.com/hrmello/starter-neural-network-3-939-lb
# and kaggle kernel on feature engineering like https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo and others


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_tqdm import TQDMNotebookCallback

import os

import tensorflow as tf
import keras as K
print(tf.__version__)
print(K.__version__)
print(tf.keras.__version__)

SEED = 2018


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import Ridge

from sklearn import preprocessing
import warnings
import datetime
warnings.filterwarnings("ignore")
import gc

from scipy.stats import describe
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# In[ ]:


#Loading Train and Test Data
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))
print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))


# In[ ]:


train["month"] = train["first_active_month"].dt.month
test["month"] = test["first_active_month"].dt.month
train["year"] = train["first_active_month"].dt.year
test["year"] = test["first_active_month"].dt.year
train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
train.head()


# In[ ]:


hist_trans = pd.read_csv("../input/historical_transactions.csv")
hist_trans.head()


# In[ ]:


hist_trans = pd.get_dummies(hist_trans, columns=['category_2', 'category_3'])
hist_trans['authorized_flag'] = hist_trans['authorized_flag'].map({'Y': 1, 'N': 0})
hist_trans['category_1'] = hist_trans['category_1'].map({'Y': 1, 'N': 0})
hist_trans.head()


# In[ ]:


def aggregate_transactions(trans, prefix):  
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
    }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans


# In[ ]:


import gc
merch_hist = aggregate_transactions(hist_trans, prefix='hist_')
del hist_trans
gc.collect()
train = pd.merge(train, merch_hist, on='card_id',how='left')
test = pd.merge(test, merch_hist, on='card_id',how='left')
del merch_hist
gc.collect()
train.head()


# In[ ]:


new_trans = pd.read_csv("../input/new_merchant_transactions.csv")
new_trans.head()


# In[ ]:


new_trans = pd.get_dummies(new_trans, columns=['category_2', 'category_3'])
new_trans['authorized_flag'] = new_trans['authorized_flag'].map({'Y': 1, 'N': 0})
new_trans['category_1'] = new_trans['category_1'].map({'Y': 1, 'N': 0})
new_trans.head()


# In[ ]:


merch_new = aggregate_transactions(new_trans, prefix='new_')
del new_trans
gc.collect()
train = pd.merge(train, merch_new, on='card_id',how='left')
test = pd.merge(test, merch_new, on='card_id',how='left')

train.head()


# In[ ]:


del merch_new
gc.collect()


# In[ ]:


cat_cols = ['feature_1', 'feature_2', 'feature_3', 'month', 'year','hist_merchant_id_nunique']


# In[ ]:


target = train['target']
drops = ['card_id', 'first_active_month', 'target']
num_cols = [col for col in train.columns if col not in cat_cols and col not in drops]
total_cols = [col for col in train.columns]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
max_values = {}
for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))
    max_values[col] = max(train[col].max(), test[col].max())  + 2
    print(max_values[col])


# In[ ]:


# printing because I'm too lazy to write everything by hand. Open output to see.
for col in cat_cols:
    n = col.replace('.', '_')
    print(f'{n} = Input(shape=[1], name="{col}")')
    print(f'emb_{n} = Embedding({max_values[col]}, {(np.min(max_values[col]+1)//2, 50)})({col})')
    print(',', n)


# In[ ]:


#filter num values
target = train['target']
drops = ['card_id', 'first_active_month', 'target']
use_cols_num = [c for c in train.columns if c in num_cols]
features_num = list(train[use_cols_num].columns)
train[features_num].head()


# In[ ]:


#scale the data and impute the null values 
#note: apparently, GPU environment doesn't have an updated version of sklearn,
#so we cannot use sklearn.impute.SimpleImputer. In CPU environement this is possible
from sklearn.preprocessing import StandardScaler, Imputer
sc = StandardScaler()
train[features_num] = train[features_num].fillna(0)
train[features_num] = sc.fit_transform(train[features_num])


# In[ ]:


#fit test data
test[features_num] = test[features_num].fillna(0)
test[features_num] = sc.fit_transform(test[features_num])


# In[ ]:


#drop columns
X_train = train.drop([col for col in drops if col in train.columns], axis=1)
X_test = test.drop([col for col in drops if col in test.columns], axis=1)
X_train.shape


# In[ ]:


#renommer certaines colonnes
X_train.rename(columns={'hist_category_2_1.0_mean': 'hist_category_2_1_0_mean',
                        'hist_category_2_2.0_mean': 'hist_category_2_2_0_mean',
                        'hist_category_2_3.0_mean': 'hist_category_2_3_0_mean',
                        'hist_category_2_4.0_mean': 'hist_category_2_4_0_mean',
                        'hist_category_2_5.0_mean': 'hist_category_2_5_0_mean',
                        'new_category_2_1.0_mean': 'new_category_2_1_0_mean',
                        'new_category_2_2.0_mean': 'new_category_2_2_0_mean',
                        'new_category_2_3.0_mean': 'new_category_2_3_0_mean',
                        'new_category_2_4.0_mean': 'new_category_2_4_0_mean',
                        'new_category_2_5.0_mean': 'new_category_2_5_0_mean'}, inplace=True)
total_cols = [col for col in X_train.columns]
total_cols


# In[ ]:


#renommer certaines colonnes
X_test.rename(columns={'hist_category_2_1.0_mean': 'hist_category_2_1_0_mean',
                        'hist_category_2_2.0_mean': 'hist_category_2_2_0_mean',
                        'hist_category_2_3.0_mean': 'hist_category_2_3_0_mean',
                        'hist_category_2_4.0_mean': 'hist_category_2_4_0_mean',
                        'hist_category_2_5.0_mean': 'hist_category_2_5_0_mean',
                        'new_category_2_1.0_mean': 'new_category_2_1_0_mean',
                        'new_category_2_2.0_mean': 'new_category_2_2_0_mean',
                        'new_category_2_3.0_mean': 'new_category_2_3_0_mean',
                        'new_category_2_4.0_mean': 'new_category_2_4_0_mean',
                        'new_category_2_5.0_mean': 'new_category_2_5_0_mean'}, inplace=True)
total_cols = [col for col in X_test.columns]
total_cols


# In[ ]:


num_cols = [col for col in X_train.columns if col not in cat_cols and col not in drops]
num_cols


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split
train_y = target.values
x_train, x_val, y_train, y_val = train_test_split(X_train, train_y, test_size = .1, random_state = SEED)


# In[ ]:


from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error as mse_loss

from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#building the network

from keras.initializers import he_normal, he_uniform,  glorot_normal,  glorot_uniform
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D


# In[ ]:


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


# In[ ]:


def model(dense_dim_1=128, dense_dim_2=64, dense_dim_3=256, dense_dim_4=512,  dense_dim_5=512, dense_dim_6=256,
dropout1=0.1, dropout2=0.1, dropout3=0.2, dropout4=0.2, dropout5=0.2, dropout6=0.2, lr=0.0001):

    #Inputs cat
    feature_1 = Input(shape=[1], name="feature_1")
    feature_2 = Input(shape=[1], name="feature_2")
    feature_3 = Input(shape=[1], name="feature_3")
    month = Input(shape=[1], name="month")
    year = Input(shape=[1], name="year")
    hist_merchant_id_nunique = Input(shape=[1], name="hist_merchant_id_nunique")
    
    #Input num    
    elapsed_time = Input(shape=[1], name="elapsed_time")
    hist_transactions_count = Input(shape=[1], name="hist_transactions_count")
    hist_authorized_flag_sum = Input(shape=[1], name="hist_authorized_flag_sum")
    hist_authorized_flag_mean = Input(shape=[1], name="hist_authorized_flag_mean")
    hist_category_1_mean = Input(shape=[1], name="hist_category_1_mean")
    hist_category_2_1_0_mean = Input(shape=[1], name="hist_category_2_1_0_mean")
    hist_category_2_2_0_mean = Input(shape=[1], name="hist_category_2_2_0_mean")
    hist_category_2_3_0_mean = Input(shape=[1], name="hist_category_2_3_0_mean")
    hist_category_2_4_0_mean = Input(shape=[1], name="hist_category_2_4_0_mean")
    hist_category_2_5_0_mean = Input(shape=[1], name="hist_category_2_5_0_mean")
    hist_category_3_A_mean = Input(shape=[1], name="hist_category_3_A_mean")
    hist_category_3_B_mean = Input(shape=[1], name="hist_category_3_B_mean")
    hist_category_3_C_mean = Input(shape=[1], name="hist_category_3_C_mean")
    hist_purchase_amount_sum = Input(shape=[1], name="hist_purchase_amount_sum")
    hist_purchase_amount_mean = Input(shape=[1], name="hist_purchase_amount_mean")
    hist_purchase_amount_max = Input(shape=[1], name="hist_purchase_amount_max")
    hist_purchase_amount_min = Input(shape=[1], name="hist_purchase_amount_min")
    hist_purchase_amount_std = Input(shape=[1], name="hist_purchase_amount_std")
    hist_installments_sum = Input(shape=[1], name="hist_installments_sum")
    hist_installments_mean = Input(shape=[1], name="hist_installments_mean")
    hist_installments_max = Input(shape=[1], name="hist_installments_max")
    hist_installments_min = Input(shape=[1], name="hist_installments_min")
    hist_installments_std = Input(shape=[1], name="hist_installments_std")
    hist_purchase_date_ptp = Input(shape=[1], name="hist_purchase_date_ptp")
    hist_month_lag_min = Input(shape=[1], name="hist_month_lag_min")
    hist_month_lag_max = Input(shape=[1], name="hist_month_lag_max")
    
    new_transactions_count = Input(shape=[1], name="new_transactions_count")
    new_authorized_flag_sum = Input(shape=[1], name="new_authorized_flag_sum")
    new_authorized_flag_mean = Input(shape=[1], name="new_authorized_flag_mean")
    new_category_1_mean = Input(shape=[1], name="new_category_1_mean")
    new_category_2_1_0_mean = Input(shape=[1], name="new_category_2_1_0_mean")
    new_category_2_2_0_mean = Input(shape=[1], name="new_category_2_2_0_mean")
    new_category_2_3_0_mean = Input(shape=[1], name="new_category_2_3_0_mean")
    new_category_2_4_0_mean = Input(shape=[1], name="new_category_2_4_0_mean")
    new_category_2_5_0_mean = Input(shape=[1], name="new_category_2_5_0_mean")
    new_category_3_A_mean = Input(shape=[1], name="new_category_3_A_mean")
    new_category_3_B_mean = Input(shape=[1], name="new_category_3_B_mean")
    new_category_3_C_mean = Input(shape=[1], name="new_category_3_C_mean")
    new_purchase_amount_sum = Input(shape=[1], name="new_purchase_amount_sum")
    new_purchase_amount_mean = Input(shape=[1], name="new_purchase_amount_mean")
    new_purchase_amount_max = Input(shape=[1], name="new_purchase_amount_max")
    new_purchase_amount_min = Input(shape=[1], name="new_purchase_amount_min")
    new_purchase_amount_std = Input(shape=[1], name="new_purchase_amount_std")
    new_installments_sum = Input(shape=[1], name="new_installments_sum")
    new_installments_mean = Input(shape=[1], name="new_installments_mean")
    new_installments_max = Input(shape=[1], name="new_installments_max")
    new_installments_min = Input(shape=[1], name="new_installments_min")
    new_installments_std = Input(shape=[1], name="new_installments_std")
    new_purchase_date_ptp = Input(shape=[1], name="new_purchase_date_ptp")
    new_month_lag_min = Input(shape=[1], name="new_month_lag_min")
    new_month_lag_max = Input(shape=[1], name="new_month_lag_max")    
    
    #Embeddings layers

    emb_feature_1 = Embedding(6, 3)(feature_1)
    emb_feature_2 = Embedding(4, 3)(feature_2)
    emb_feature_3 = Embedding(4, 3)(feature_3)
    emb_month = Embedding(26, 13)(month)
    emb_year = Embedding(18, 9)(year)
    emb_hist_merchant_id_nunique = Embedding(333, 50)(hist_merchant_id_nunique)


    concat_emb1 = concatenate([
           Flatten() (emb_feature_1),
            Flatten() (emb_feature_2),
            Flatten() (emb_feature_3),
            Flatten() (emb_month),
            Flatten() (emb_year),
            Flatten() (emb_hist_merchant_id_nunique)
    ])
    
    categ = Dropout(dropout1)(Dense(dense_dim_1,kernel_initializer=he_uniform(seed=SEED),activation='relu') (concat_emb1))
    categ = BatchNormalization()(categ)
    categ = Dropout(dropout2)(Dense(dense_dim_2,kernel_initializer=he_uniform(seed=SEED),activation='relu') (categ))
        
    #main layer
    main_l = concatenate([
          categ
        , elapsed_time
        , hist_transactions_count
        , hist_authorized_flag_sum
        , hist_authorized_flag_mean
        , hist_category_1_mean
        , hist_category_2_1_0_mean
        , hist_category_2_2_0_mean
        , hist_category_2_3_0_mean
        , hist_category_2_4_0_mean 
        , hist_category_2_5_0_mean
        , hist_category_3_A_mean
        , hist_category_3_B_mean
        , hist_category_3_C_mean
        , hist_purchase_amount_sum
        , hist_purchase_amount_mean
        , hist_purchase_amount_max
        , hist_purchase_amount_min
        , hist_purchase_amount_std
        , hist_installments_sum
        , hist_installments_mean
        , hist_installments_max
        , hist_installments_min
        , hist_installments_std 
        , hist_purchase_date_ptp
        , hist_month_lag_min
        , hist_month_lag_max
        , new_transactions_count
        , new_authorized_flag_sum
        , new_authorized_flag_mean
        , new_category_1_mean
        , new_category_2_1_0_mean
        , new_category_2_2_0_mean
        , new_category_2_3_0_mean
        , new_category_2_4_0_mean 
        , new_category_2_5_0_mean
        , new_category_3_A_mean
        , new_category_3_B_mean
        , new_category_3_C_mean
        , new_purchase_amount_sum
        , new_purchase_amount_mean
        , new_purchase_amount_max
        , new_purchase_amount_min
        , new_purchase_amount_std
        , new_installments_sum
        , new_installments_mean
        , new_installments_max
        , new_installments_min
        , new_installments_std 
        , new_purchase_date_ptp
        , new_month_lag_min
        , new_month_lag_max
    ])
    
    main_l = Dropout(dropout3)(Dense(dense_dim_3,kernel_initializer=he_uniform(seed=SEED),activation='relu') (main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dropout4)(Dense(dense_dim_4,kernel_initializer=he_uniform(seed=SEED),activation='relu') (main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dropout5)(Dense(dense_dim_5,kernel_initializer=he_uniform(seed=SEED),activation='relu') (main_l))
    main_l = BatchNormalization()(main_l) 
    main_l = Dropout(dropout6)(Dense(dense_dim_6,kernel_initializer=he_uniform(seed=SEED),activation='relu') (main_l))

    
    #output
    output = Dense(1) (main_l)

    model = Model([feature_1,
                    feature_2,
                    feature_3,
                    month,
                    year,
                    hist_merchant_id_nunique,
                    elapsed_time,
                    hist_transactions_count,
                    hist_authorized_flag_sum,
                    hist_authorized_flag_mean,
                    hist_category_1_mean,
                    hist_category_2_1_0_mean,
                    hist_category_2_2_0_mean,
                    hist_category_2_3_0_mean,
                    hist_category_2_4_0_mean, 
                    hist_category_2_5_0_mean,
                    hist_category_3_A_mean,
                    hist_category_3_B_mean,
                    hist_category_3_C_mean,
                    hist_purchase_amount_sum,
                    hist_purchase_amount_mean,
                    hist_purchase_amount_max,
                    hist_purchase_amount_min,
                    hist_purchase_amount_std,
                    hist_installments_sum,
                    hist_installments_mean,
                    hist_installments_max,
                    hist_installments_min,
                    hist_installments_std, 
                    hist_purchase_date_ptp,
                    hist_month_lag_min,
                    hist_month_lag_max,
                    new_transactions_count,
                    new_authorized_flag_sum,
                    new_authorized_flag_mean,
                    new_category_1_mean,
                    new_category_2_1_0_mean,
                    new_category_2_2_0_mean,
                    new_category_2_3_0_mean,
                    new_category_2_4_0_mean, 
                    new_category_2_5_0_mean,
                    new_category_3_A_mean,
                    new_category_3_B_mean,
                    new_category_3_C_mean,
                    new_purchase_amount_sum,
                    new_purchase_amount_mean,
                    new_purchase_amount_max,
                    new_purchase_amount_min,
                    new_purchase_amount_std,
                    new_installments_sum,
                    new_installments_mean,
                    new_installments_max,
                    new_installments_min,
                    new_installments_std, 
                    new_purchase_date_ptp,
                    new_month_lag_min,
                    new_month_lag_max], output)

    #model = Model([**params], output)
    model.compile(optimizer = Adam(lr=lr),
                  loss= rmse,
                  metrics=[rmse])
    return model


#model=model()


# In[ ]:


# converting data to format which can be used by Keras
def get_keras_data(df, num_cols, cat_cols):
    cols = num_cols + cat_cols

    X = {col: np.array(df[col]) for col in cols}
    # print("Data ready for Vectorization")
    
    return X


# In[ ]:


X_test_keras = get_keras_data(X_test, num_cols, cat_cols)


# In[ ]:


def train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, reduce_lr=True, patience=7):
    """
    Helper function to train model. Also I noticed that ReduceLROnPlateau is rarely
    useful, so added an option to turn it off.
    """
    
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    model_checkpoint = ModelCheckpoint("model.hdf5",
                                       save_best_only=True, verbose=1, monitor='val_loss', mode='min')
    if reduce_lr:
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.000005, verbose=1)
        hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_v, y_valid), verbose=True,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr])
    
    else:
        hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_v, y_valid), verbose=True,
                            callbacks=[early_stopping, model_checkpoint])

    
    return hist


# In[ ]:


X_t = get_keras_data(x_train, num_cols, cat_cols)
X_v = get_keras_data(x_val, num_cols, cat_cols)


# In[ ]:


keras_model = model()


# In[ ]:


keras_model.summary()


# In[ ]:


from keras.utils import plot_model
plot_model(keras_model, to_file='model.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(keras_model).create(prog='dot', format='svg'))


# In[ ]:


batch_size = 64
epochs = 100
hist = train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_val, reduce_lr=True, patience=7)


# In[ ]:


import matplotlib.pyplot as plt
#plotting training and validations losses
plt.plot(hist.history['val_loss'], label = "val_loss")
plt.plot(hist.history['loss'], label = "loss")
plt.legend()


# In[ ]:


keras_modelmax = load_model("model.hdf5", custom_objects={'rmse': rmse})


# In[ ]:


y_pred_valid = keras_modelmax.predict(X_v)
valid_score = mean_squared_error(y_val,y_pred_valid)** 0.5
print("Validation score: ", valid_score)


# In[ ]:


predictions = keras_modelmax.predict(X_test_keras).reshape(-1,)


# In[ ]:


#saving the card_ids
ids = test['card_id'].values
submission = pd.DataFrame(ids, columns=['card_id'])


# In[ ]:


submission['target'] = predictions
submission.head(10)


# In[ ]:


submission.to_csv("submission_neuralnet.csv", index = False, header = True)


# In[ ]:


print("done")

