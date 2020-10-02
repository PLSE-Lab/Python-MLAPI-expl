#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import string
import numpy as np
import pandas as pd
from pandasql import sqldf

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding
from keras.initializers import RandomNormal, Constant
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import seaborn as sns
import warnings

from math import sqrt

import itertools
from tqdm import tqdm

np.random.seed(42)  # for reproducibility

sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

pd.set_option('display.max_columns', 60)

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ## Analisys Exploratory

# In[ ]:


def concat_data():
    df_train = pd.read_csv('../input/dataset_treino.csv')
    df_test = pd.read_csv('../input/dataset_teste.csv')
    df_extra = pd.read_csv('../input/lojas.csv')
    df_test['Sales'] = -1
    df_full = pd.concat([df_train, df_test]).reset_index(drop=True)

    #Merge extra information about stores
    df_full = df_full.merge(df_extra, left_on=['Store'], right_on=['Store'], how='left')
    
    df_full['Year'] = pd.DatetimeIndex(df_full['Date']).year
    df_full['Month'] = pd.DatetimeIndex(df_full['Date']).month
    df_full['Day'] = pd.DatetimeIndex(df_full['Date']).day
    df_full['WeekOfYear'] = pd.DatetimeIndex(df_full['Date']).weekofyear
    
    # Calculate competition open in months
    df_full['CompetitionOpen'] = 12 * (df_full.Year - df_full.CompetitionOpenSinceYear) +         (df_full.Month - df_full.CompetitionOpenSinceMonth)

    # Calculate promo open time in months
    df_full['PromoOpen'] = 12 * (df_full.Year - df_full.Promo2SinceYear) +         (df_full.WeekOfYear - df_full.Promo2SinceWeek) / 4.0
    df_full['PromoOpen'] = df_full.PromoOpen.apply(lambda x: x if x > 0 else 0)
    df_full.loc[df_full.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Transform month interval in a boolean column 
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df_full['monthStr'] = df_full.Month.map(month2str)
    df_full.loc[df_full.PromoInterval == 0, 'PromoInterval'] = ''
    df_full['IsPromoMonth'] = 0
    for interval in df_full.PromoInterval.unique():
        interval = str(interval)
        if interval != '':
            for month in interval.split(','):
                df_full.loc[(df_full.monthStr == month) & (df_full.PromoInterval == interval), 'IsPromoMonth'] = 1


    return df_full

df_full = concat_data()


# In[ ]:


def extrat_test_data(df_full):
    df_train = df_full.loc[df_full['Sales'] != -1]
    df_test = df_full.loc[df_full['Sales'] == -1]

    return df_train, df_test

df_train, df_test = extrat_test_data(df_full)


# In[ ]:


df_full.head()


# **Missing Values**

# In[ ]:


df_full.info()


# In[ ]:


# Function to calculate missing values by column (By DSA)
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

missing_values_table(df_full)


# In[ ]:


df_full.groupby('StoreType')['Sales'].describe()


# In[ ]:


df_full.groupby('StoreType')['Customers', 'Sales'].sum()


# In[ ]:


# Plotting correlations
num_feat=df_full.columns[df_full.dtypes!=object]
num_feat=num_feat[1:-1] 
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(df_full[col].values, df_full['Sales'].values)[0,1])
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(10,15))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t Sales")


# In[ ]:


# Heatmap of correlations features
corrMatrix=df_full[["Sales", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "CompetitionDistance",
                    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
                    "Promo2SinceWeek", "Promo2SinceYear", "Year", "Month", "Day",
                    "CompetitionOpen", "PromoOpen", "IsPromoMonth", "Store"]].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(30, 30))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features')


# ## Functions to create and train the model

# In[13]:


def clean_data(use_text_columns = True):
    '''
    Function that clean data and create a new features to enrich the model
    '''
    cols_num = ["Sales", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "CompetitionDistance",
                "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
                "Promo2SinceWeek", "Promo2SinceYear", "Wapp", "Avg_Customers", "Year", "Month", "Day",
                "CompetitionOpen", "PromoOpen", "IsPromoMonth", "Store"]

    cols_text = ["StateHoliday", "StoreType", "Assortment"]

    df_train = pd.read_csv('../input/dataset_treino.csv')    
    len_train_data = len(df_train)

    df_test = pd.read_csv('../input/dataset_teste.csv')

    # Setting null values of column Open in test dataset
    df_test.loc[df_test['DayOfWeek'] != 7, 'Open'] = 1
    df_test.loc[df_test['DayOfWeek'] == 7, 'Open'] = 0

    avg_customer = sqldf(
      """
      SELECT
      Store,
      DayOfWeek,
      sum(case when Customers is not null then Sales/Customers else 0 end) as Wapp,
      round(avg(Customers)) Avg_Customers
      from df_train
      group by Store,DayOfWeek
      """
    )
    
    df_test = sqldf(
      """
      SELECT
      t.*,
      ac.Wapp,
      ac.Avg_Customers
      from df_test t
      left join avg_customer ac on t.Store = ac.Store and t.DayOfWeek = ac.DayOfWeek
      """
    )
    
    df_train = sqldf(
      """
      SELECT
      t.*,
      ac.Wapp,
      ac.Avg_Customers
      from df_train t
      left join avg_customer ac on t.Store = ac.Store and t.DayOfWeek = ac.DayOfWeek
      """
    )

    # Merge train and test dataset
    all_data = pd.concat([df_train, df_test], ignore_index=True)

    df_extra = pd.read_csv('../input/lojas.csv')
    df_full = pd.concat([df_train, df_test]).reset_index(drop=True)

    # Merge extra information about stores
    all_data = df_full.merge(df_extra, left_on=['Store'], right_on=['Store'], how='left')

    # Separate date in Year, Month and Day
    all_data.loc[all_data['StateHoliday'] == 0, 'StateHoliday'] = 'd'
    all_data['Year'] = pd.DatetimeIndex(all_data['Date']).year
    all_data['Month'] = pd.DatetimeIndex(all_data['Date']).month
    all_data['Day'] = pd.DatetimeIndex(all_data['Date']).day
    all_data['WeekOfYear'] = pd.DatetimeIndex(all_data['Date']).weekofyear

    # Calculate competition open in months
    all_data['CompetitionOpen'] = 12 * (all_data.Year - all_data.CompetitionOpenSinceYear) +         (all_data.Month - all_data.CompetitionOpenSinceMonth)

    # Calculate promo open time in months
    all_data['PromoOpen'] = 12 * (all_data.Year - all_data.Promo2SinceYear) +         (all_data.WeekOfYear - all_data.Promo2SinceWeek) / 4.0
    all_data['PromoOpen'] = all_data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    all_data.loc[all_data.Promo2SinceYear == 0, 'PromoOpen'] = 0
    
    # Transform month interval in a boolean column 
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    all_data['monthStr'] = all_data.Month.map(month2str)
    all_data.loc[all_data.PromoInterval == 0, 'PromoInterval'] = ''
    all_data['IsPromoMonth'] = 0
    for interval in all_data.PromoInterval.unique():
        interval = str(interval)
        if interval != '':
            for month in interval.split(','):
                all_data.loc[(all_data.monthStr == month) & (all_data.PromoInterval == interval), 'IsPromoMonth'] = 1

    data_numeric = all_data[cols_num]
    
    # Fill NAN values
    # Only column CompetitionDistance is fill NaN with a median value
    data_numeric['CompetitionDistance'].fillna(data_numeric['CompetitionDistance'].median(), inplace = True)

    # Other values is fill with zero
    data_numeric.fillna(0, inplace = True)

    if (use_text_columns):
        data_text = all_data[cols_text]
        data_text = pd.get_dummies(data_text, dummy_na=False)

        complete_data = pd.concat([data_numeric, data_text], axis = 1)

        df_train = complete_data.iloc[:len_train_data,:]
        df_test = complete_data.iloc[len_train_data:,:]
    else:
        df_train = data_numeric.iloc[:len_train_data,:]
        df_test = data_numeric.iloc[len_train_data:,:]

    return df_train, df_test


# In[4]:


def load_train_data(scaler_x, scaler_y):
    '''
    Transform train data set and separate a test dataset to validate the model in the end of training and normalize data
    '''
    X_train = train.drop(["Sales"], axis=1) # Features
    y_train = np.array(train["Sales"]).reshape((len(X_train), 1)) # Targets
    X_train = scaler_x.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    return (X_train, y_train), (X_test, y_test)


# In[5]:


def load_test_data():
    '''
    Remove column of predictions and normalize data of submission test data set.
    '''
    X_test = test.drop(["Sales"], axis=1) # Features
    X_test = StandardScaler().fit_transform(X_test)

    return X_test


# In[6]:


# Show info of model
def show_info(model, X, y, log, weights = None):
    '''
    Show metrics about the evaluation model and plots about loss, rmse and rmspe
    '''
    if (log != None):
        # summarize history for loss
        plt.figure(figsize=(14,10))
        plt.plot(log.history['loss'])
        plt.plot(log.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        print('\n')
        
        # summarize history for rmse
        plt.figure(figsize=(14,10))
        plt.plot(log.history['rmse'])
        plt.plot(log.history['val_rmse'])
        plt.title('Model RMSE')
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        print('\n')
        
        # summarize history for rmspe
        plt.figure(figsize=(14,10))
        plt.plot(log.history['rmspe'])
        plt.plot(log.history['val_rmspe'])
        plt.title('Model RMSPE')
        plt.ylabel('rmspe')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    if (weights != None):
        model.load_weights(weights)

    predictions = model.predict(X, verbose=1)

    mse = mean_squared_error(y, predictions)
    rmse = sqrt(mse)
    rmspe = rmspe_val(y, predictions)

    print('MSE: %.3f' % mse)
    print('RMSE: %.3f' % rmse)
    print('RMSPE: %.3f' % rmspe)


# ## RMSPE Formula
# $\textrm{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}$

# In[7]:


def rmspe_val(y_true, y_pred):
    '''
    RMSPE calculus to validate evaluation metric about the model
    '''
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true), axis=0))[0]


# In[8]:


def rmspe(y_true, y_pred):
    '''
    RMSPE calculus to use during training phase
    '''
    return K.sqrt(K.mean(K.square((y_true - y_pred) / y_true), axis=-1))


# In[9]:


def rmse(y_true, y_pred):
    '''
    RMSE calculus to use during training phase
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[10]:


def create_model():
    '''
    Create a neural network
    '''
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

    model = Sequential()
    model.add(Dense(512, input_dim=X_train.shape[1], activation="relu", kernel_initializer=initializer))
    model.add(Dropout(0.4))
    model.add(Dense(512, input_dim=X_train.shape[1], activation="relu", kernel_initializer=initializer))
    model.add(Dropout(0.4))
    model.add(Dense(512, input_dim=X_train.shape[1], activation="relu", kernel_initializer=initializer))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="linear", kernel_initializer=initializer))
    adam = Adam(lr=1e-3, decay=1e-3)

    # Compile model
    model.compile(loss="mean_squared_error", optimizer=adam, metrics=[rmse, rmspe])

    return model


# In[14]:


train, test = clean_data(use_text_columns = True)


# In[15]:


train.head()


# In[16]:


# Hyperparameters and load data to train the model
batch_size = 512
nb_epoch = 300

scaler_x = StandardScaler()
scaler_y = StandardScaler()

print('Loading data...')
(X_train, y_train), (X_test, y_test) = load_train_data(scaler_x, scaler_y)

print('Build model...')
model = create_model()
model.summary()


# In[17]:


print('Fit model...')
filepath="weights_solar.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
callbacks_list = [checkpoint, early_stopping]

log = model.fit(X_train, y_train,
          validation_split=0.20, batch_size=batch_size, epochs=nb_epoch, shuffle=True, callbacks=callbacks_list)


# In[18]:


show_info(model, X_test, y_test, log, weights='weights_solar.best.hdf5')


# In[ ]:


test_data = load_test_data()

df_teste = pd.read_csv('../input/dataset_teste.csv')


# In[ ]:


predict = model.predict(test_data)
predict = scaler_y.inverse_transform(predict)


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = df_teste["Id"]
submission['Sales'] = predict

submission.to_csv('submission.csv', index=False)


# In[ ]:




