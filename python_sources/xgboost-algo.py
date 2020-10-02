#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_log_error


# In[ ]:


train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission_df = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.isna().sum()


# In[ ]:


test_df.isna().sum()


# In[ ]:


train_df['Province_State'].unique()


# In[ ]:


train_df['Province_State'].fillna('',inplace=True)
test_df['Province_State'].fillna('',inplace=True)

lb = LabelEncoder()
train_df['Country_Region'] = lb.fit_transform(train_df['Country_Region'])
test_df['Country_Region'] = lb.transform(test_df['Country_Region'])

lb1 = LabelEncoder()
train_df['Province_State'] = lb.fit_transform(train_df['Province_State'])
test_df['Province_State'] = lb.transform(test_df['Province_State'])

def split_date(date):
    date = date.split('-')
    date[0] = int(date[0])
    if(date[1][0] == '0'):
        date[1] = int(date[1][1])
    else:
        date[1] = int(date[1])
    if(date[2][0] == '0'):
        date[2] = int(date[2][1])
    else:
        date[2] = int(date[2])    
    return date
train_df.Date = train_df.Date.apply(split_date)
test_df.Date = test_df.Date.apply(split_date)

year = []
month = []
day = []
for i in train_df.Date:
    year.append(i[0])
    month.append(i[1])
    day.append(i[2])
train_df['Year'] = year
train_df['Month'] = month
train_df['Day'] = day
del train_df['Date']


year = []
month = []
day = []
for i in test_df.Date:
    year.append(i[0])
    month.append(i[1])
    day.append(i[2])
test_df['Year'] = year
test_df['Month'] = month
test_df['Day'] = day
del test_df['Date']
del train_df['Id']
del test_df['ForecastId']
del train_df['Year']
del test_df['Year']

train_df['ConfirmedCases'] = train_df['ConfirmedCases'].apply(int)
train_df['Fatalities'] = train_df['Fatalities'].apply(int)

cases = train_df.ConfirmedCases
fatalities = train_df.Fatalities
del train_df['ConfirmedCases']
del train_df['Fatalities']

scaler = MinMaxScaler()
X = scaler.fit_transform(train_df.values)
x_test = scaler.transform(test_df.values)


# In[ ]:


train_df.head()


# In[ ]:


# X_train, X_valid, y_train, y_valid = train_test_split( X, cases, test_size=0.3, random_state=42)


# In[ ]:


# for i in range(20):
    xg = XGBRegressor(n_estimators = 10*1000 , random_state = 0 , max_depth = 15)
    xg.fit(X,cases)

    cases_pred = xg.predict(x_test)
    cases_pred[cases_pred < 0] = 0
#     print("error",'\t',mean_squared_log_error(y_valid, cases_pred))


# In[ ]:


# from keras.callbacks import ModelCheckpoint
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error 
# from matplotlib import pyplot as plt
# import seaborn as sb
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import warnings 
# warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# from xgboost import XGBRegressor


# In[ ]:


# model = Sequential()

# # The Input Layer :
# model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# # The Hidden Layers :
# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# # The Output Layer :
# model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# # Compile the network :
# model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
# model.summary()


# In[ ]:



# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]


# In[ ]:


# history = model.fit(X, cases, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# In[ ]:


# loss_train = history.history['loss']
# loss_val = history.history['val_loss']
# epochs = range(50)
# plt.plot(epochs, loss_train, 'g', label='Training loss')
# plt.plot(epochs, loss_val, 'b', label='validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# In[ ]:


# loss_train = history.history['acc']
# loss_val = history.history['val_acc']
# epochs = range(1,11)
# plt.plot(epochs, loss_train, 'g', label='Training accuracy')
# plt.plot(epochs, loss_val, 'b', label='validation accuracy')
# plt.title('Training and Validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# In[ ]:


# wights_file = 'Weights-039--3.66521.hdf5'
# model.load_weights(wights_file) # load it


# In[ ]:


X_cas = []
for i in range(len(X)):
    x = list(X[i])
    x.append(cases[i])
    X_cas.append(x)

X_cas = np.array(X_cas)

# X_train, X_valid, y_train, y_valid = train_test_split( X_cas, fatalities, test_size=0.3, random_state=42)


# In[ ]:


rf = XGBRegressor(n_estimators = 10000 , random_state = 0 , max_depth = 15)
rf.fit(X,fatalities)
fatalities_pred = rf.predict(x_test)
fatalities_pred[fatalities_pred < 0] = 0
# mean_squared_log_error(y_valid, fatalities_pred)


# In[ ]:


submission_df['ConfirmedCases'] = cases_pred
submission_df['Fatalities'] = fatalities_pred


# In[ ]:


submission_df.head()


# In[ ]:


submission_df.to_csv("submission.csv" , index = False)


# In[ ]:




