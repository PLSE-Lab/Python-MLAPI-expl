#!/usr/bin/env python
# coding: utf-8

# # Ukrainian hryvnia exchange rate prediction using Keras
# 
# **by Tsepa Oleksii, Samoshin Andriy and Mysak Yuriy**
# 
# In this notebook we will walk through time series forecasting using Keras Neural Network, also the same problem was solved by XGBoost and ARIMA, but they were not as good as Keras. This notebook was made for NBU IT Challenge on topic "Building a neural network for forecasting exchange rates".
# Full pack of datasets and solutions you can find [here](https://github.com/imgremlin/NBU-IT-Challenge).

# # Data Exploration
# 
# **Main dataframe**
# 
# So let's turn to the code. First of all import main file with our data about which you can read in "data_tutorial.txt".

# In[ ]:


import numpy as np
import pandas as pd

data = pd.read_excel('/kaggle/input/final_data.xlsx', parse_dates=True, index_col='date')
data=data[:-1]
data_raw=data.copy()

data.head()


# There are a lot of missing data, but we can deal with them using interpolation. We choose linear interpolation.
# 
# Missing data appears due to the fact that some data is given quarterly, other annually, other monthly.

# In[ ]:


data = data.interpolate(method='polynomial', order=1)

data.head()


# **USD dataframe**
# 
# Then we're importing exchange rate data which is given daily

# In[ ]:


data_usd = pd.read_csv('/kaggle/input/exchange_rate.csv')
data_usd_raw = data_usd.copy()

data_usd.head()


# So, let's clean the dataset: we need only date columns, which will be used as an index and column 'exrate'.

# In[ ]:


data_usd['date'] = pd.to_datetime(data_usd['date'],
                    format='%d.%m.%Y', errors='ignore')
data_usd = data_usd.set_index('date')
data_usd = data_usd['exrate']

data_usd.head()


# Build a plot to understand our exchange rate.

# In[ ]:


import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.plot(data_usd)
plt.show()


# As I said, we have got daily usd data and other data is given monthly, so we've to reshape usd data by taking a mean across the month. Then we're cropping our dataset due to inconsistencies in date limits. Soon we'll work with interbank dataset, which beginning from 2012.

# In[ ]:


data_usd = data_usd.resample('M').mean()

end_date = '2019-12-31'
data_usd = data_usd[:end_date]

data = data.assign(exrate = data_usd.values)

start_date = '2012-01-01'
data = data[start_date:]

data.head()


# **Interbank dataframe**
# 
# Import one more dataset, where we can find how much currency sold NBU on a given day

# In[ ]:


data_interbank = pd.read_excel('/kaggle/input/interbank.xlsx')

data_interbank.head()


# If we want to convert this dataset to the Series with date index and value, we need to clean it. Firstly let's use regexp to get rid of last to cyrillic characters and adapt to the datetime format.

# In[ ]:


import re
 
def regexp(reg):

    res = re.findall(r'\d{2}.\d{2}.\d{4}', reg)
    return res[0]     

data_interbank['date'] = data_interbank['date'].apply(regexp)

data_interbank['date'].head()


# Then, interbank data has got another problem - last two columns are not numerical, so we have to replace all ' , ' to ' . ' and then convert them to float.

# In[ ]:


def replace(rep):

    rep = rep.replace(',', '.') 
    return rep

def to_float(fl):

    fl = float(fl)
    return fl

data_interbank['total_amount_usd'] = data_interbank['total_amount_usd'].apply(replace)
data_interbank['total_amount_usd'] = data_interbank['total_amount_usd'].apply(to_float)

data_interbank['total_amount_usd'].head()


# Finally, we can do the same things with interbank data as with usd data and then concatenate 3 datasets to make a single whole.

# In[ ]:


data_interbank['date'] = pd.to_datetime(data_interbank['date'],
                    format='%d.%m.%Y', errors='ignore')
data_interbank = data_interbank.set_index('date')
data_interbank = data_interbank['total_amount_usd']
data_interbank = data_interbank.resample('M').sum()
data_interbank = data_interbank[start_date:]

data_interbank.head()

data = data.assign(interbank = data_interbank.values)


# # Data Preprocessing
# 
# **Scaling**
# 
# Divide dataset to X and y. Because of using Neural Networks it is a must to scale all data. We used MinMaxScaler. If you want then decode our predictions, you have to use two separate scalers - for X and y.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler(feature_range = (0, 1))

X = data.drop(labels=['exrate'], axis=1)
X = pd.DataFrame(scaler_X.fit_transform(X), columns = X.columns)

X.head()


# Here do the same things, but to scale 1D dataset, we need to reshape it.

# In[ ]:


scaler_y = MinMaxScaler(feature_range = (0, 1))

y = np.array(data['exrate'])
y = np.reshape(y, (len(y),-1))
y = pd.DataFrame(scaler_y.fit_transform(y))

y.head()


# **Fixing outliers**
# 
# Now, we've to work with each column in X and fix outliers - extreme values that deviate from other observations on data. As an example let's take 'ppi' column, where you can see outlier right before 40s index.

# In[ ]:


def raw_plot(data, column_name):

    plt.plot(data.index, data[column_name], label=column_name)
    plt.legend()
    plt.show()    
    
raw_plot(X, 'ppi')


# But it easily to detect outliers by using box plots. Above plot shows point equals to 1, that is outlier as that is not included in the box of other observation i.e no where near the quartiles.

# In[ ]:


def box(feat):

    plt.boxplot(x=X[feat])
    plt.title(feat)
    plt.show()     
    
box('ppi')


# Make a list of our columns in order to further work with it.

# In[ ]:


features = list(X.columns)
print(features)


# We write a function that removes outliers by equating them to a certain quartile, the quartile is chosen manually.

# In[ ]:


def fix_outliers(column):
    
    learning_rate = 0.35
    
    q1 = X[column].quantile(0.25)
    q3 = X[column].quantile(0.75)
    iqr = q3-q1
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    
    X[column].loc[(X[column] >= fence_high)] = X[column].quantile(1-learning_rate)
    X[column].loc[(X[column] <= fence_low)] = X[column].quantile(learning_rate)
        
for col in features:
    fix_outliers(col)  


# You can see that now we haven't got outliers on both plots.

# In[ ]:


raw_plot(X, 'ppi')
box('ppi')


# **Feature lags**
# 
# Due to the fact that our task is to predict the exchange rate, we can not use the data from the same month to predict on itself, so we created feature lags. Feature lags are new variables which shows impact of a certain feature a few months later. Then drop our non-lagged features.

# In[ ]:


def feature_lag(features):

    for feature in features:
        X[feature + '-lag1'] = X[feature].shift(1)
        X[feature + '-lag2'] = X[feature].shift(2)
        X[feature + '-lag3'] = X[feature].shift(3)
        X[feature + '-lag6'] = X[feature].shift(6)
        X[feature + '-lag12'] = X[feature].shift(12)
    
feature_lag(features)  
X.drop(features, axis=1, inplace=True)

print(X.columns)


# Because of we are using lags, we get new missing values. Remove them in X dataframe and then remove the same amount of rows in y dataframe.

# In[ ]:


X.head()

real_X_size = len(X)
X = X.dropna()
dropna_X_size = len(X)
y = y[real_X_size-dropna_X_size:]


# # Modeling
# 
# Define training size, then splitting X and y on train and test sets.

# In[ ]:


train_size = 0.78
separator = round(len(X.index)*train_size)

X_train, y_train = X.iloc[0:separator], y.iloc[0:separator]
X_test, y_test = X.iloc[separator:], y.iloc[separator:]


# Build a model using Keras Regressor. Almost all parameters were taking by using GridSearch, it's not hard, so you can do it on your own, but it takes some time. To make the process faster - you can some other tools such as Random Search or Keras Tuner.

# In[ ]:


from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.constraints import maxnorm

def build_model():

    model = Sequential([
    Dense(128, activation='relu', input_shape=[len(X.columns)],
                        kernel_constraint=maxnorm(5)),
    Dropout(0.3),
    Dense(1, kernel_initializer='normal', activation='sigmoid')])
    optimizer = Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

model = KerasRegressor(build_fn=build_model, epochs=200, batch_size=10, verbose=0)


# Fit model and make predictions. 

# In[ ]:


history = model.fit(X_train, y_train)
preds = model.predict(X_test)

print(preds)


# Now we have to transform our predictions from 0-1 scope to real values.

# In[ ]:


predictions = scaler_y.inverse_transform([preds])
preds_real = [x for x in predictions[0]]

print(preds_real)


# Plot our predictions to compare them with real values.

# In[ ]:


def predict_plot():

    ind_preds = data['2018-07-01':'2019-12-31']
    fig, axs = plt.subplots(1, figsize=(9,7))
    fig.suptitle('Predictions/real values')
    axs.plot(data.index, data.exrate, 'b-', label='real')
    axs.plot(ind_preds.index, preds_real, 'r-', label='prediction')
    axs.legend(loc=2)

predict_plot()  


# Calculate error to evaluate model and improve in future.

# In[ ]:


from sklearn.metrics import mean_absolute_error

def errors(y_true, y_pred, r):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs( (y_true - y_pred)/y_true))*100
    print('MAPE: {}%'.format(mape.round(r)))
    print('MAE: {}'.format(mean_absolute_error(y_true, y_pred).round(r)))  

errors(data_usd['2018-7-01':'2019-12-31'], preds_real, 3)


# That's all, thank you for reading! Leave your comments and suggestion what would you change in this model.
