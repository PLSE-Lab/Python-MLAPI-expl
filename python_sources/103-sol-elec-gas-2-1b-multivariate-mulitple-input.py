#!/usr/bin/env python
# coding: utf-8

# These notebooks are based on the excellent article by Jason Brownlee:
# How to Develop Convolutional Neural Network Models for Time Series Forecasting.  
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import keras


# In[ ]:


print('tf version:',tf.__version__,'\n' ,'keras version:',keras.__version__,'\n' ,'numpy version:',np.__version__)


# This notbook uses :  
# tf version: 2.0.0-beta1 ;
#  keras version: 2.2.4 ; 
#  numpy version: 1.16.4 

# test 103 : test prediction solarpower with *multivariate mulitiple input series* and CNN
# 
# Multi-Headed 1D CNN model

# In[ ]:


p_data_filename = '../input/101-univariate-and-cnn-model-on-daily-solar-power/predicted_data.hdf5'
p_data_filename2 = 'predicted_data103B_2.hdf5'
p_data_filename3 = 'predicted_data103B_3.hdf5'
predicted_data = pd.read_hdf(p_data_filename)


# In[ ]:


solarpower = pd.read_csv("../input/solarpanelspower/PV_Elec_Gas2.csv",header = None,skiprows=1 ,names = ['date','cum_power','Elec_kW', 
                                                                            'Gas_mxm'], sep=',',usecols = [0,1,2,3],
                     
                     parse_dates={'dt' : ['date']}, infer_datetime_format=True,index_col='dt')
print(solarpower.head(2))


# 1. Make cumulative solar power stationary

# In[ ]:


# make cum_power stationary

solarpower2 = solarpower.shift(periods=1, freq='D', axis=0)
solarpower['cum_power_shift'] = solarpower2.loc[:,'cum_power']
solarpower['day_power'] = solarpower['cum_power'].values - solarpower['cum_power_shift']
solarpower.iloc[0:1].day_power.value = 0.
A = solarpower.dropna()
del A['cum_power'], A['cum_power_shift']
solarpower = A


# In[ ]:


solarpower.head(2), solarpower.tail(2)


# 2. Make Elec_kW postive

# In[ ]:


solarpower_plus =   solarpower.copy()
solarpower_plus['Elec_kW'] = solarpower_plus.Elec_kW - solarpower_plus.Elec_kW.min()
solarpower_plus.describe()


# 3. Split in training set from beginning to oktober 2018, validation is one year after training set

# In[ ]:


X_train = solarpower_plus[:'2018-10-28']
X_valid = solarpower_plus['2018-10-29':'2019-10-28'] # is 365 days
X_train.shape, X_valid.shape


# In[ ]:


X_train.tail(2), X_valid.head(2)


# 4. Devide the series into multiple input and output patterns

# In[ ]:


# we devide the series into multiple input and output patterns
def my_split_window(array, out_sequence, window):
    '''
    the Pandas dataframe has the columns (features) that we use as input (X).
    out_sequence is the time series that matches the input
    Returns array X with the features windowed in shape (number of windows, window, n_features)
    and array y
    '''
    X = []
    y = []
    n_steps = len(out_sequence) - window + 1
    for step in range(n_steps):
        X_w = []
        for i in range(window):
            X_w.append(array[(step + i),:])
        X.append(X_w)
        y.append(out_sequence[step + window -1])
    X = np.array(X)
    y = np.array(y)
    return X, y


# 5. Test my_split_window on test dataframe to check if function ok

# In[ ]:


import timeit
start = timeit.timeit()
# test my_split_window
df = pd.DataFrame()
df['feature1'] = [10,20,30,40,50,60,70,80,90]
df['feature2'] = [11,21,31,41,51,61,71,81,91]
array = np.array(df[['feature1', 'feature2']])

out_sequence = [26, 46, 66, 86, 106, 126, 146, 166, 186]
window = 3
X_, y_ = my_split_window(array, out_sequence, window)
stop = timeit.timeit()
print(stop-start)
X_, y_


# In[ ]:


X_.shape, y_.shape


# 6. Apply my_split_window on X_train

# In[ ]:


# apply my_split_window on dayly solar power with a window of 365 days (we do not make account for leap years)
# the input series is the daily solar power
start = timeit.timeit()
X_features = ['Elec_kW' , 'Gas_mxm']
X_train_input = np.array(X_train[ X_features])
out_sequence = X_train.day_power.values
window = 365
X, y = my_split_window(X_train_input, out_sequence,  window)
X_solar_f2 = X
y_solar_f2 = y
stop = timeit.timeit()
print(stop-start)
# print a sample
for i in range(3):
    print(X[i][-5:], y[i])


# We want to use a one-dimensional Convolutional Neural Network (1D CNN). Just like in a CNN for images,  
# a 1D CNN extracts features. It is very usefull in timeseries. More info is on the links:  
# https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/  
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/  
# 

# # 7. Multi-headed CNN model
# There is an other more elaborate way to the problem. Each input series can be handled by a separate CNN and the output of each of these submodels can be combined.  
# It offers more flexibility or better performance.  
# We will make a two-headed model first.
# 
# 

# In[ ]:


# plot funciotn for the loss of the fitted moddel
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.title('loss')
    plt.xlabel('epochs')
    plt.show()
    return


# In[ ]:


# input model is part 1 of the multi head
def input_model_visible(window, n_features=1):
    visible = tf.keras.layers.Input(shape=(window, n_features))
    cnn = tf.keras.layers.BatchNormalization()(visible)
    cnn = tf.keras.layers.Conv1D(filters = 32, kernel_size=2, activation='relu')(cnn)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = tf.keras.layers.Flatten()(cnn)
    return cnn, visible

# output model for 2 input features
def output_model_2_heads(cnn1, cnn2, visible1, visible2):
    merge = tf.keras.layers.Concatenate()(inputs=([cnn1, cnn2]))
    bn = tf.keras.layers.BatchNormalization()(merge)
    dense = tf.keras.layers.Dense(50, activation='relu')(bn)
    output = tf.keras.layers.Dense(1)(merge)
    model = tf.keras.Model(inputs=[visible1, visible2], outputs=output)
    return model

# output model for 3 features
def output_model_3_heads(cnn1, cnn2, cnn3, visible1, visible2, visible3):
    merge = tf.keras.layers.Concatenate()(inputs=([cnn1, cnn2, cnn3]))
    bn = tf.keras.layers.BatchNormalization()(merge)
    dense = tf.keras.layers.Dense(50, activation='relu')(bn)
    output = tf.keras.layers.Dense(1)(merge)
    model = tf.keras.Model(inputs=[visible1, visible2, visible3], outputs=output)
    return model

def multi_2_head_model(window):
    cnn1, visible1 = input_model_visible(window)
    cnn2, visible2 = input_model_visible(window)
    model = output_model_2_heads(cnn1, cnn2, visible1, visible2)
    return model

def multi_3_head_model(window):
    cnn1, visible1 = input_model_visible(window)
    cnn2, visible2 = input_model_visible(window)
    cnn3, visible3 = input_model_visible(window)
    model = output_model_3_heads(cnn1, cnn2, cnn3, visible1, visible2, visible3)
    return model


# In[ ]:


### tf.keras.backend.clear_session()  # For easy reset of notebook state.

# This model requires input split into two elements
# we need an input shape = (number of windows, window, feature1=1) 
#  and we have a window size of one year (365 days) 
# we have to reshape

window = 365
X1 = X[:,:,0].reshape(X.shape[0], X.shape[1], 1)
X2 = X[:,:,1].reshape(X.shape[0], X.shape[1], 1)
    
model_solar = multi_2_head_model(window)

model_solar.compile(optimizer='adam', loss='mae')

epochs = 20
# fit model
history_solar = model_solar.fit([X1, X2], y, epochs=epochs, verbose=0)
plot_history(history_solar)


# 8. Predict the solar power based on the validation features to check the model

# In[ ]:


# predicting next year
# the model expects an input of shape(n_time steps = window size, n_features)


y_hat = []
features = ['Elec_kW' , 'Gas_mxm']


def predict_next_year_from_valid(X_train, features, X_valid, model):
    X_input =  np.array(X_train[features][-365:]) #  next value based on data of last year
    X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1]) # input must have 3 dimensions
    x_input = X_input
    for i in range(365):
        new_x = np.array(X_valid[features].iloc[i])
        new_x = new_x.reshape(1, 1, X_input.shape[2])
        x_input = np.concatenate((x_input[:, -364:], new_x), axis=1)
        x_input1 = x_input[:,:,0].reshape(1, 365, 1)
        x_input2 = x_input[:,:,1].reshape(1, 365, 1)
        y_hat.append((model.predict([x_input1, x_input2], verbose=0).ravel())[0])
    
    return y_hat

y_hat_solar = predict_next_year_from_valid(X_train, features, X_valid, model_solar)


# In[ ]:



plt.plot(y_hat_solar, label='predicted_power')

y_true = X_valid.day_power.values
plt.plot(y_true, label='true_power')
plt.legend()
plt.show()


# In[ ]:


first_r2_score = r2_score(y_true, y_hat) # Best possible score is 1.0 
first_mae = mean_absolute_error(y_true, y_hat)
print('r2_score %.5f' % first_r2_score)
print('mae %.2f' % first_mae)


# In[ ]:


# 100 epochs : 0.42520212661926315


# # 9. But the cumulative power is actually much more intersting.#
# # It tels us what the the total expected solar power of that year will be. #

# In[ ]:


def cumulate(series, start=0):
    '''
    start is the starting cumulative power, the series is the daily solar power
    a list with daily cumulative power is the result
    '''
    cum = [start]
    for i in range(len(series)):
        sum_plus = cum[i] + series[i]
        cum.append(sum_plus)
    return cum


# In[ ]:


y_true_cumulative = cumulate(y_true)
y_predicted_cumulative = cumulate(y_hat)

plt.plot(y_predicted_cumulative, label='predicted_power')
plt.plot(y_true_cumulative, label='true_power')
plt.legend()
plt.show()


# In[ ]:


true_cumulative_power_after_one_year = int(y_true_cumulative[-1])
predicted_cumulative_power_after_one_year = int(y_predicted_cumulative[-1])
print('true cumulative power after one year:', true_cumulative_power_after_one_year)
print('predicted cumulative power after one year:', predicted_cumulative_power_after_one_year)

acc_one_year = 1- (true_cumulative_power_after_one_year - predicted_cumulative_power_after_one_year)/true_cumulative_power_after_one_year
acc_one_year = acc_one_year * 100

print('accuracy after one year: %.2f' %  acc_one_year,'%')
print('r2 score %.5f ' % r2_score(y_true_cumulative, y_predicted_cumulative))
print('mae  %.2f' % mean_absolute_error(y_true_cumulative, y_predicted_cumulative))


# # 10. We have used the validation set for predicting but when we want to predict the future we have to use the last prediction(s) of all features
# 

# In[ ]:


# we have to train the models for every feature otherwise we can not predict future features
# first we have to make new split windows
window = 365

features_solar = ['Elec_kW' , 'Gas_mxm']
features_elec = ['day_power', 'Gas_mxm']
features_gas = ['day_power', 'Elec_kW']

X_solar, y_solar = X_solar_f2, y_solar_f2  # restore backup X,y
print('solar done')

X_train_input_elec = np.array(X_train[ features_elec])
out_sequence_elec = X_train.Elec_kW.values
X_elec, y_elec = my_split_window(X_train_input_elec, out_sequence_elec,  window)
print('elec done')

X_train_input_gas = np.array(X_train[ features_gas ])
out_sequence_gas = X_train.Gas_mxm.values
X_gas, y_gas = my_split_window(X_train_input_gas, out_sequence_gas,  window)
print('gas done')

X1_solar = X_solar[:,:,0].reshape(X_solar.shape[0], X_solar.shape[1], 1)
X2_solar = X_solar[:,:,1].reshape(X_solar.shape[0], X_solar.shape[1], 1)

X1_elec = X_elec[:,:,0].reshape(X_elec.shape[0], X_elec.shape[1], 1)
X2_elec = X_elec[:,:,1].reshape(X_elec.shape[0], X_elec.shape[1], 1)

X1_gas = X_gas[:,:,0].reshape(X_gas.shape[0], X_gas.shape[1], 1)
X2_gas = X_gas[:,:,1].reshape(X_gas.shape[0], X_gas.shape[1], 1)

model_solar = multi_2_head_model(window)
model_solar.compile(optimizer='adam', loss='mae')

model_elec = multi_2_head_model(window)
model_elec.compile(optimizer='adam', loss='mae')

model_gas = multi_2_head_model(window)
model_gas.compile(optimizer='adam', loss='mae')

epochs = 25
# fit models
history_solar = model_solar.fit([X1_solar, X2_solar], y_solar, epochs=epochs, verbose=0)
print('solar model ok')
#epochs = 50
history_elec = model_elec.fit([X1_elec, X2_elec], y_elec, epochs=epochs, verbose=0)
print('elec model ok')
#epochs = 30
history_gas = model_gas.fit([X1_gas, X2_gas], y_gas, epochs=epochs, verbose=0) 
print('gas model ok')

print('solar')
plot_history(history_solar)
print('elec')
plot_history(history_elec)
print('gas')
plot_history(history_gas)


# In[ ]:


# predicting next year
# the model expects an input of shape(n_time steps = window size, n_features)


def predict_next_year_from_valid(X_train, 
                                 features_solar, 
                                 features_elec,
                                 features_gas,
                                 model_solar, model_elec, model_gas):
    y_hat_solar = []
    y_hat_elec = []
    y_hat_gas = []
    X_in_solar =  np.array(X_train[features_solar][-365:]) #  next value based on data of last year
    X_in_solar = X_in_solar.reshape(1, X_in_solar.shape[0], X_in_solar.shape[1]) # input must have 3 dimensions
    X_in_elec =  np.array(X_train[features_elec][-365:]) #  next value based on data of last year
    X_in_elec = X_in_elec.reshape(1, X_in_elec.shape[0], X_in_elec.shape[1]) # input must have 3 dimensions
    X_in_gas =  np.array(X_train[features_gas][-365:]) #  next value based on data of last year
    X_in_gas = X_in_gas.reshape(1, X_in_gas.shape[0], X_in_gas.shape[1]) # input must have 3 dimensions
    print(X_in_gas.shape)
    for i in range(365):
        X_in1_solar = X_in_solar[:,:,0].reshape(1, 365, 1)
        X_in2_solar = X_in_solar[:,:,1].reshape(1, 365, 1)
        y_hat_solar.append((model_solar.predict([X_in1_solar, X_in2_solar], verbose=0).ravel())[0])
        if np.array(y_hat_solar[i]) < 0:
            y_hat_solar[i] = 0 
        new_X_solar = np.array(y_hat_solar[i])

        #print(new_X_solar)
        X_in1_elec = X_in_elec[:,:,0].reshape(1, 365, 1)
        X_in2_elec = X_in_elec[:,:,1].reshape(1, 365, 1)
        y_hat_elec.append((model_elec.predict([X_in1_elec, X_in2_elec], verbose=0).ravel())[0])
        if np.array(y_hat_elec[i]) < 0:
            y_hat_elec[i] = 0        
        new_X_elec = np.array(y_hat_elec[i])

        #print(new_X_elec)
        X_in1_gas = X_in_gas[:,:,0].reshape(1, 365, 1)
        X_in2_gas = X_in_gas[:,:,1].reshape(1, 365, 1)
        y_hat_gas.append((model_gas.predict([X_in1_gas, X_in2_gas], verbose=0).ravel())[0])
        if np.array(y_hat_gas[i]) < 0:
            y_hat_gas[i] = 0
        new_X_gas = np.array(y_hat_gas[i])
        
        new_X_gas2 = np.array((new_X_solar, new_X_elec))
        new_X_gas2 = new_X_gas2.reshape(1,1,2)
        
        new_X_elec2 = np.array((new_X_solar, new_X_gas))
        new_X_elec2 = new_X_elec2.reshape(1,1,2)
        
        new_X_solar2 = np.array((new_X_elec, new_X_gas))
        new_X_solar2 = new_X_solar2.reshape(1,1,2)
        
        X_in_gas = np.concatenate((X_in_gas[:, -364:], new_X_gas2), axis=1)        
        X_in_solar = np.concatenate((X_in_solar[:, -364:], new_X_solar2), axis=1)
        X_in_elec = np.concatenate((X_in_elec[:, -364:], new_X_elec2), axis=1)



    return y_hat_solar, y_hat_elec, y_hat_gas

y_hat_solar, y_hat_elec, y_hat_gas  = predict_next_year_from_valid(X_train, features_solar, features_elec, features_gas,
                                           model_solar, model_elec, model_gas)


# In[ ]:


plt.plot(y_hat_solar, label='predicted_power')
y_true = X_valid.day_power.values
plt.plot(y_true, label='true_power')
plt.legend()
plt.show()


# In[ ]:


first_r2_score = r2_score(y_true, y_hat_solar) # Best possible score is 1.0 
first_mae = mean_absolute_error(y_true, y_hat)
print('r2_score %.2f' % first_r2_score)
print('mae %.2f' % first_mae)


# In[ ]:


y_true_cumulative = cumulate(y_true)
y_predicted_cumulative = cumulate(y_hat_solar)

plt.plot(y_predicted_cumulative, label='predicted_power')
plt.plot(y_true_cumulative, label='true_power')
plt.legend()
plt.show()


# In[ ]:



predicted_data['103B_Elec_Gas_2features_gas_elec'] = y_hat_solar

predicted_data.to_hdf(p_data_filename2, key='predicted_data', mode='a')

# predicted_data = pd.read_hdf('predicted_data.hdf5')


# # 11 what if we add a feature?
# We can make an extra feature by adding Elecricty and Gas

# In[ ]:


X_train = X_train.copy()
X_valid = X_valid.copy()
X_train['Gas_plus_Elec'] = X_train.Gas_mxm + X_train.Elec_kW
X_valid['Gas_plus_Elec'] = X_valid.Gas_mxm + X_valid.Elec_kW

# apply my_split_window on dayly solar power with a window of 365 days (we do not make account for leap years)
# the input series is the daily solar power
X_features = ['Elec_kW' , 'Gas_mxm', 'Gas_plus_Elec']
X_train_input = np.array(X_train[ X_features])
out_sequence = X_train.day_power.values
window = 365
X, y = my_split_window(X_train_input, out_sequence,  window)
X_solar_3f = X
y_solar_3f = y
# print a sample
for i in range(3):
    print(X[i][-2:], y[i])


tf.keras.backend.clear_session()  # For easy reset of notebook state.

# This model requires input split into three elements
# we need an input shape = (number of windows, window, feature1=1) 
#  and we have a window size of one year (365 days) 
# we have to reshape

window = window = 365
X1 = X[:,:,0].reshape(X.shape[0], X.shape[1], 1)
X2 = X[:,:,1].reshape(X.shape[0], X.shape[1], 1)
X3 = X[:,:,2].reshape(X.shape[0], X.shape[1], 1)

model_solar2 = multi_3_head_model(window)
model_solar2.compile(optimizer='adam', loss='mae')

# fit model
epochs = 25
history = model_solar2.fit([X1, X2, X3], y, epochs=epochs, verbose=0)

# graph of the loss shows convergence
plot_history(history)


# predicting next year with X_valid as input
# the model expects an input of shape(n_time steps = window size, n_features)
y_hat = []
features = ['Elec_kW' , 'Gas_mxm', 'Gas_plus_Elec']
X_input =  np.array(X_train[features][-365:]) #  next value based on data of last year
X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1]) # input must have 3 dimensions
x_input = X_input
for i in range(365):
    new_x = np.array(X_valid[features].iloc[i])
    new_x = new_x.reshape(1, 1, X_input.shape[2])
    x_input = np.concatenate((x_input[:, -364:], new_x), axis=1)
    x_input1 = x_input[:,:,0].reshape(1, 365, 1)
    x_input2 = x_input[:,:,1].reshape(1, 365, 1)
    x_input3 = x_input[:,:,2].reshape(1, 365, 1)    
    y_hat.append((model_solar2.predict([x_input1, x_input2, x_input3], verbose=0).ravel())[0])
    


plt.plot(y_hat, label='predicted_power')
y_true = X_valid.day_power.values
plt.plot(y_true, label='true_power')
plt.legend()
plt.show()

first_r2_score = r2_score(y_true, y_hat) # Best possible score is 1.0 
first_mae = mean_absolute_error(y_true, y_hat)
print('r2_score %.2f' % first_r2_score)
print('mae %.2f' % first_mae)

y_true_cumulative = cumulate(y_true)
y_predicted_cumulative = cumulate(y_hat)

plt.plot(y_predicted_cumulative, label='predicted_power')
plt.plot(y_true_cumulative, label='true_power')
plt.legend()
plt.show()

true_cumulative_power_after_one_year = int(y_true_cumulative[-1])
predicted_cumulative_power_after_one_year = int(y_predicted_cumulative[-1])
print('true cumulative power after one year:', true_cumulative_power_after_one_year)
print('predicted cumulative power after one year:', predicted_cumulative_power_after_one_year)

acc_one_year = 1- (true_cumulative_power_after_one_year - predicted_cumulative_power_after_one_year)/true_cumulative_power_after_one_year
acc_one_year = acc_one_year * 100

print('accuracy after one year: %.2f' %  acc_one_year,'%')
print('r2 score %.5f ' % r2_score(y_true_cumulative, y_predicted_cumulative))
print('mae  %.2f' % mean_absolute_error(y_true_cumulative, y_predicted_cumulative))


# # 12. Use predictions to predict the future

# In[ ]:


# 11 Use three features 'Elec_kW' ,'Gas_mxm' ,'Gas_plus_Elec' to predict the next solar power step
# we have to train the models for every feature otherwise we can not predict future features
# first we have to make new split windows
window = 365

features_solar = ['Elec_kW' ,'Gas_mxm' ,'Gas_plus_Elec']
features_elec = ['day_power','Gas_mxm' , 'Gas_plus_Elec']
features_gas = ['day_power', 'Elec_kW', 'Gas_plus_Elec']
features_gas_elec = ['day_power', 'Elec_kW', 'Gas_mxm']

X_train_input_solar = np.array(X_train[ features_solar])
out_sequence_solar = X_train.day_power.values
X_solar, y_solar = my_split_window(X_train_input_solar, out_sequence_solar,  window)
print('solar done')

X_train_input_elec = np.array(X_train[ features_elec])
out_sequence_elec = X_train.Elec_kW.values
X_elec, y_elec = my_split_window(X_train_input_elec, out_sequence_elec,  window)
print('elec done')

X_train_input_gas = np.array(X_train[ features_gas])
out_sequence_gas = X_train.Gas_mxm.values
X_gas, y_gas = my_split_window(X_train_input_gas, out_sequence_gas,  window)
print('gas done')

X_train_input_gas_elec = np.array(X_train[ features_gas_elec ])
out_sequence_gas_elec = X_train.Gas_plus_Elec.values
X_gas_elec, y_gas_elec = my_split_window(X_train_input_gas_elec, out_sequence_gas_elec,  window)
print('gas_elec done')

X1_solar = X_solar[:,:,0].reshape(X_solar.shape[0], X_solar.shape[1], 1)
X2_solar = X_solar[:,:,1].reshape(X_solar.shape[0], X_solar.shape[1], 1)
X3_solar = X_solar[:,:,2].reshape(X_solar.shape[0], X_solar.shape[1], 1)

X1_elec = X_elec[:,:,0].reshape(X_elec.shape[0], X_elec.shape[1], 1)
X2_elec = X_elec[:,:,1].reshape(X_elec.shape[0], X_elec.shape[1], 1)
X3_elec = X_elec[:,:,2].reshape(X_elec.shape[0], X_elec.shape[1], 1)

X1_gas = X_gas[:,:,0].reshape(X_gas.shape[0], X_gas.shape[1], 1)
X2_gas = X_gas[:,:,1].reshape(X_gas.shape[0], X_gas.shape[1], 1)
X3_gas = X_gas[:,:,2].reshape(X_gas.shape[0], X_gas.shape[1], 1)

X1_gas_elec = X_gas_elec[:,:,0].reshape(X_gas_elec.shape[0], X_gas_elec.shape[1], 1)
X2_gas_elec = X_gas_elec[:,:,1].reshape(X_gas_elec.shape[0], X_gas_elec.shape[1], 1)
X3_gas_elec = X_gas_elec[:,:,2].reshape(X_gas_elec.shape[0], X_gas_elec.shape[1], 1)

model_solar = multi_3_head_model(window)
model_solar.compile(optimizer='adam', loss='mae')

model_elec = multi_3_head_model(window)
model_elec.compile(optimizer='adam', loss='mae')

model_gas = multi_3_head_model(window)
model_gas.compile(optimizer='adam', loss='mae')

model_gas_elec = multi_3_head_model(window)
model_gas_elec.compile(optimizer='adam', loss='mae')

epochs = 25
# fit models
history_solar = model_solar.fit([X1_solar, X2_solar, X3_solar], y_solar, epochs=epochs, verbose=0)
print('solar model ok')
epochs = 50
history_elec = model_elec.fit([X1_elec, X2_elec, X3_elec] , y_elec, epochs=epochs, verbose=0)
print('elec model ok')
#epochs = 30
history_gas = model_gas.fit([X1_gas, X2_gas, X3_gas], y_gas, epochs=epochs, verbose=0) 
print('gas model ok')
epochs = 50
history_gas_elec = model_gas_elec.fit([X1_gas_elec, X2_gas_elec, X3_gas_elec], y_gas_elec, epochs=epochs, verbose=0) 
print('gas_elec model ok')

print('solar')
plot_history(history_solar)
print('elec')
plot_history(history_elec)
print('gas')
plot_history(history_gas)
print('gas_elec')
plot_history(history_gas_elec)


# In[ ]:


# predicting next year
# the model expects an input of shape(n_time steps = window size, n_features)


def predict_next_year_from_valid(X_train, 
                                 features_solar, 
                                 features_elec,
                                 features_gas,
                                 features_gas_elec,
                                 model_solar, model_elec, model_gas ,model_gas_elec):
    y_hat_solar = []
    y_hat_elec = []
    y_hat_gas = []
    y_hat_gas_elec = []
    X_in_solar =  np.array(X_train[features_solar][-365:]) #  next value based on data of last year
    X_in_solar = X_in_solar.reshape(1, X_in_solar.shape[0], X_in_solar.shape[1]) # input must have 3 dimensions
    X_in_elec =  np.array(X_train[features_elec][-365:]) #  next value based on data of last year
    X_in_elec = X_in_elec.reshape(1, X_in_elec.shape[0], X_in_elec.shape[1]) # input must have 3 dimensions
    X_in_gas =  np.array(X_train[features_gas][-365:]) #  next value based on data of last year
    X_in_gas = X_in_gas.reshape(1, X_in_gas.shape[0], X_in_gas.shape[1])
    X_in_gas_elec =  np.array(X_train[features_gas_elec][-365:]) #  next value based on data of last year
    X_in_gas_elec = X_in_gas_elec.reshape(1, X_in_gas_elec.shape[0], X_in_gas_elec.shape[1]) # input must have 3 dimensions
    print(X_in_gas_elec.shape)
    for i in range(365):
        # split X_solar into 3 input series
        X_in1_solar = X_in_solar[:,:,0].reshape(1, 365, 1)
        X_in2_solar = X_in_solar[:,:,1].reshape(1, 365, 1)
        X_in3_solar = X_in_solar[:,:,2].reshape(1, 365, 1)
        y_hat_solar.append((model_solar.predict([X_in1_solar, X_in2_solar, X_in3_solar], verbose=0).ravel())[0])
        if np.array(y_hat_solar[i]) < 0:
            y_hat_solar[i] = 0 
        new_X_solar = np.array(y_hat_solar[i])

        # split X_elec into 3 input series
        X_in1_elec = X_in_elec[:,:,0].reshape(1, 365, 1)
        X_in2_elec = X_in_elec[:,:,1].reshape(1, 365, 1)
        X_in3_elec = X_in_elec[:,:,2].reshape(1, 365, 1)
        y_hat_elec.append((model_elec.predict([X_in1_elec, X_in2_elec, X_in3_elec], verbose=0).ravel())[0])
        if np.array(y_hat_elec[i]) < 0:
            y_hat_elec[i] = 0        
        new_X_elec = np.array(y_hat_elec[i])

        # split X_gas into 3 input series
        X_in1_gas = X_in_gas[:,:,0].reshape(1, 365, 1)
        X_in2_gas = X_in_gas[:,:,1].reshape(1, 365, 1)
        X_in3_gas = X_in_gas[:,:,2].reshape(1, 365, 1)
        y_hat_gas.append((model_gas.predict([X_in1_gas, X_in2_gas, X_in3_gas], verbose=0).ravel())[0])
        if np.array(y_hat_gas[i]) < 0:
            y_hat_gas[i] = 0
        new_X_gas = np.array((y_hat_gas[i]))

        # split X_gas_elec into 3 input series
        X_in1_gas_elec = X_in_gas_elec[:,:,0].reshape(1, 365, 1)
        X_in2_gas_elec = X_in_gas_elec[:,:,1].reshape(1, 365, 1)
        X_in3_gas_elec = X_in_gas_elec[:,:,2].reshape(1, 365, 1)
        y_hat_gas_elec.append((model_gas_elec.predict([X_in1_gas_elec, X_in2_gas_elec, X_in3_gas_elec], verbose=0).ravel())[0])
        if np.array(y_hat_gas_elec[i]) < 0:
            y_hat_gas_elec[i] = 0
        new_X_gas_elec = np.array((y_hat_gas_elec[i]))
        
        # reshape new_X_solar
        new_X_solar2 = np.array((new_X_elec, new_X_gas ,new_X_gas_elec))
        new_X_solar2 = new_X_solar2.reshape(1,1,3)
        
        # reshape new_X_elec
        new_X_elec2 = np.array((new_X_solar, new_X_gas, new_X_gas_elec))
        new_X_elec2 = new_X_elec2.reshape(1,1,3)
        
        # reshape new_X_gas
        new_X_gas2 = np.array((new_X_solar, new_X_elec, new_X_gas_elec))
        new_X_gas2 = new_X_gas2.reshape(1,1,3)
        
        # reshape new_X_gas_elec
        new_X_gas_elec2 = np.array((new_X_solar, new_X_elec, new_X_gas))
        new_X_gas_elec2 = new_X_gas_elec2.reshape(1,1,3)
        
        # concatenate with new       
        X_in_solar = np.concatenate((X_in_solar[:, -364:], new_X_solar2), axis=1)
        X_in_elec = np.concatenate((X_in_elec[:, -364:], new_X_elec2), axis=1)
        X_in_gas = np.concatenate((X_in_gas[:, -364:], new_X_gas2), axis=1) 
        X_in_gas_elec = np.concatenate((X_in_gas_elec[:, -364:], new_X_gas_elec2), axis=1) 


    return y_hat_solar, y_hat_elec, y_hat_gas, y_hat_gas_elec


# In[ ]:


y_hat_solar, y_hat_elec, y_hat_gas, y_hat_gas_elec  = predict_next_year_from_valid(X_train, 
                                            features_solar, features_elec, features_gas , features_gas_elec,
                                               model_solar, model_elec, model_gas , model_gas_elec)

plt.plot(y_hat_solar, label='predicted_power')
y_true = X_valid.day_power.values
plt.plot(y_true, label='true_power')
plt.legend()
plt.show()

plt.plot(y_hat_elec)
plt.plot(y_hat_gas)
plt.plot(y_hat_gas_elec)
plt.show()
y_true_cumulative = cumulate(y_true)
y_predicted_cumulative = cumulate(y_hat_solar)

plt.plot(y_predicted_cumulative, label='predicted_power')
plt.plot(y_true_cumulative, label='true_power')
plt.legend()
plt.show()


# In[ ]:


# import h5py

predicted_data['103B_Elec_Gas_3features_elec_gas_gas_plus_elec'] = y_hat_solar

predicted_data.to_hdf(p_data_filename3 ,key='predicted_data', table='true',mode='a')


predicted_data.head()


# In[ ]:


predicted_data.columns


# In[ ]:


true_cumulative_power_after_one_year = int(y_true_cumulative[-1])
predicted_cumulative_power_after_one_year = int(y_predicted_cumulative[-1])
print('true cumulative power after one year:', true_cumulative_power_after_one_year)
print('predicted cumulative power after one year:', predicted_cumulative_power_after_one_year)

acc_one_year = 1- (true_cumulative_power_after_one_year - predicted_cumulative_power_after_one_year)/true_cumulative_power_after_one_year
acc_one_year = acc_one_year * 100

print('accuracy after one year: %.2f' %  acc_one_year,'%')
print('r2 score %.5f ' % r2_score(y_true_cumulative, y_predicted_cumulative))
print('mae  %.2f' % mean_absolute_error(y_true_cumulative, y_predicted_cumulative))


# In[ ]:




