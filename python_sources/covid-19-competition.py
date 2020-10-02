#!/usr/bin/env python
# coding: utf-8

# # Read dataset

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import tensorflow as tf
import math

train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')


# # Explore dataset

# In[ ]:


train_df[train_df['Country/Region'] == 'Korea, South'].head(200)
#test_df.head()


# In[ ]:


train_df.groupby(['Country/Region', 'Province/State'])['Id'].agg(['count']).reset_index()


# # Preprocess dataset

# * Fill NaN values
# * Convert Date to Day count from the start date of major pandemic

# In[ ]:


train_df = train_df.fillna({'Province/State': 'Unknown'})
test_df = test_df.fillna({'Province/State': 'Unknown'})
train_df.isna().sum()


# In[ ]:


def to_datetime(dt):
    return datetime.datetime.strptime(dt, '%Y-%m-%d')

def count_days(dt):
    return (dt - datetime.datetime.strptime('2020-01-22', "%Y-%m-%d")).days

plot_df = train_df[train_df['Country/Region'] == 'Iran']
plot_df['Date'] = plot_df['Date'].map(to_datetime)
plot_df['Day'] = plot_df['Date'].map(count_days)
plt.plot(plot_df['Day'], plot_df['ConfirmedCases'].cumsum())


# * Logtransform ConfirmedCases because the distribution is left-skewed

# In[ ]:


train_df['ConfirmedCases_log'] = train_df['ConfirmedCases'].map(math.log1p)
train_df['ConfirmedCases_log'].hist(bins=50)


# In[ ]:


train_df['Country/Region'].unique()


# In[ ]:


train_df['Date_dt'] = train_df['Date'].map(to_datetime)
train_df['Day'] = train_df['Date_dt'].map(count_days)
test_df['Date_dt'] = test_df['Date'].map(to_datetime)
test_df['Day'] = test_df['Date_dt'].map(count_days)


# In[ ]:


historical_steps = 30

# todo: better split validate data
val_df = train_df[train_df['Day'] > (train_df['Day'].max() - historical_steps)]
#val_df = train_df[train_df['Country/Region'].isin(['China'])]
#train_drop_df = train_df.drop(val_df.index)
print('# Train DF \n {} \n# Val DF \n {} \n# Test DF \n {}'.format(train_df.head(), val_df.head(), test_df.head()))


# # Model Selection

# # 1. LSTM Model (Initially Chosen)
# 
# **Reshape dataset to sequences**
# 
# Input data includes a series of historical responses from historical_steps to the given time wherareas targets only include last snapshot

# In[ ]:


# historical_steps = 30
# output_steps = 10

# def make_sequential_input(df):
    
#     inputs, targets = [], []
    
#     for i in range(len(df) - historical_steps - 1):
        
#         if df.iloc[i]['Lat'] == df.iloc[i + historical_steps]['Lat'] and \
#             df.iloc[i]['Long'] == df.iloc[i + historical_steps]['Long']:
            
#             # iloc[a:b] startnig from index 'a' and ending before b
#             inputs.append(np.array(df.iloc[i:i + historical_steps][['Day', 'Lat', 'Long', 'ConfirmedCases_log', 'Fatalities']]).tolist())
#             targets.append(np.array(df.iloc[i + historical_steps][['ConfirmedCases_log']]).tolist())
        
#     return inputs, targets


# # Make sequential input for training and validation
# train_inputs, train_targets = make_sequential_input(train_df)
# val_inputs, val_targets = make_sequential_input(val_df)
# np.shape(train_inputs)


# **Create LSTM layers**

# In[ ]:


# historical_steps = 30

# input_feature_count = 5
# output_feature_count = 1
# hidden_node_count = 32

# batch_size = 32
# epochs = 20
# lr = 0.001

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(hidden_node_count, batch_input_shape=[None, historical_steps, input_feature_count], return_sequences=True))
# model.add(tf.keras.layers.LSTM(hidden_node_count))
# model.add(tf.keras.layers.Dense(output_feature_count, activation='sigmoid'))

# optimizer = tf.keras.optimizers.Adam(lr=lr)
# model.compile(loss="mean_squared_error", optimizer=optimizer)

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5)
# history = model.fit(train_inputs, train_targets, \
#                     epochs=epochs, \
#                     batch_size=batch_size, \
#                     validation_data=(val_inputs, val_targets), \
#                     callbacks=[early_stopping])


# **Predict Next Day**

# Found that the predicted results are way inaccurate, and one of main reasons is due to that:
# * Depending on geograhy, ConfirmedCases & Fatalities data can show different degree in performance
# 
# To train all data in a model, random effects should beb computed for multiple countries and applied to the model.
# 

# In[ ]:


# def _inverse_log(val):
#     return math.expm1(val)

# _inverse_log_v = np.vectorize(_inverse_log)

# # For certain country,
# df = train_df[(train_df['Country/Region'] == 'Korea, South') & (train_df['Province/State'] == 'Unknown') ]

# # Get latest tail & Conver to array
# inputs = np.array(df[['Day', 'Lat', 'Long', 'ConfirmedCases_log', 'Fatalities']][-tail_len:])

# # Get next ConfirmedCases & Fatalities
# _inverse_log_v(model.predict(np.array(inputs).reshape(1, historical_steps, input_feature_count)).reshape(-1))


# # 2. Linear Mixed Model (Finally Chosen)
# 
# Linear Mixed Effects models are used for regression analyses involving dependent data. Such data arise when working with longitudinal and other study designs in which multiple observations are made on each subject. Some specific linear mixed effects models are
# 
# * Random intercepts models, where all responses in a group are additively shifted by a value that is specific to the group.
# * Random slopes models, where the responses in a group follow a (conditional) mean trajectory that is linear in the observed covariates, with the slopes (and possibly intercepts) varying by group.
# * Variance components models, where the levels of one or more categorical covariates are associated with draws from distributions. These random terms additively determine the conditional mean of each observation based on its covariate values.

# **Train the model**

# In[ ]:


import statsmodels.api as sm

historical_steps = 7
train_df['Geo'] = train_df['Province/State'].astype(str) + ',' +train_df['Country/Region'].astype(str)
train_df['Fatalities_log'] = train_df['Fatalities'].map(math.log1p)

x_arr, y_arr, y_inverse_arr, grp_arr, day_arr = [], [], [], [], []
xf_arr, yf_arr, yf_inverse_arr = [], [], []

for i in range(len(train_df) - historical_steps -1):
    if train_df.iloc[i]['Lat'] == train_df.iloc[i+historical_steps]['Lat'] and         train_df.iloc[i]['Long'] == train_df.iloc[i+historical_steps]['Long']: 
        
        x_arr.append(np.array(train_df.iloc[i:i+historical_steps][['ConfirmedCases_log']]).reshape(-1).tolist())
        y_arr.append(np.array(train_df.iloc[i+historical_steps][['ConfirmedCases_log']]).tolist())
        y_inverse_arr.append(np.array(train_df.iloc[i+historical_steps][['ConfirmedCases']]).tolist())
        
        xf_arr.append(np.array(train_df.iloc[i:i+historical_steps][['Fatalities_log']]).reshape(-1).tolist())
        yf_arr.append(np.array(train_df.iloc[i+historical_steps][['Fatalities_log']]).tolist())
        yf_inverse_arr.append(np.array(train_df.iloc[i+historical_steps][['Fatalities']]).tolist())
        
        grp_arr.append(np.array(train_df.iloc[i+historical_steps][['Geo']]).tolist())
        day_arr.append(np.array(train_df.iloc[i+historical_steps][['Day']]).tolist())


lmm_df = pd.DataFrame(np.hstack((x_arr, xf_arr)), columns=['L1','L2','L3','L4','L5','L6','L7', 'F1','F2','F3','F4','F5','F6','F7'])
lmm_df['ConfirmedCases_log'] = sum(y_arr,[])
lmm_df['ConfirmedCases'] = sum(y_inverse_arr,[])

lmm_df['Fatalities_log'] = sum(yf_arr,[])
lmm_df['Fatalities'] = sum(yf_inverse_arr,[])

lmm_df['Geo'] = sum(grp_arr,[])
lmm_df['Day'] = sum(day_arr,[])
lmm_df.tail()


# In[ ]:


model = sm.MixedLM(endog=lmm_df['ConfirmedCases_log'], exog=lmm_df[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']], exog_re=lmm_df[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']], groups=lmm_df['Geo'])
model_fat = sm.MixedLM(endog=lmm_df['Fatalities_log'], exog=lmm_df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']], exog_re=lmm_df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']],groups=lmm_df['Geo'])
fitted = model.fit()
fitted_fat = model_fat.fit()


# In[ ]:


test_df['Geo'] = test_df['Province/State'] + ',' + test_df['Country/Region']
test_lmm_df = test_df

# For each row
for i in range(len(test_lmm_df)):
    
    day = test_lmm_df.iloc[i].Day
    geo = test_lmm_df.iloc[i]['Geo']
    confirmedCases = 0
    fatalities = 0
    
    # Find previous day
    prev_df = lmm_df[(lmm_df['Day'] == day-1) & (lmm_df['Geo'] == geo)]
    
    # Confirmed Cases 
    if len(prev_df) != 0:
        
        # Generate new time lags 
        temp_l_df = prev_df[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']].iloc[:,1:]        
        temp_l_df.columns = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        temp_l_df['L7'] = prev_df['ConfirmedCases_log']        
        
        # Compute new exog array & Predict
        confirmedCases_log = fitted.predict(exog=np.array(temp_l_df).reshape(-1).tolist())[0]        
        confirmedCases = math.expm1(confirmedCases_log)
        
    # Fatalities 
    if len(prev_df) != 0:
        
        # Generate new time lags 
        temp_f_df = prev_df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']].iloc[:,1:]        
        temp_f_df.columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
        temp_f_df['F7'] = prev_df['Fatalities_log']        
        
        # Compute new exog array & Predict
        fatalities_log = fitted_fat.predict(exog=np.array(temp_f_df).reshape(-1).tolist())[0]        
        fatalities = math.expm1(fatalities_log)
    
    current_df = lmm_df[(lmm_df['Day'] == day) & (lmm_df['Geo'] == geo)]
    
    if len(current_df) != 0:
        lmm_df = lmm_df.drop(current_df.index.tolist())
    lmm_df = lmm_df.append(pd.Series([temp_l_df['L1'].values[0], temp_l_df['L2'].values[0], temp_l_df['L3'].values[0], temp_l_df['L4'].values[0], temp_l_df['L5'].values[0], temp_l_df['L6'].values[0], temp_l_df['L7'].values[0],                                       temp_f_df['F1'].values[0], temp_f_df['F2'].values[0], temp_f_df['F3'].values[0], temp_f_df['F4'].values[0], temp_f_df['F5'].values[0], temp_f_df['F6'].values[0], temp_f_df['F7'].values[0],                                       confirmedCases_log, confirmedCases, fatalities_log, fatalities, geo, day], index=lmm_df.columns ), ignore_index=True)
                                        
lmm_df[(lmm_df['Geo'] == 'Unknown,Korea, South')].tail()


# **Prediction by Geography**

# In[ ]:


plot_df = lmm_df[(lmm_df['Geo'] == 'New South Wales,Australia')]
plt.plot(plot_df['Day'], plot_df['ConfirmedCases'].cumsum())


# In[ ]:


plot_df = lmm_df[(lmm_df['Geo'] == 'Hubei,China')]
plt.plot(plot_df['Day'], plot_df['ConfirmedCases'].cumsum())


# In[ ]:


plot_df = lmm_df[(lmm_df['Geo'] == 'Unknown,Korea, South')]
plt.plot(plot_df['Day'], plot_df['ConfirmedCases'].cumsum())


# In[ ]:


plot_df = lmm_df[(lmm_df['Geo'] == 'Unknown,Italy')]
plt.plot(plot_df['Day'], plot_df['ConfirmedCases'].cumsum())


# In[ ]:


plot_df = lmm_df[(lmm_df['Geo'] == 'Unknown,Iran')]
plt.plot(plot_df['Day'], plot_df['ConfirmedCases'].cumsum())


# In[ ]:


confirmedCases = []
fatalities = [] 
for i in range(len(test_df)):
    
    day = test_lmm_df.iloc[i].Day
    geo = test_lmm_df.iloc[i]['Geo']    
       
    current_df = lmm_df[(lmm_df['Day'] == day) & (lmm_df['Geo'] == geo)]    
    
    if len(current_df) != 0:
        confirmedCases.append(current_df['ConfirmedCases'].values[0])
        fatalities.append(current_df['Fatalities'].values[0])        
    else:
        confirmedCases.append(0)
        fatalities.append(0)    

test_df['ConfirmedCases'] = confirmedCases
test_df['Fatalities'] = fatalities
test_df.head()


# In[ ]:


test_df[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)


# In[ ]:


train_df[(train_df['Country/Region'] == 'Korea, South') & (train_df['Province/State'] == 'Unknown')].tail(20) 
test_df[(test_df['Country/Region'] == 'Korea, South') & (test_df['Province/State'] == 'Unknown') ].head(25)
lmm_df[(lmm_df['Geo'] == 'Unknown,Korea, South')].tail(35)
train_df[(train_df['Country/Region'] == 'Korea, South') & (train_df['Province/State'] == 'Unknown') & (train_df['Day'] == 88)].head()
test_df[(test_df['Country/Region'] == 'Korea, South') & (test_df['Province/State'] == 'Unknown') & (test_df['Day'] == 88)].head()


