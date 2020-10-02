#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from scipy.signal import savgol_filter


# In[ ]:


data_raw = pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')
data_copy = data_raw.copy(deep=True)


# In[ ]:


data_raw.head()


# In[ ]:


# The task is to predict the 4 target variables ('pm', 'stator_yoke', 'stator_winding', 'stator_tooth') for
# for entries with 'profile_id' equal to 65 and 72 using data with smaller values of 'profile_id'.
# For the purposes of this notebook target variables only for entries with 'profile_id' = 65 will be predicted

target_variables = ['pm', 'stator_yoke', 'stator_winding', 'stator_tooth']
drop_cols = ['profile_id']


# In[ ]:


### 'torque' column is dropped according to the task 

data_copy.drop(columns=['torque'], inplace=True)

### 'profile_id' less than 65 for the training set

profile_list = np.array([i for i in data_raw['profile_id'].unique() if i < 65])


# In[ ]:


profile_list


# In[ ]:


cols = [item for item in list(data_copy.columns) if item not in drop_cols + target_variables]


# In[ ]:


cols


# In[ ]:


# apply MinMaxScaler in order to improve performance of MLPRegressor

scaler = MinMaxScaler()
data_copy[cols] = scaler.fit_transform(data_copy[cols])


# In[ ]:


data_copy[cols]


# In[ ]:


data_train = data_copy[data_copy['profile_id'].isin(profile_list)]

data_test = data_copy[data_copy['profile_id'] == 65]


# In[ ]:


data_train


# In[ ]:


# Take a look at how predictor variables behave 
for profile_id in [4, 11, 30, 45, 52]:
    print('id: ', profile_id)
    plt.figure(figsize=(26, 3))
    temp_data = data_train[data_train['profile_id'] == profile_id]
    i=1
    for col in cols: 
        sub = plt.subplot(1,7,i)
        i+=1
        plt.plot(temp_data[col])
        sub.set(xlabel='index', ylabel=col)
    plt.show()


# In[ ]:


# Take a look at how target variables behave  
for profile_id in [4, 11, 30, 45, 52]:
    print('id: ', profile_id)
    plt.figure(figsize=(22, 3))
    temp_data = data_train[data_train['profile_id'] == profile_id]
    i=1
    for col in target_variables: 
        sub = plt.subplot(1,4,i)
        i+=1
        plt.plot(temp_data[col])
        sub.set(xlabel='index', ylabel=col)
    plt.show()


# 'stator_yoke', 'stator_winding' & 'stator_tooth' appear to change similarly, but in different bounds of values (e.g. with id=30 stator_winding values is between -2 and 1.5, while stator_tooth is between -2 and 0.5).
# 'pm' variable seems to behave differently from the rest, which might complicate the prediction and warrant a use of a ChainRegressor: predict the 'stator_...' variables first, and include them in the prediction for 'pm'

# In[ ]:


corr = data_train.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


X_train = data_train.drop(columns = target_variables + drop_cols)
Y_train = data_train[target_variables]


# In[ ]:


X_train.head()


# In[ ]:


reg = MLPRegressor(hidden_layer_sizes=(49), max_iter=2000, activation='tanh', verbose=False, random_state=1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'reg.fit(X_train, Y_train)')


# In[ ]:


X_test = data_test.drop(columns = target_variables + drop_cols)
Y_test_actual = data_test[target_variables]


# In[ ]:


Y_test_prediction = reg.predict(X_test)  


# In[ ]:


prediction_df = pd.DataFrame(Y_test_prediction)


# In[ ]:


prediction_df.head()


# In[ ]:


prediction_df = prediction_df.rename(columns={0: 'pm_pred', 1: 'stator_yoke_pred', 
                                              2: 'stator_winding_pred', 3: 'stator_tooth_pred'})


# In[ ]:


Y_test_actual


# In[ ]:


prediction_df = Y_test_actual.reset_index().merge(prediction_df, left_index = True, right_index=True).set_index('index')


# In[ ]:


plt.figure(figsize=(30, 7))
print('actual')

for idx,col in enumerate(target_variables): 
    plt.subplot(1, 4, idx+1)
    plt.plot(prediction_df[col + '_pred'])
    plt.plot(prediction_df[col], color='red')
    plt.legend(loc="upper right")
plt.show()


# 
# 
# In order to smooth the predicted values, a Savitzky-Golay filter has been applied:
# https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

# In[ ]:


predicted_cols = [col + '_pred' for col in target_variables] 
smoothed_cols = [col + '_smoothed' for col in predicted_cols] 


# In[ ]:


for column in predicted_cols:
    prediction_df[column+'_smoothed'] = savgol_filter(prediction_df[column], 
                                                      501, 1)


# In[ ]:


prediction_df.head()


# In[ ]:


plt.figure(figsize=(30, 7))
print('actual')

for idx,col in enumerate(target_variables): 
    plt.subplot(1, 4, idx+1)
    plt.plot(prediction_df[col + '_pred_smoothed'], label=col+': Predicted value')
    plt.plot(prediction_df[col], color='red', lw=1 , label=col+': Actual value')
    plt.legend(loc="upper right")
plt.show()


# In[ ]:


for column in target_variables:
    print('column: ', column)
    print('no smooth: ', mean_squared_error(Y_test_actual[column], prediction_df[column+'_pred'].to_numpy()))
    print('smooth: ', mean_squared_error(Y_test_actual[column], prediction_df[column+'_pred_smoothed'].to_numpy()), '\r\n')
    


# the predictions for the "stator_..." variables appear to be somewhat precise, 
# however, the slope formed by "pm" values does not seem to be predicted accurately. 
# A possible solution to that might be to restructure the neural network by adding more hidden layers, however, that is going to increase the computational comlexity and, therefore, training time

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




