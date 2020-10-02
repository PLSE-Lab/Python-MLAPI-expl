#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import Normalizer, MinMaxScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import metrics
from sklearn.utils import shuffle

####atom://teletype/portal/3f155216-29d7-4279-882f-d6aec0f4c85b

df_train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
df_test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

train_data = df_train.drop(
    ['Id', 'County', 'Province_State', 'Country_Region'], axis=1)
#test_data = df_test.drop(
#    ['County', 'Province_State', 'Country_Region'], axis=1)
train_data.set_index('Date', inplace=True)
#test_data.set_index('Date', inplace=True)

train_confirm = train_data[train_data['Target'] == 'ConfirmedCases']
train_confirm = train_confirm.drop(['Target'], axis = 1)
train_confirm['TargetValue'] = np.where(train_confirm['TargetValue'] <=0, 0, train_confirm['TargetValue'])
#print(train_confirm)

X = train_confirm.iloc[:, 0: 4].to_numpy()
Y = train_data.iloc[:, 4: 5].to_numpy()

# MinMaxScaling

sc_pop = MinMaxScaler(feature_range=(0, 1))
sc_tg = MinMaxScaler(feature_range=(0, 1))
X[:, 0:1] = sc_pop.fit_transform(X[:, 0:1])
X[:, 2:3] = sc_tg.fit_transform(X[:, 2:3])


X = X.reshape(-1,108,3)
print(X.shape)
#print(df_train.dtypes)
#print(X)

def multivariate_data(dataset, target, start_index, end_index, time_step) :
	data=list()
	label =list()

	start_index = start_index + time_step
	for i in range(start_index, end_index) :
		indices = range(i-time_step, i)
		data.append(dataset[indices])
		label.append(target[i])

	return np.array(data), np.array(label)

time_step = 40
partition = 108-4-time_step
X_train, Y_train = multivariate_data(X[0,:,:], X[0,:,2], 0, 108, time_step)

for i in range(1,3463) :
	X_dummy, Y_dummy = multivariate_data(X[i,:,:], X[i,:,2], 0, 108, time_step)
	X_train = np.concatenate((X_train, X_dummy), axis = 0)
	Y_train = np.concatenate((Y_train, Y_dummy), axis = 0)


# In[ ]:


regressor = Sequential()

regressor.add(LSTM(units = 16, return_sequences = True, input_shape = (X_train.shape[1], 3)))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 16, return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 16, return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 16))
regressor.add(Dropout(0.3))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')
regressor.fit(X_train, Y_train, epochs = 7, batch_size = 32)


# In[ ]:


test_data = df_test.drop(
    ['ForecastId','County', 'Province_State', 'Country_Region'], axis=1)
test_data.set_index('Date', inplace=True)
test_confirm = test_data[test_data['Target'] == 'ConfirmedCases']
test_confirm = test_confirm.drop(['Target'], axis = 1)

x_test = test_confirm.iloc[:, 0: 4].values
(a,b) = x_test.shape
x_test[:, 0:1] = sc_pop.fit_transform(x_test[:, 0:1])

X_modify = np.zeros(shape = (a,b+1))
X_modify[:,:-1] = x_test
X_modify = X_modify.reshape(-1,45,3)

X_t1 = X_modify[0,:,:]
X_t2 = X[0,:,:]
X_t3 = np.concatenate((X_t2,X_t1), axis = 0)
st_index = 107 - time_step
X_t3 = X_t3[st_index:160,:]
X_test1, Y_d1 = multivariate_data(X_t3, X_t3[:,2], 0, time_step + 45, time_step)
print(X_test1.shape)

for i in range (1,3463) :
	x_t1 = X_modify[i,:,:]
	X_t2 = X[i,:,:]
	X_t3 = np.concatenate((X_t2,X_t1), axis = 0)
	X_t3 = X_t3[st_index:160,:]
	X_test_dummy, Y_d1 = multivariate_data(X_t3, X_t3[:,2], 0, time_step + 45, time_step)
	X_test1 = np.concatenate((X_test1, X_test_dummy), axis =0)

predicted_test = regressor.predict(X_test1)
predicted_test = sc_tg.inverse_transform(predicted_test)


# In[ ]:


pred_test_flat = predicted_test.flatten()
pred_test_flat = pred_test_flat.astype(int)


# In[ ]:


pred_test_flat.shape


# In[ ]:


df = pd.DataFrame(pred_test_flat)
my_list = [*range(2,935011,6)]
df['Id_1'] = my_list
df.rename(columns = {0 : 'Predicted_Results'}, inplace = True)
df['Predicted_Results'] = np.where(df['Predicted_Results'] <0, 0, df['Predicted_Results'])


# In[ ]:


s1 = df['Predicted_Results']
l1 = s1.tolist()
l2 = [i for i in l1]
l1 = [i for i in l2]
l2 = [round(i) for i in l1]
df2 = pd.DataFrame(l2)

my_list = [*range(3,935011,6)]
df2['Id_1'] = my_list
df2.rename(columns = {0 : 'Predicted_Results'}, inplace = True)

s1 = df['Predicted_Results']
l1 = s1.tolist()
l2 = [i for i in l1]
l1 = [i for i in l2]
l2 = [round(i) for i in l1]
df3 = pd.DataFrame(l2)

my_list = [*range(1,935011,6)]
df3['Id_1'] = my_list
df3.rename(columns = {0 : 'Predicted_Results'}, inplace = True)

result_confirmed_cases = pd.concat([df, df2, df3])
result_confirmed_cases.sort_values(by = ['Id_1'], inplace = True)


# In[ ]:


train_confirm1 = train_data[train_data['Target'] == 'Fatalities']
train_confirm1 = train_confirm1.drop(['Target'], axis = 1)
train_confirm1['TargetValue'] = np.where(train_confirm1['TargetValue'] <=0, 0, train_confirm1['TargetValue'])
#print(train_confirm)

X1 = train_confirm1.iloc[:, 0: 4].to_numpy()

# MinMaxScaling

X1[:, 0:1] = sc_pop.fit_transform(X1[:, 0:1])
X1[:, 2:3] = sc_tg.fit_transform(X1[:, 2:3])


X1 = X1.reshape(-1,108,3)
print(X1.shape)
#print(df_train.dtypes)
#print(X)

def multivariate_data(dataset, target, start_index, end_index, time_step) :
	data=list()
	label =list()

	start_index = start_index + time_step
	for i in range(start_index, end_index) :
		indices = range(i-time_step, i)
		data.append(dataset[indices])
		label.append(target[i])

	return np.array(data), np.array(label)

X_train1, Y_train1 = multivariate_data(X1[0,:,:], X1[0,:,2], 0, 108, time_step)

for i in range(1,3463) :
	X_dummy, Y_dummy = multivariate_data(X1[i,:,:], X1[i,:,2], 0, 108, time_step)
	X_train1 = np.concatenate((X_train1, X_dummy), axis = 0)
	Y_train1 = np.concatenate((Y_train1, Y_dummy), axis = 0)


# In[ ]:


regressor1 = Sequential()

regressor1.add(LSTM(units = 16, return_sequences = True, input_shape = (X_train1.shape[1], 3)))
regressor1.add(Dropout(0.3))
regressor1.add(LSTM(units = 16, return_sequences = True))
regressor1.add(Dropout(0.3))
regressor1.add(LSTM(units = 16, return_sequences = True))
regressor1.add(Dropout(0.3))
regressor1.add(LSTM(units = 16))
regressor1.add(Dropout(0.3))
regressor1.add(Dense(units = 1))
regressor1.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')
regressor1.fit(X_train1, Y_train1, epochs = 7, batch_size = 32)


# In[ ]:


test_data = df_test.drop(
    ['ForecastId','County', 'Province_State', 'Country_Region'], axis=1)
test_data.set_index('Date', inplace=True)
test_confirm = test_data[test_data['Target'] == 'Fatalities']
test_confirm = test_confirm.drop(['Target'], axis = 1)

x_test = test_confirm.iloc[:, 0: 4].values
(a,b) = x_test.shape
x_test[:, 0:1] = sc_pop.fit_transform(x_test[:, 0:1])

X_modify = np.zeros(shape = (a,b+1))
X_modify[:,:-1] = x_test
X_modify = X_modify.reshape(-1,45,3)

X_t1 = X_modify[0,:,:]
X_t2 = X[0,:,:]
X_t3 = np.concatenate((X_t2,X_t1), axis = 0)
st_index = 107 - time_step
X_t3 = X_t3[st_index:160,:]
X_test1, Y_d1 = multivariate_data(X_t3, X_t3[:,2], 0, time_step + 45, time_step)

for i in range (1,3463) :
	x_t1 = X_modify[i,:,:]
	X_t2 = X[i,:,:]
	X_t3 = np.concatenate((X_t2,X_t1), axis = 0)
	X_t3 = X_t3[st_index:160,:]
	X_test_dummy, Y_d1 = multivariate_data(X_t3, X_t3[:,2], 0, time_step + 45, time_step)
	X_test1 = np.concatenate((X_test1, X_test_dummy), axis =0)

predicted_test = regressor1.predict(X_test1)
predicted_test = sc_tg.inverse_transform(predicted_test)


# In[ ]:


pred_test_flat = predicted_test.flatten()
pred_test_flat = pred_test_flat.astype(int)
df4 = pd.DataFrame(pred_test_flat)
my_list = [*range(5,935011,6)]
df4['Id_1'] = my_list
df4.rename(columns = {0 : 'Predicted_Results'}, inplace = True)
df4['Predicted_Results'] = np.where(df4['Predicted_Results'] <0, 0, df4['Predicted_Results'])


# In[ ]:


s1 = df4['Predicted_Results']
l1 = s1.tolist()
l2 = [i for i in l1]
l1 = [i for i in l2]
l2 = [round(i) for i in l1]
df5 = pd.DataFrame(l2)

my_list = [*range(6,935011,6)]
df5['Id_1'] = my_list
df5.rename(columns = {0 : 'Predicted_Results'}, inplace = True)

s1 = df4['Predicted_Results']
l1 = s1.tolist()
l2 = [i for i in l1]
l1 = [i for i in l2]
l2 = [round(i) for i in l1]
df6 = pd.DataFrame(l2)

my_list = [*range(4,935011,6)]
df6['Id_1'] = my_list
df6.rename(columns = {0 : 'Predicted_Results'}, inplace = True)

result_fatalities = pd.concat([df4, df5, df6])
result_fatalities.sort_values(by = ['Id_1'], inplace = True)


# In[ ]:


result_total = pd.concat([result_confirmed_cases, result_fatalities])
result_total.sort_values(by = ['Id_1'], inplace = True)


# In[ ]:


df_submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
my_list = [*range(1,935011)]
df_submission['Id'] = my_list 
final_result = pd.merge(df_submission,result_total, left_on = 'Id', right_on ='Id_1', how = 'inner')
final_result.drop(['TargetValue','Id','Id_1'], axis =1, inplace = True)
final_result.rename({'Predicted_Results' : 'TargetValue'}, axis = 1, inplace = True)
final_result


# In[ ]:


final_result.to_csv('submission.csv', index = False)

