#!/usr/bin/env python
# coding: utf-8

# Based on this kernel https://www.kaggle.com/avirl3364/lstm-covid19, credit to @avirl3364.
# I changed it to Bi-LSTM with Mish and Swish and some other minor modifications.
# It's also config. to be tested on public LB, with no leak.
# 

# In[ ]:


get_ipython().system('pip install tensorflow_addons')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense, GRU
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import Normalizer, MinMaxScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import metrics
from sklearn.utils import shuffle
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from tensorflow.keras import losses, models, optimizers

SEED = 321
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


# In[ ]:


df_train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
df_test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

#Public LB eval
#test_date_min = df_test['Date'].min()
#test_date_max = df_test['Date'].max()
#df_train = df_train[df_train['Date']<test_date_min]


# In[ ]:


df_train['Date'].max()


# In[ ]:


from datetime import date

d0 = date(2020,1,23)
d1 = date(2020,5,10)
delta = d1 - d0
number = delta.days+1
number


# In[ ]:




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


X = X.reshape(-1,number,3)
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
partition = number-4-time_step
X_train, Y_train = multivariate_data(X[0,:,:], X[0,:,2], 0, number, time_step)

for i in range(1,3463) :
	X_dummy, Y_dummy = multivariate_data(X[i,:,:], X[i,:,2], 0, number, time_step)
	X_train = np.concatenate((X_train, X_dummy), axis = 0)
	Y_train = np.concatenate((Y_train, Y_dummy), axis = 0)

print(X_train.shape)
print(Y_train.shape)


# In[ ]:


def swishE(x):
   beta = 1.75 #1, 1.5 or 2
   return beta * x * tf.keras.backend.sigmoid(x)

def swish(x):
    return x * tf.keras.backend.sigmoid(x)

def phrishII(x):
    return x*tf.keras.backend.tanh(1.75 * x * tf.keras.backend.sigmoid(x))
def phrishI(x):
    return x*tf.keras.backend.tanh(x * tf.keras.backend.sigmoid(x))

def mish(x):
    return x*tf.keras.backend.tanh(tf.keras.backend.softplus(x))

def gelu_new(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


# In[ ]:





# In[ ]:


del regressor, checkpoint, es, ReduceLROnPlateau,regr1


# In[ ]:


def regr1():
        
    inp = Input(shape = (X_train.shape[1], 3))
    x = tf.keras.layers.Bidirectional(LSTM(units = 32, return_sequences = True,input_shape = (X_train.shape[1], 3)))(inp)
    x = Activation(mish)(x)
    x = Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(LSTM(units = 32, return_sequences = True))(x)
    x = Activation(mish)(x)
    x = Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(GRU(units = 32))(x)
    x = Activation(mish)(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation = swish, name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    #opt1 = tf.keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)#'RMSprop'
    opt = 'RMSprop'
    #opt2 = tfa.optimizers.AdamW(lr=0.001,weight_decay=5.5e-6)
    #opt = tfa.optimizers.Lookahead(tfa.optimizers.AdamW(lr=0.001,weight_decay=0.006), sync_period=5, slow_step_size=0.5)
    #opt = tfa.optimizers.Lookahead(opt1, sync_period=5, slow_step_size=0.5)
    #opt = tfa.optimizers.SWA(opt)

    model.compile(optimizer = opt, loss = 'mean_squared_error')
    return model

regressor = regr1()
regressor.summary()
tf.keras.utils.plot_model(
    regressor, to_file='model1.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=True, dpi=96
)


# In[ ]:


checkpoint = tf.keras.callbacks.ModelCheckpoint("model_1_.h5".format(i), monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min')
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 64, validation_split=0.05, shuffle=True,callbacks=[checkpoint,es])
regressor.load_weights("model_1_.h5")


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
st_index = number - time_step
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
print(predicted_test)


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
df


# In[ ]:


s1 = df['Predicted_Results']
l1 = s1.tolist()
l2 = [i*2 for i in l1]
l1 = [i*0.95 for i in l2]
l2 = [round(i) for i in l1]
df2 = pd.DataFrame(l2)

my_list = [*range(3,935011,6)]
df2['Id_1'] = my_list
df2.rename(columns = {0 : 'Predicted_Results'}, inplace = True)

s1 = df['Predicted_Results']
l1 = s1.tolist()
l2 = [i*2 for i in l1]
l1 = [i*0.05 for i in l2]
l2 = [round(i) for i in l1]
df3 = pd.DataFrame(l2)

my_list = [*range(1,935011,6)]
df3['Id_1'] = my_list
df3.rename(columns = {0 : 'Predicted_Results'}, inplace = True)

result_confirmed_cases = pd.concat([df, df2, df3])
result_confirmed_cases.sort_values(by = ['Id_1'], inplace = True)
result_confirmed_cases


# In[ ]:


train_confirm1 = train_data[train_data['Target'] == 'Fatalities']
train_confirm1 = train_confirm1.drop(['Target'], axis = 1)
train_confirm1['TargetValue'] = np.where(train_confirm1['TargetValue'] <=0, 0, train_confirm1['TargetValue'])
#print(train_confirm)

X1 = train_confirm1.iloc[:, 0: 4].to_numpy()

# MinMaxScaling

X1[:, 0:1] = sc_pop.fit_transform(X1[:, 0:1])
X1[:, 2:3] = sc_tg.fit_transform(X1[:, 2:3])


X1 = X1.reshape(-1,number,3)
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

X_train1, Y_train1 = multivariate_data(X1[0,:,:], X1[0,:,2], 0, number, time_step)

for i in range(1,3463) :
	X_dummy, Y_dummy = multivariate_data(X1[i,:,:], X1[i,:,2], 0, number, time_step)
	X_train1 = np.concatenate((X_train1, X_dummy), axis = 0)
	Y_train1 = np.concatenate((Y_train1, Y_dummy), axis = 0)

print(X_train1.shape)
print(Y_train1.shape)


# In[ ]:


del regr1,regressor, checkpoint, es, ReduceLROnPlateau


# In[ ]:


def regr2():
        
    inp = Input(shape = (X_train.shape[1], 3))
    x = tf.keras.layers.Bidirectional(LSTM(units = 32, return_sequences = True,input_shape = (X_train.shape[1], 3)))(inp)
    x = Activation(mish)(x)
    x = Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(LSTM(units = 32, return_sequences = True))(x)
    x = Activation(mish)(x)
    x = Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(GRU(units = 32))(x)
    x = Activation(mish)(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation = swish, name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    #opt1 = tf.keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)#'RMSprop'
    opt = 'RMSprop'
    #opt2 = tfa.optimizers.AdamW(lr=0.001,weight_decay=5.5e-6)
    #opt = tfa.optimizers.Lookahead(tfa.optimizers.AdamW(lr=0.001,weight_decay=0.006), sync_period=5, slow_step_size=0.5)
    #opt = tfa.optimizers.Lookahead(opt1, sync_period=5, slow_step_size=0.5)
    #opt = tfa.optimizers.SWA(opt)

    model.compile(optimizer = opt, loss = 'mean_squared_error')
    return model

regressor1 = regr2()
regressor1.summary()
tf.keras.utils.plot_model(
    regressor1, to_file='model2.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=True, dpi=96
)


# In[ ]:


checkpoint = tf.keras.callbacks.ModelCheckpoint("model_2_.h5".format(i), monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min')
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
regressor1.fit(X_train, Y_train, epochs = 100, batch_size = 64, validation_split=0.05, shuffle=True,callbacks=[checkpoint,es])
regressor1.load_weights("model_2_.h5")


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
st_index = number - time_step
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
print(predicted_test)


# In[ ]:


pred_test_flat = predicted_test.flatten()
pred_test_flat = pred_test_flat.astype(int)
df4 = pd.DataFrame(pred_test_flat)
my_list = [*range(5,935011,6)]
df4['Id_1'] = my_list
df4.rename(columns = {0 : 'Predicted_Results'}, inplace = True)
df4['Predicted_Results'] = np.where(df4['Predicted_Results'] <0, 0, df4['Predicted_Results'])
df4


# In[ ]:


s1 = df4['Predicted_Results']
l1 = s1.tolist()
l2 = [i*2 for i in l1]
l1 = [i*0.95 for i in l2]
l2 = [round(i) for i in l1]
df5 = pd.DataFrame(l2)

my_list = [*range(6,935011,6)]
df5['Id_1'] = my_list
df5.rename(columns = {0 : 'Predicted_Results'}, inplace = True)

s1 = df4['Predicted_Results']
l1 = s1.tolist()
l2 = [i*2 for i in l1]
l1 = [i*0.05 for i in l2]
l2 = [round(i) for i in l1]
df6 = pd.DataFrame(l2)

my_list = [*range(4,935011,6)]
df6['Id_1'] = my_list
df6.rename(columns = {0 : 'Predicted_Results'}, inplace = True)

result_fatalities = pd.concat([df4, df5, df6])
result_fatalities.sort_values(by = ['Id_1'], inplace = True)
result_fatalities


# In[ ]:


result_total = pd.concat([result_confirmed_cases, result_fatalities])
result_total.sort_values(by = ['Id_1'], inplace = True)
result_total


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

