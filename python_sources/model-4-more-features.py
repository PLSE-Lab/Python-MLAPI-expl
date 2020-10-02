#!/usr/bin/env python
# coding: utf-8

# # CNN-LSTM without features

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


import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.metrics import mean_absolute_error


# In[ ]:


train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32}).values


# In[ ]:


train [0:150000, 0 ] .mean(axis=0)


# In[ ]:


# pandas doesn't show us all the decimals
pd.options.display.precision = 15


# In[ ]:


rows = 150_000
segments = int(np.floor(train.shape[0] / rows))
print('train.shape',train.shape)
segments


# # for data set without features

# In[ ]:


def create_X(x, last_index=None, n_steps=150000, step_length=1):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0

    
    # Reshaping and approximate standardization with mean 5 and std 3.
    temp = (x[(int(last_index) - n_steps * step_length):int(last_index)].reshape(n_steps,step_length ).astype(np.float32) - 5 ) / 3   
    # convert (150000) to [150 1000 ]
    # then extract feature from each row of length 1000. so total 150 
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    
    return temp


# [](http://)

# # Creating Training Data without features

# In[ ]:


# Query "create_X" to figure out the number of features
n_features = create_X(train [0:150000,0]).shape
print("Output segment shape", n_features)     # 18 features each row of segment ie 150x18 features of 150000 chunk input

n_steps=150000
step_length=1
maxsize=train .shape[0]
seg = int(np.floor(maxsize / (n_steps*step_length))) 
batch_size = seg-1   # (4193,) 
xx=350


##############################################################################################
rows_initialize = np.zeros((seg), dtype=float)
print(rows_initialize.shape)

for seg1 in tqdm(range(1,seg)) :      # for loop from 1 to 4194 segment value
    rows_initialize [seg1] = seg1 * (n_steps*step_length) 

rows=np.delete(rows_initialize,0)    # (4193,)

print(rows.shape)

########################################################################################
batch_size=batch_size-xx    # training data
#batch_size=xx              # validation data
split_point=xx
second_earthquake = rows[xx]



##########################################################################################

if batch_size < 1000  :    # validation set 
               rows_1 = rows[:split_point+1]    #  0:350 
        
if batch_size > 1000 :   # training set
               rows_1 = rows[split_point+1 :]    # (351,) ie 351:4193    
            

       
    # Initialize feature matrices and targets
samples_tr= np.zeros((rows_1.shape[0], n_features[0], n_features[1]), dtype=float)   #  for validation (350,150000)  for training ( 3842, 150000) 
targets_tr = np.zeros(rows_1.shape[0], )    # (16,)  for validation (350)    for training ( 3843)
        
for j, row in enumerate(rows_1):             # 16 for validation (350)    for training ( 3843)
    samples_tr[j] = create_X(train[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
    targets_tr[j] = train[int(row - 1), 1]         
    
    
################################################################################################

print('samples_tr shape', samples_tr.shape)
print('targets_tr shape', targets_tr.shape)

samples_tr.shape


# # Creating Validation data without features

# In[ ]:


#batch_size=batch_size-xx    # training data
batch_size=xx              # validation data
split_point=xx
second_earthquake = rows[xx]

##########################################################################################

if batch_size < 1000  :    # validation set 
               rows_1 = rows[:split_point+1]    #  0:350 
        
if batch_size > 1000 :   # training set
               rows_1 = rows[split_point+1 :]    # (351,) ie 351:4193    
            

       
    # Initialize feature matrices and targets
samples_vd= np.zeros((rows_1.shape[0], n_features[0], n_features[1]), dtype=float)    #  for validation (350,150000)  for training ( 3842, 150000) 
targets_vd = np.zeros(rows_1.shape[0], )    # (16,)  for validation (350)    for training ( 3843)
        
for j, row in enumerate(rows_1):             # 16 for validation (350)    for training ( 3843)
    samples_vd[j] = create_X(train[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
    targets_vd[j] = train[int(row - 1), 1]         
    
    
################################################################################################

    
print('samples_tr shape', samples_tr.shape)
print('targets_tr shape',targets_tr.shape) 
    
print('samples_vd shape', samples_vd.shape)
print('targets_vd shape',targets_vd.shape)  
#print('rows_1 shape',rows_1.shape[0])
    


# # Define the model

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, SimpleRNN, LSTM ,  Dropout, Activation, Flatten, Input, Conv1D, MaxPooling1D, Reshape,  Conv2D, MaxPooling2D, Reshape, Flatten
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.utils import plot_model


# Shared Input Layer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


# In[ ]:


# VGG Conv2D

i = (n_features[0],n_features[1])
#i = (X_train.shape[1])
i
model = Sequential ()

model.add(Reshape((1500,100,1), input_shape=i))
model.add(Conv2D(32, (3, 3), strides = (2,2), kernel_initializer='he_normal', padding='same'))
model.add(LeakyReLU(0.1))
model.add(Conv2D(32, (3, 3), strides = (2,2), kernel_initializer='he_normal', padding='same'))
model.add(LeakyReLU(0.1))
model.add(MaxPooling2D())
model.add(Dropout(0.3))


model.add(Conv2D(64, (3, 3), strides = (2,2), kernel_initializer='he_normal', padding='same'))
model.add(LeakyReLU(0.1))
model.add(Conv2D(64, (3, 3), strides = (2,2), kernel_initializer='he_normal', padding='same'))
model.add(LeakyReLU(0.1))
model.add(MaxPooling2D())
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Softmax())
model.add(Reshape((1472,1)))

#model.add(Reshape((23,60)))
#model.add(Flatten())

model.add(LSTM(16,  return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16))
model.add(Dropout(0.2))

#model.add(Dense(60))
#model.add(Dense(30))
model.add(Dense(1))

model.summary()
#plot_model(model, to_file='model.png')


# In[ ]:


# ALEX NET Conv 2d

i = (1500,100,1)
#i = (X_train.shape[1])
i
model = Sequential ()

#model.add(Reshape((1500,100,1), input_shape=i))
#model.add(Conv2D(32, (3, 3), strides = (2,2), kernel_initializer='he_normal', padding='same'))
#model.add(LeakyReLU(0.1))


# input layer
visible = Input(shape=i)
# first feature extractor
conv1 = Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
pool1 = MaxPooling2D()(conv1)
flat1 = Flatten()(pool1)
# second feature extractor
conv2 = Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
pool2 = MaxPooling2D()(conv2)
flat2 = Flatten()(pool2)
# merge feature extractors
merge1 = concatenate([pool1, pool2])

# interpretation layer
#merge1_1 =Reshape(( 19199744,1))(merge1)

# first feature extractor
conv3 = Conv2D(16, kernel_size=(3,3), activation='relu')(merge1)
pool3 = MaxPooling2D()(conv3)
flat3 = Flatten()(pool3)
# second feature extractor
conv4 = Conv2D(16, kernel_size=(3,3), activation='relu')(merge1)
pool4 = MaxPooling2D()(conv4)
flat4 = Flatten()(pool4)
# merge feature extractors
merge2 = concatenate([flat3 , flat4] )

merge2_2 =Reshape(( 274528,1))(merge2)


## interpretation layer
#hidden1 = Dense(100, activation='relu')(merge1)
## prediction output
#output = Dense(1, activation='relu')(hidden1)


LSTM1= LSTM(32,  return_sequences=True)(merge2_2)
#Dropout=Dropout(0.2)(LSTM1)
LSTM2= LSTM(32)(LSTM1)
#Dropout=Dropout(0.2)(LSTM2)

#output1=Dense(62)(LSTM2)
output2=Dense(1)(LSTM2)




model = Model(inputs=visible, outputs=output2)
# summarize layers
print(model.summary())
# plot graph
#plot_model(model, to_file='shared_input_layer.png')


# In[ ]:


## CNN combined with LSTM Model 
i = (n_features[0],n_features[1])
model = Sequential ()

model.add(Conv1D (kernel_size = (3), filters = 32, strides=2, input_shape=i, kernel_initializer='he_normal', activation='relu')) 
#model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D (kernel_size = (3), filters = 16, strides=2, kernel_initializer='he_normal', activation='relu')) 
#model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling1D())

model.add(Conv1D (kernel_size = (3), filters = 8, strides=2, kernel_initializer='he_normal', activation='relu')) 
#model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling1D())


#model.add(Flatten())
#model.add(Dense (250, activation='relu', kernel_initializer='he_normal'))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
    
model.add(LSTM(256,  return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))





model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


model.summary()
##1st model
#model.add(Conv1D(5, 3, activation='relu', input_shape= i))
#model.add(MaxPooling1D(2))
#model.add(LSTM(50,  return_sequences=True))
#model.add(LSTM(10))
#model.add(Dense(240))
#model.add(Dense(120))
#model.add(Dense(60))
#model.add(Dense(30))
#model.add(Dense(1))

##2nd model 

#model.add(Conv1D(16, 3, activation='relu', input_shape= i))
#model.add(MaxPooling1D(2))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(MaxPooling1D(2))
#model.add(Conv1D(16, 3, activation='relu'))
#model.add(MaxPooling1D(2))
#model.add(Dropout(0.1))
#model.add(LSTM(48,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
#model.add(LSTM(20,dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
#model.add(Dense(1, activation='linear'))
 


# In[ ]:


# VGG 16

i = (n_features[0],n_features[1])
#i = (X_train.shape[1])
i
model = Sequential ()

#model.add(Reshape((1500,100,1), input_shape=i))
model.add(Conv1D(16, 3, strides = 2, kernel_initializer='he_normal',input_shape=i, padding='same'))
model.add(LeakyReLU(0.1))
model.add(Conv1D(16, 3, strides = 2, padding='same'))
model.add(LeakyReLU(0.1))
model.add(MaxPooling1D())
#model.add(Dropout(0.3))

model.add(Conv1D(32, 3, strides = 2, padding='same'))
model.add(LeakyReLU(0.1))
model.add(Conv1D(32, 3, strides = 2, padding='same'))
model.add(LeakyReLU(0.1))
model.add(MaxPooling1D())
#model.add(Dropout(0.3))

model.add(Conv1D(64, 3, strides = 2, padding='same'))
model.add(LeakyReLU(0.1))
model.add(Conv1D(64, 3, strides = 2, padding='same'))
model.add(LeakyReLU(0.1))
model.add(MaxPooling1D())
#model.add(Dropout(0.3))

#model.add(Conv1D(128, 3, strides = 2, kernel_initializer='he_normal', padding='same'))
#model.add(LeakyReLU(0.1))
#model.add(Conv1D(128, 3, strides = 2, kernel_initializer='he_normal', padding='same'))
#model.add(LeakyReLU(0.1))
#model.add(MaxPooling1D())

model.add(Flatten())
model.add(Softmax())
model.add(Reshape((18752,1)))
#model.add(Dense(60))
#model.add(Dense(1))


#model.add(Conv2D(30, (3, 3), strides = (2,2), kernel_initializer='he_normal', padding='same'))
#model.add(LeakyReLU(0.2))
#model.add(MaxPooling2D())
#model.add(Dropout(0.3))

#model.add(Reshape((23,60)))
#model.add(Flatten())

model.add(LSTM(32,  return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))

#model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(1))

model.summary()


# In[ ]:


# ALEX NET

i = (n_features[0],n_features[1])
#i = (X_train.shape[1])
i
model = Sequential ()

# input layer
visible = Input(shape=i)
# first feature extractor
conv1 = Conv1D(128, kernel_size=3, activation='relu')(visible)
pool1 = MaxPooling1D()(conv1)
flat1 = Flatten()(pool1)
# second feature extractor
conv2 = Conv1D(128, kernel_size=3, activation='relu')(visible)
pool2 = MaxPooling1D()(conv2)
flat2 = Flatten()(pool2)
# merge feature extractors
merge1 = concatenate([flat1, flat2])
# interpretation layer
merge1_1 =Reshape(( 19199744,1))(merge1)

# first feature extractor
conv3 = Conv1D(64, kernel_size=3, activation='relu')(merge1_1)
pool3 = MaxPooling1D()(conv3)
flat3 = Flatten()(pool3)
# second feature extractor
conv4 = Conv1D(64, kernel_size=3, activation='relu')(merge1_1)
pool4 = MaxPooling1D()(conv4)
flat4 = Flatten()(pool4)
# merge feature extractors
merge2 = concatenate([flat3 , flat4] )
merge2_2 =Reshape(( 1228783488,1))(merge2)


## interpretation layer
#hidden1 = Dense(100, activation='relu')(merge1)
## prediction output
#output = Dense(1, activation='relu')(hidden1)


LSTM1= LSTM(128,  return_sequences=True)(merge2_2)
#Dropout=Dropout(0.2)(LSTM1)
LSTM2= LSTM(128)(LSTM1)
#Dropout=Dropout(0.2)(LSTM2)

#output1=Dense(62)(LSTM2)
output2=Dense(1)(LSTM2)




model = Model(inputs=visible, outputs=output2)
# summarize layers
print(model.summary())
# plot graph
#plot_model(model, to_file='shared_input_layer.png')


# # Compile and fit model

# In[ ]:


# vggg 16
import keras
from keras.optimizers import RMSprop
opt = keras.optimizers.adam(lr=.005)

model.compile(loss="mae",
              optimizer=opt, metrics=['mean_absolute_error'])
             # metrics=['accuracy'])


batch_size = 64 # mini-batch with 32 examples
epochs = 30
history = model.fit(
    samples_tr, targets_tr,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
   validation_data=(samples_vd  ,targets_vd ))


# In[ ]:


# alex net
import keras
from keras.optimizers import RMSprop
opt = keras.optimizers.adam(lr=.005)

model.compile(loss="mae",
              optimizer=opt, metrics=['mean_absolute_error'])
             # metrics=['accuracy'])


batch_size = 32 # mini-batch with 32 examples
epochs = 50
history = model.fit(
    samples_tr, targets_tr,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
   validation_data=(samples_vd  ,targets_vd ))


# # Load submission file
# 

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})


# ## Prepare submission data
# Load each test data, create the feature matrix, get numeric prediction

# In[ ]:


for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))

submission.head()


# ## Save submission file

# In[ ]:


submission.to_csv('submission_without features new model VGG 16.csv')


# # OTHER TECHNIQUES

# In[ ]:


x.mean()  


# In[ ]:


X_train


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


# In[ ]:


X_train_scaled


# In[ ]:


y_train.values


# In[ ]:


y_train.values.flatten()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

## Train the  Model
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train_scaled, y_train.values.flatten())      # .fit used for training
y_pred = model.predict(X_train_scaled)


# In[ ]:


# number support vector regressor

# svm = NuSVR()
# svm.fit(X_train_scaled, y_train.values.flatten())
# y_pred = svm.predict(X_train_scaled)


# In[ ]:


y_pred.shape


# In[ ]:


plt.figure(figsize=(6, 6))
plt.scatter(y_train.values.flatten(), y_pred)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.show()


# In[ ]:


score = mean_absolute_error(y_train.values.flatten(), y_pred)
print(f'Score: {score:0.3f}')


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')


# In[ ]:


X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)


# In[ ]:


for seg_id in X_test.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()


# In[ ]:


X_test_scaled = scaler.transform(X_test)
submission['time_to_failure'] = svm.predict(X_test_scaled)
submission.to_csv('submission.csv')

