#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# * Adding multiple target values spanning multiple segments.
# 
# # Reference
# * [RNN starter for huge time series](https://www.kaggle.com/mayer79/rnn-starter-for-huge-time-series) 
# * [RNN starter notebook](https://www.kaggle.com/devilears/rnn-starter-kernel-with-notebook)
# * [Intro to RNN with LSTM and GRU](https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru)
# * [Wavelet denoising](https://www.kaggle.com/tarunpaparaju/lanl-earthquake-prediction-signal-denoising)

# In[1]:


import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.linear_model import Lasso, Ridge

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from numpy.random import seed
seed(802)

import warnings
warnings.filterwarnings("ignore")


# In[2]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv("../input/train.csv", \n                         dtype={"acoustic_data": np.int16, \n                                "time_to_failure": np.float32}).values')


# # Feature generation
# 
# For a given ending position "last_index", we split the last 150000 values 
# of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150000.
# From each piece, a set features are extracted. This results in a feature matrix of dimension (150 time steps x features).  

# In[3]:


## the simplified one
def extract_features(z):
    z_abs = np.abs(z)
    return np.c_[z.mean(axis=1),
                 z_abs.max(axis=1),
                 (z*(z>=0)*(z<=10)).std(axis=1),
                 np.transpose(np.quantile(z, q=[0.7, 0.95], axis=1)),
                 np.transpose(np.quantile(z_abs, q=[0.3, 0.95], axis=1))]


def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index = len(x)
       
    assert last_index - n_steps * step_length >= 0

    temp = x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1)/3
    
    # Extracts features of sequences of full length 1000 and some fractions of it
    return np.c_[extract_features(temp),
                 extract_features(temp[:, -step_length // 3:]),
                 extract_features(temp[:, -step_length // 10:])]


# ### Some global variables

# In[4]:


get_ipython().run_cell_magic('time', '', 'rows = 150_000\nn_features = create_X(train[:rows,0]).shape[1]\nn_targets = 5\nprint("The model RNN is based on {0} features and {1} targets.\\n".format(n_features, n_targets))')


# In[5]:


batch_size = 32
eps = 0.5 # weights for the second target (time_after_failure)
n_steps = 150
step_length = rows//n_steps

index_earthquake_start = np.nonzero(np.diff(train[:,1]) > 0)[0] + 1
cv_earthquake_index = index_earthquake_start[1]


# In[6]:


print(index_earthquake_start)


# # Data generator
# The generator endlessly selects `batch_size` ending positions of sub-time series. For each ending position, the `time_to_failure` serves as target, while the features are created by the function `create_X`.

# In[7]:


def generator(data, min_index=0, max_index=None, batch_size=32, n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        seg_length = n_steps * step_length
        index_range = range(min_index + seg_length, max_index, 20000)
        rows = np.random.choice(index_range, size=batch_size, replace=False)
#         for limit in index_earthquake_start: 
#             rows = rows[np.logical_not\
#                         (np.logical_and(rows>limit, rows<(limit+160000)))]
         
        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        
        ## adding a sequence of targets or a single target
        targets = np.zeros((batch_size, n_targets))
        
        for j, row in enumerate(rows):
            samples[j] = create_X(data[:, 0], 
                                  last_index = row, 
                                  n_steps = n_steps,
                                  step_length = step_length)
            
            if n_targets == 1:
                ## single target
                targets[j] = data[row - 1, 1]
            elif n_targets == 2:
                ## here the training needs to be chosen as the ones after the first earthquake
                targets[j,0] = data[row - 1, 1]
                ## time_after_failure 
                taf_idx = index_earthquake_start[np.sum(row > index_earthquake_start) - 1]
                targets[j,1] = eps*(data[taf_idx, 1] - targets[j,0])
            elif n_targets > 2:
                ## multiple targets (preferably odd number)
                for i in range(n_targets):
                    # targets are all in one segments
                    # targets[j,i] = data[row-1-i*(seg_length//n_targets), 1]  
                    # targets are spanning multiple neighboring segments
                    targets[j,i] = data[row - 1 - i*(seg_length//2), 1] 

        yield samples, targets


# In[8]:


# Initialize generators
train_gen = generator(train, 
                      batch_size=batch_size,
                      n_steps=n_steps, 
                      step_length=step_length) 

# train_gen = generator(train, 
#                       batch_size=batch_size, 
#                       min_index=cv_earthquake_index,
#                       n_steps=n_steps, 
#                       step_length=step_length)

valid_gen = generator(train, 
                      batch_size=batch_size, 
                      max_index=cv_earthquake_index-1,
                      n_steps=n_steps, 
                      step_length=step_length)

# verify the generator
aux, aux2 = next(train_gen)
print(aux.shape, aux2.shape)


# # RNN Model

# In[9]:


from keras.models import Sequential
from keras.engine.topology import Layer
from keras.layers import Dense, CuDNNGRU, Dropout, LSTM, CuDNNLSTM, Bidirectional, BatchNormalization
from keras.optimizers import adam, RMSprop
from keras import initializers, regularizers, constraints
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from tensorflow import set_random_seed
set_random_seed(1127)


# In[11]:


# The LSTM architecture
model = Sequential()

''' 
LSTM based RNN (GPU)
'''
# # First RNN layer
# model.add(CuDNNLSTM(units=96, return_sequences=True, input_shape=(None,n_features)))
# model.add(Dropout(0.2))

# # Second LSTM layer
# model.add(CuDNNLSTM(units=48, return_sequences=True))
# model.add(Dropout(0.2))

# # Third LSTM layer
# model.add(CuDNNLSTM(units=48, return_sequences=True))
# model.add(Dropout(0.2))

# # Fourth LSTM layer
# model.add(CuDNNLSTM(units=48))

# model.add(Dense(units=n_targets))


''' 
GRU based RNN (GPU)
'''
# First RNN layer
model.add(CuDNNGRU(units=50, return_sequences=True, input_shape=(None,n_features)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(CuDNNGRU(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Third LSTM layer
model.add(CuDNNGRU(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Fourth LSTM layer
model.add(CuDNNGRU(units=50))
# model.add(Dropout(0.2))


# The output layer
model.add(Dense(units=n_targets))


model.summary()


# In[13]:


# Compile and fit model

cb = [ModelCheckpoint("model.hdf5", monitor='val_mean_absolute_error',
                      save_best_only=True, period=3)]

model.compile(optimizer = 'rmsprop',
              loss = 'logcosh',
              metrics = ['mae'])

# model.compile(optimizer = 'rmsprop',
#               loss = 'mae')


history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,
                              epochs=50,
                              verbose=2,
                              callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=200
                             )


# In[ ]:


# Visualize accuracies

loss = history.history['mean_absolute_error']
val_loss = history.history['val_mean_absolute_error']

# loss = history.history['loss']
# val_loss = history.history['val_loss']
epochs = np.asarray(history.epoch) + 1
    
plt.plot(epochs, loss, 'bo', label = "Training MAE")
plt.plot(epochs, val_loss, 'b*', label = "Validation MAE")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.legend();


# # Predictions

# In[ ]:


rows = 150000
num_segments = int(np.floor(train.shape[0] / rows))

y_tr = np.zeros(num_segments)
y_tr_pred = np.zeros((num_segments,n_targets))

for i in tqdm_notebook(range(num_segments)):
    x = train[i*rows : i*rows+rows, 0]
    y_tr[i] = train[i*rows+rows-1, 1]
    y_tr_pred[i] = model.predict(np.expand_dims(create_X(x), 0))[0]


# In[ ]:


print("The training MAE is {:.7}.".format(np.abs(y_tr_pred[:,0] - y_tr).mean()))


# In[ ]:


if n_targets>2:
    time = np.arange(n_targets).reshape(-1,1)
    y_tr_extrapolated = np.zeros(num_segments)

    for i in tqdm_notebook(range(num_segments)):
        clf = Ridge(alpha=0.1)
        clf.fit(time, y_tr_pred[i])
        if clf.coef_>0:
            y_tr_extrapolated[i]  = clf.predict(time[:3]).mean()      
        else:
            y_tr_extrapolated[i] = y_tr_pred[i,:3].mean()
    print("The training MAE for extrapolation is {:.7}."          .format(np.abs(y_tr_extrapolated - y_tr).mean()))
    
    plt.figure(figsize=(18, 6))
    plt.plot(y_tr, color='g', label='time_to_failure', linewidth = 2)
    plt.plot(y_tr_extrapolated, color='b', label='RNN extrapolated prediction')
    plt.legend(loc='best');
    plt.title('RNN prediction vs ttf');
else:
    plt.figure(figsize=(18, 6))
    plt.plot(y_tr, color='g', label='time_to_failure', linewidth = 2)
    plt.plot(y_tr_pred, color='b', label='RNN predictions')
    plt.legend(loc='best');
    plt.title('RNN prediction vs ttf');


# In[ ]:


# Load submission file
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
x = None
y_test_pred = np.zeros((len(submission),n_targets))


# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm_notebook(submission.index)):
  #  print(i)
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values[:]
    y_test_pred[i] = model.predict(np.expand_dims(create_X(x), 0))[0]
    if n_targets > 2:
        clf = Ridge(alpha=0.1)
        clf.fit(time, y_test_pred[i].reshape(-1,1))
        if clf.coef_>0:
            y_pred_extrapolated = clf.predict(time[:n_targets-2]).mean()      
        else:
            y_pred_extrapolated = y_test_pred[i,:n_targets-2].mean()
        
        submission.time_to_failure[i] = y_pred_extrapolated
    else:
        submission.time_to_failure[i] = y_test_pred[i]

# Save
submission.to_csv('submission_lstm_extrapolated.csv')


# In[ ]:


pd.DataFrame(y_tr_pred).to_csv('y_tr_pred.csv',index=False)
pd.DataFrame(y_test_pred).to_csv('y_test_pred.csv',index=False)

