#!/usr/bin/env python
# coding: utf-8

# ![](https://1.bp.blogspot.com/-MQsmXC1vIXE/XEGw7tLb_nI/AAAAAAAACEQ/p4mZ4JiaoFMaact12BUREkn7NPcV0ajJACLcBGAs/s1600/cascades-alpine-tarns-20120819_0135.jpg)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


TrainData = pd.read_csv(
    filepath_or_buffer = '../input/train.csv', # file path
    # nrows = 1500000, # number of rows 150,000
    dtype = {
        'acoustic_data ' : np.int16,
        'time_to_failure' : np.float16
    }
)
TrainData.rename({
    'acoustic_data':'signal',
    'time_to_failure':'time'
},
axis = 'columns',
inplace = True
)


# In[ ]:


fig, ax1 = plt.subplots(figsize=(16, 8))
sns.lineplot(
    data = TrainData['signal'].values[0:10000],
)
ax2 = ax1.twinx()
sns.lineplot(
    data = TrainData['time'].values[0:10000],
    color = 'orange'
)


# In[ ]:


fig, ax1 = plt.subplots(figsize=(16, 8))
sns.lineplot(
    data = TrainData['signal'].values[0:150000],
)
ax2 = ax1.twinx()
sns.lineplot(
    data = TrainData['time'].values[0:150000],
    color = 'orange'
)


# In[ ]:


segment_size = 150_000 # segement size rows
segment_count = int(TrainData.shape[0]/segment_size) # shape[0] means row count

SignalData = pd.DataFrame(
    index = range(segment_count),
    dtype = np.float16,
    columns = [
        'mean', # average
        'stdev', # standard deviation
        'max', #maximum value
        'min' #minimum value
    ]
)

TimeData = pd.DataFrame(
    index = range(segment_count),
    dtype = np.float16,
    columns = [
        'time', # average
    ]
)


# In[ ]:


for segment in tqdm(range(segment_count)):
    slice_from = segment * segment_size
    slice_to  = slice_from + segment_size
    
    slicing = TrainData.iloc[slice_from : slice_to]
    
    Signal = slicing['signal'].values
    Time = slicing['time'].values[-1]
    
    SignalData.loc[segment, 'mean'] = Signal.mean()
    SignalData.loc[segment, 'stdev'] = Signal.std()
    SignalData.loc[segment, 'max'] = Signal.max()
    SignalData.loc[segment, 'min'] = Signal.min()
    
    TimeData.loc[segment, 'time'] = Time


# In[ ]:


data = pd.DataFrame(SignalData.stdev, SignalData.index)
sns.lineplot(data=data, palette="tab10", linewidth=2.5)


# In[ ]:


plt.scatter(SignalData.stdev, SignalData.index) 
plt.xlabel('x') 
plt.xlabel('y') 
plt.title("Training Data") 
plt.show() 


# In[ ]:


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[64]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


plt.figure(figsize=(6, 6))
plt.scatter(TimeData.values.flatten(), TimePredict)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.show()


# In[ ]:


Score = mean_absolute_error(TimeData.values.flatten(), TimePredict)
print(f'Score: {Score:0.3f}')


# In[ ]:


Submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')


# In[ ]:


SignalDataTest = pd.DataFrame(columns=SignalData.columns, dtype=np.float64, index=Submission.index)


# In[ ]:


for seg_id in SignalDataTest.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    SignalDataTest.loc[seg_id, 'ave'] = x.mean()
    SignalDataTest.loc[seg_id, 'std'] = x.std()
    SignalDataTest.loc[seg_id, 'max'] = x.max()
    SignalDataTest.loc[seg_id, 'min'] = x.min()

