#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">CNNlstm less features </font></center></h1>
# 
# <br>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data analysis</a>  
# - <a href='#3'>Calculate aggregated features</a>
# - <a href='#4'>New features exploration</a>  
# - <a href='#5'>Conclusions</a>  
# - <a href='#6'>References</a>  
# 

# # <a id='1'>Introduction</a>  
# 
# ## Simulated earthquake experiment
# The data are from an experiment conducted on rock in a double direct shear geometry subjected to bi-axial loading, a classic laboratory earthquake model.
# 
# Two fault gouge layers are sheared simultaneously while subjected to a constant normal load and a prescribed shear velocity. The laboratory faults fail in repetitive cycles of stick and slip that is meant to mimic the cycle of loading and failure on tectonic faults. While the experiment is considerably simpler than a fault in Earth, it shares many physical characteristics.
# 
# Los Alamos' initial work showed that the prediction of laboratory earthquakes from continuous seismic data is possible in the case of quasi-periodic laboratory seismic cycles.
# 
# ## Competition
# In this competition, the team has provided a much more challenging dataset with considerably more aperiodic earthquake failures.
# Objective of the competition is to predict the failures for each test set.

# # <a id='2'>Prepare the data analysis</a>  
# 
# ## Load packages

# In[ ]:


import gc
import os
import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
warnings.filterwarnings('ignore')


# ## Load data

# In[ ]:


PATH="../input/"
os.listdir(PATH)


# In[ ]:


print("There are {} files in test folder".format(len(os.listdir(os.path.join(PATH, 'test' )))))


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv(os.path.join(PATH,'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})")


# In[ ]:


print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))


# # <a id='3'>Calculate aggregated features</a>  

# In[ ]:


rows = 150000
segments = int(np.floor(train_df.shape[0] / rows))
print("Number of segments: ", segments)


# In[ ]:


train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
train_X .shape


# In[ ]:


def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)   
    zc = np.fft.fft(xc)
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    X.loc[seg_id, 'sum'] = xc.sum()
    X.loc[seg_id, 'mad'] = xc.mad()
  


# In[ ]:


# iterate over all segments
for seg_id in tqdm_notebook(range(segments)):
    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]
    create_features(seg_id, seg, train_X)
    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]


# In[ ]:


train_X


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)


# In[ ]:


for seg_id in tqdm_notebook(test_X.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    create_features(seg_id, seg, test_X)


# In[ ]:


print("Train X: {} y: {} Test X: {}".format(train_X.shape, train_y.shape, test_X.shape))


# In[ ]:


train_X.head()


# In[ ]:


test_X.head()


# # <a id='4'>New features exploration</a> 
# 
# 
# ## Aggregated features
# 
# Let's visualize the new features distributions. The graphs below shows the distplot (histograms and density plots) for all the new features, for train (<font color="green">green</font>) and test (<font color="blue">blue</font>) data.

# In[ ]:


def plot_distplot(feature):
    plt.figure(figsize=(16,6))
    plt.title("Distribution of {} values in the train and test set".format(feature))
    sns.distplot(train_X[feature],color="green", kde=True,bins=120, label='train')
    sns.distplot(test_X[feature],color="blue", kde=True,bins=120, label='test')
    plt.legend()
    plt.show()


# In[ ]:


def plot_distplot_features(features, nlines=3, colors=['green', 'blue'], df1=train_X, df2=test_X):
    i = 0
    plt.figure()
    fig, ax = plt.subplots(nlines,2,figsize=(16,4*nlines))
    for feature in features:
        i += 1
        plt.subplot(nlines,2,i)
        sns.distplot(df1[feature],color=colors[0], kde=True,bins=40, label='train')
        sns.distplot(df2[feature],color=colors[1], kde=True,bins=40, label='test')
    plt.show()


# In[ ]:


features = ['mean', 'std', 'max', 'min', 'sum', 'mad']

plot_distplot_features(features)


# ## Scaled features
# 
# Let's scale now the aggregated features and show again the resulting graphs.   
# We are fiting the scaler with both train and test data.
# We use <font color="red">red</font> from train and <font color="magenta">magenta</font> for test data.

# In[ ]:


scaler = StandardScaler()
scaler.fit(pd.concat([train_X, test_X]))
scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)


# In[ ]:


features = ['mean', 'std', 'max', 'min', 'sum', 'mad']
plot_distplot_features(features, nlines=4, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)


# ## Aggregated features and time to failure
# 
# Let's also show aggregated features and time to failure on the same graph. 

# In[ ]:


def plot_acc_agg_ttf_data(feature, title="Averaged accoustic data and ttf"):
    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title('Averaged accoustic data ({}) and time to failure'.format(feature))
    plt.plot(train_X[feature], color='r')
    ax1.set_xlabel('training samples')
    ax1.set_ylabel('acoustic data ({})'.format(feature), color='r')
    plt.legend(['acoustic data ({})'.format(feature)], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_y, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)


# In[ ]:


plot_acc_agg_ttf_data('mean')


# In[ ]:


plot_acc_agg_ttf_data('std')


# In[ ]:


plot_acc_agg_ttf_data('max')


# In[ ]:


plot_acc_agg_ttf_data('min')


# In[ ]:


plot_acc_agg_ttf_data('sum')


# In[ ]:


plot_acc_agg_ttf_data('mad')


# # <a id='5'>Rearranging data </a> 

# In[ ]:


validation_point=351
endpoint=train_X.shape[0]-1
tttt=validation_point-endpoint

X_train=scaled_train_X.values[validation_point:endpoint,].reshape(tttt,3,2)
print('X_train.shape',X_train.shape)
y_train=train_y.values[validation_point:endpoint,]
print('y_train.shape',y_train.shape)

X_validation=scaled_train_X.values[0:validation_point-1,].reshape(validation_point-1,3,2)
print('X_validation.shape',X_validation.shape)
y_validation=train_y.values[0:validation_point-1,]
print('y_validation.shape',y_validation.shape)

X_train.shape[1]


# # <a id='5'>Defining Model </a> 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, SimpleRNN, LSTM ,  Dropout, Activation, Flatten, Input, Conv1D, MaxPooling1D
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import datetime


# In[ ]:


#used to help some of the timing functions
now = datetime.datetime.now


# In[ ]:


i = (X_train.shape[1],X_train.shape[2])
model = Sequential ()
model.add(Conv1D(2, 2, activation='relu', input_shape= i))
model.add(MaxPooling1D(2))
model.add(LSTM(50,  return_sequences=True))
model.add(LSTM(10))
model.add(Dense(240))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(1))


model.summary()


# # <a id='5'>Compile and fit model </a>                                            

# In[ ]:


import keras
from keras.optimizers import RMSprop
opt = keras.optimizers.adam(lr=.005)

model.compile(loss="mae",
              optimizer=opt, metrics=['mean_absolute_error'])
             # metrics=['accuracy'])


batch_size = 128 # mini-batch with 32 examples
epochs = 50
t = now()

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_validation ,y_validation ))
print('Training time: %s' % (now() - t))


# # <a id='5'>Load submission file </a>                                                                                         
# 

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})


# # <a id='5'>Prepare submission data </a>                              

# In[ ]:


X_test=scaled_test_X.values.reshape(test_X.shape[0],3,2)
print(X_test.shape)

for i, seg_id in enumerate(tqdm(submission.index)):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i]= model.predict(np.expand_dims(X_test[i], 0))
    
    


# In[ ]:


submission_newfeatures=submission
submission_newfeatures.head()


# # <a id='5'>Save submission file </a>                              

# In[ ]:


submission_newfeatures.to_csv('submission_newfeatures.csv')


# # <a id='6'>References</a>  
# 
# [1] LANL Earthquake Prediction, https://www.kaggle.com/c/LANL-Earthquake-Prediction  
# [2] Shaking Earth, https://www.kaggle.com/allunia/shaking-earth  
# [3] Earthquake FE - more features and samles, https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples  
# [4] Laboratory observations of slow earthquakes and the spectrum of tectonic fault slip modes, https://www.nature.com/articles/ncomms11104   
# [5] Machine Learning Predicts Laboratory Earthquakes, https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL074677  
# 
