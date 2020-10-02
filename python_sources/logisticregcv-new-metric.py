#!/usr/bin/env python
# coding: utf-8

# ## The visualizations are inspired by ChewZy's [kernel](https://www.kaggle.com/chewzy/eda-ion-switching-signal-vs-open-channels) 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.open_channels.unique()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


size=16
params = {'legend.fontsize': 'large',
          'figure.figsize': (16,4),
          'axes.labelsize': size*1.1,
          'axes.titlesize': size*1.3,
          'xtick.labelsize': size*0.9,
          'ytick.labelsize': size*0.9,
          'axes.titlepad': 25}
plt.rcParams.update(params)


# In[ ]:


signal_batch_size = 500_000
train['signal_batch'] = np.arange(len(train)) // signal_batch_size


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(12,6))

train    .groupby('signal_batch')['open_channels']    .apply(lambda x: len(set(x)))    .value_counts()    .sort_index()    .plot(kind='bar', ax=ax, width=0.8)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', 
                color='black', fontsize=14, 
                #fontweight='heavy',
                xytext=(0,5), 
                textcoords='offset points')

ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels([0,1,2,3,4])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_xlabel('No. of Labels per Signal Batch')
ax.set_ylabel('No. of Signal Batch')
ax.set_title('Distribution of No. of Labels Per Signal Batch '+'$(n = 10)$')

for loc in ['right','top']:
    ax.spines[loc].set_visible(False)


# In[ ]:


def plot_signal_and_label(segment_size=200):
    fig, ax = plt.subplots(1,1, figsize=(14,6))

    sample = np.random.randint(0,9)
    segment = np.random.randint(0,500_000 - segment_size)
    
    df_segment = train.query('signal_batch == @sample')
    
    df_segment['signal'].iloc[segment:segment+segment_size]        .plot(ax=ax, label='Signal', alpha=0.8, linewidth=2)
    
    ax_2nd = ax.twinx()
    df_segment['open_channels'].iloc[segment:segment+segment_size]        .plot(ax=ax_2nd, label='Open Channels (Ground Truth)', color='C1', linewidth=2)

    time_start = df_segment['time'].iloc[segment]
    time_end = df_segment['time'].iloc[segment + segment_size-1]
    
    xticklabels = [val for i, val in enumerate(df_segment['time'].iloc[segment:segment + segment_size + 1]) if i%(segment_size//10) == 0]
    xtickloc = [val for i, val in enumerate(df_segment.iloc[segment:segment + segment_size + 1].index) if i%(segment_size//10) == 0]
    
    ax.set_xticks(xtickloc)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Timestamp')
    
    ax.set_ylabel('Signal')
    ax_2nd.set_ylabel('Open Channels')
    
    ax.set_title(f'Signal Batch #{sample} \n('
                 r'$t_{start} = $' + f'${time_start} s, $'
                 r'$t_{end} = $' + f'${time_end} s$' + ')')
    fig.legend(bbox_to_anchor=(1.03,0.5), loc='center left')
    
    ax.spines['top'].set_visible(False)
    ax_2nd.spines['top'].set_visible(False)
    ax.grid(which='major',axis='x', linestyle='--')

    plt.tight_layout()
    plt.show()
    


# In[ ]:


for i in range(10):
    plot_signal_and_label(segment_size=200)


# ### Modeling

# In[ ]:


train_time = train['time'].values


# In[ ]:


train_time_0 = train_time[:50000]


# In[ ]:


for i in range(1,100):
    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])


# In[ ]:


train['time'] = train_time_0


# In[ ]:


train_time_0 = train_time[:50000]
for i in range(1,40):
    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])
test['time'] = train_time_0


# In[ ]:


X = train[['time', 'signal']].values
y = train['open_channels'].values


# In[ ]:


train.open_channels.nunique()


# In[ ]:


sns.countplot(x='open_channels',data=train)


# In[ ]:


sns.heatmap(train.corr())


# In[ ]:


train.corr()['open_channels'].sort_values().plot(kind='bar')


# In[ ]:


train.corr()['signal'][:-1].sort_values().plot(kind='bar')


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


scaler.fit(X)


# ## Creating a Deep Learning Model

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout


# In[ ]:


model = Sequential()

model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=50,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=25,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=11,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


help(EarlyStopping)


# In[ ]:


early_stop = EarlyStopping(monitor='train_loss', mode='min', verbose=1, patience=25)


# In[ ]:


model.fit(x=X, 
          y=y, 
          epochs=10,
          verbose=1,
          validation_data=(X, y), 
          batch_size = 128,
          callbacks=[early_stop]
          )


# In[ ]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# ## This is absolutely horrible

# ## Lets try LogisticRegressionCV

# In[ ]:


from sklearn.linear_model import LogisticRegressionCV


# In[ ]:


clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)


# In[ ]:


train_preds = clf.predict(X)
train_preds = np.clip(train_preds, 0, 10)
train_preds = train_preds.astype(int)
X_test = test[['time', 'signal']].values


# In[ ]:


test_preds = clf.predict(X_test)
test_preds = np.clip(test_preds, 0, 10)
test_preds = test_preds.astype(int)
submission['open_channels'] = test_preds
submission.head(20)


# In[ ]:


np.set_printoptions(precision=4)
submission['time'] = [format(submission.time.values[x], '.4f') for x in range(2000000)]
submission.to_csv('submission.csv', index=False)


# In[ ]:




