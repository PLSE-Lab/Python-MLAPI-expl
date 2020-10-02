#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.layers import Dense,Dropout,LayerNormalization,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['patch.force_edgecolor']=True
plt.rcParams['figure.figsize'] = (10,7)


# In[ ]:


sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")
df_spec = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
df_test = pd.read_csv("../input/data-science-bowl-2019/test.csv",parse_dates=['timestamp'])
df = pd.read_csv("../input/data-science-bowl-2019/train.csv",parse_dates=['timestamp'])
df_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df_labels.isnull().sum()


# In[ ]:


df.describe().T


# In[ ]:


plt.figure(figsize=(10,7 ))
sns.barplot(x=df.type.value_counts().index, y=df.type.value_counts())


# In[ ]:


plt.figure(figsize=(10,7 ))
sns.barplot(x=df.world.value_counts().index, y=df.world.value_counts())


# In[ ]:


df['date'] = df['timestamp'].apply(lambda date:date.date)


# In[ ]:


df['year'] = df['date'].apply(lambda date:date.year)
df['month'] = df['date'].apply(lambda date:date.month)
df['day'] = df['date'].apply(lambda date:date.day)


# In[ ]:


event_counts = df.groupby(['date'])['event_id'].agg('count')


# In[ ]:


game_time_sum = df.groupby(['date'])['game_time'].agg('sum')


# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(event_counts.index,event_counts.values)
plt.title('Events Counts by Date')
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(game_time_sum.index,game_time_sum.values)
plt.title('Total Game time by Date')
plt.ylabel('In Billions')
plt.show()


# In[ ]:


df['Weekday'] = df['timestamp'].apply(lambda date: date.day_name())


# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(df['Weekday'])


# In[ ]:


gametime_wdays = df.groupby(['Weekday'])['game_time'].agg('sum')


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(gametime_wdays.index,gametime_wdays.values)
plt.title('Total Gametime by Day')
plt.ylabel('Count in Billions')
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(df_labels.corr(),cmap='coolwarm',annot=True)


# In[ ]:


sns.countplot(df_labels['accuracy_group'])


# In[ ]:


df.groupby('installation_id')     .count()['event_id']     .apply(np.log1p)     .plot(kind='hist',
          bins=40,
          color='orange',
         figsize=(15, 5),
         title='Log(Count) of Observations by installation_id')
plt.show()


# In[ ]:


df.groupby('title')['event_id'].count().sort_values().plot(kind='barh',
                                                           title='Count of Observation by Game/Video title',
                                                          color='orange',
                                                          figsize=(15,15))


# In[ ]:


def group_and_reduce(df):
    # group1 and group2 are intermediary "game session" groups,
    # which are reduced to one record by game session. group1 takes
    # the max value of game_time (final game time in a session) and 
    # of event_count (total number of events happened in the session).
    # group2 takes the total number of event_code of each type
    group1 = df.drop(columns=['event_id', 'event_code','timestamp']).groupby(
        ['game_session', 'installation_id', 'title', 'type', 'world']
    ).max().reset_index()

    group2 = pd.get_dummies(
        df[['installation_id', 'event_code']], 
        columns=['event_code']
    ).groupby(['installation_id']).sum()

    # group3, group4 and group5 are grouped by installation_id 
    # and reduced using summation and other summary stats
    group3 = pd.get_dummies(
        group1.drop(columns=['game_session', 'event_count', 'game_time']),
        columns=['title', 'type', 'world']
    ).groupby(['installation_id']).sum()

    group4 = group1[
        ['installation_id', 'event_count', 'game_time']
    ].groupby(
        ['installation_id']
    ).agg([np.sum, np.mean, np.std])

    return group2.join(group3).join(group4)


# In[ ]:


get_ipython().run_line_magic('time', '')
train = group_and_reduce(df)
test = group_and_reduce(df_test)


# In[ ]:


df.drop(df.index, inplace=True)


# In[ ]:


small_labels = df_labels[['installation_id', 'accuracy_group']].set_index('installation_id')


# In[ ]:


train_joined = train.join(small_labels).dropna()


# In[ ]:


X = train_joined.drop(columns='accuracy_group').values


# In[ ]:


y = train_joined['accuracy_group'].values.astype(np.int32)


# In[ ]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[ ]:


X_train.shape


# In[ ]:


# Early stopping
early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1)


# In[ ]:


#Model
model = Sequential()

model.add(Input(shape=X_train.shape[1]))
model.add(Dense(200,activation='relu'))
model.add(LayerNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(100,activation='relu'))
model.add(LayerNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(50,activation='relu'))
model.add(LayerNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(25,activation='relu'))
model.add(LayerNormalization())
model.add(Dropout(rate=0.1))

model.add(Dense(4,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.fit(x=X_train,y=y_train,
         validation_data=(X_test,y_test),
         batch_size=256,
         epochs=200,
         callbacks=[early_stop])


# In[ ]:


loss_rep = pd.DataFrame(model.history.history)


# In[ ]:


loss_rep.head()


# In[ ]:


model.save('model.h5')


# In[ ]:


loss_rep[['loss','val_loss']].plot()


# In[ ]:


loss_rep[['accuracy','val_accuracy']].plot()


# In[ ]:


a = model.predict_classes(X_test)


# In[ ]:


np.unique(a)


# In[ ]:


orig=np.argmax(y_test, axis=1)


# In[ ]:


pred_df = pd.DataFrame(orig,columns=['Test Y'])


# In[ ]:


test_predicition = pd.Series(a.reshape(4421,))


# In[ ]:


pred_df = pd.concat([pred_df,test_predicition],axis=1)


# In[ ]:


pred_df.columns = ['Test_Y','Predictions']


# In[ ]:


print(confusion_matrix(orig,a))


# In[ ]:


print(classification_report(orig,a))


# In[ ]:





# In[ ]:


# ON test Dataset:


# In[ ]:


df_test.head()


# In[ ]:


df_test['date'] = df_test['timestamp'].apply(lambda date:date.date)


# In[ ]:


df_test['year'] = df_test['date'].apply(lambda date:date.year)
df_test['month'] = df_test['date'].apply(lambda date:date.month)
df_test['day'] = df_test['date'].apply(lambda date:date.day)


# In[ ]:


test = group_and_reduce(df_test)


# In[ ]:


test.shape


# In[ ]:


submission = pd.DataFrame(test.index)


# In[ ]:


test = test.values


# In[ ]:


# scaler1 = MinMaxScaler()
test_scaled = scaler.transform(test)


# In[ ]:


a1 = model.predict_classes(test_scaled)


# In[ ]:


submission = pd.concat([submission,pd.Series(a1.reshape(1000,))],axis=1)


# In[ ]:


submission.columns = ['installation_id','accuracy_group']


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=None)


# In[ ]:


submission['accuracy_group'].hist()

