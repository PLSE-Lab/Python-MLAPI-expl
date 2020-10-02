#!/usr/bin/env python
# coding: utf-8

# # Time Series Analysis<br>
# Damien Park<br>
# 2019.02.21

# ### Importing Library

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler

import matplotlib.pyplot as plt

import keras


# ### Load data

# In[2]:


df = pd.read_csv("../input/GOOGL_2006-01-01_to_2018-01-01.csv")
df.head()


# ### Data check

# In[3]:


df.Date.is_unique


# In[5]:


print("Number of Na")
for i in df.columns:
    print(i,":", sum(df[i].isna()))


# ### Preprocessing

# In[6]:


df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")


# In[7]:


df.info()


# In[8]:


df.set_index("Date", inplace=True)
df.drop("Name", axis=1, inplace=True)
scaler = MinMaxScaler()
df_1 = scaler.fit_transform(df)
df_1 = pd.DataFrame(df_1, columns=df.columns)


# In[ ]:


# df.Open = (df.Open - min(df.Open)) / (max(df.Open) - min(df.Open))


# ### Visualization

# In[9]:


plt.figure(figsize=(25, 12))
plt.suptitle("Google stock price", size=20)

plt.subplot(2, 2, 1)
plt.plot(df.index, df_1.Open)
plt.xlabel("Date")
plt.title("Open")

plt.subplot(2, 2, 2)
plt.plot(df.index, df_1.High)
plt.xlabel("Date")
plt.title("High")

plt.subplot(2, 2, 3)
plt.plot(df.index, df_1.Low)
plt.xlabel("Date")
plt.title("Low")

plt.subplot(2, 2, 4)
plt.plot(df.index, df_1.Close)
plt.xlabel("Date")
plt.title("Close")

plt.show()


# In[10]:


plt.figure(figsize=(25, 12))
plt.suptitle("Google stock Volume", size=20)

plt.subplot(2, 2, 1)
plt.plot(df.index, df_1.Volume)
plt.xlabel("Date")
plt.title("Volume")

plt.subplot(2, 2, 2)
plt.plot(df.index, np.log10(df_1.Volume))
plt.xlabel("Date")
plt.title("Volume log scale")

plt.show()


# ### Data setting

# In[11]:


input_len = 30


# In[12]:


test = df_1[df.index>="2017-01-01"]
train = df_1[df.index<"2017-01-01"]


# In[13]:


train_x = []
for i in range(input_len, len(train)):
    train_x.append(train.iloc[i-30:i, 1])

train_y = train.iloc[input_len:, 1]

train_x = np.array(train_x)
train_y = np.array(train_y)

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))


# In[14]:


test_x = []
for i in range(input_len, len(test)):
    test_x.append(test.iloc[i-30:i, 1])

test_y = test.iloc[input_len:, 1]

test_x = np.array(test_x)
test_y = np.array(test_y)

test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))


# In[15]:


train_x.shape, train_y.shape


# In[16]:


test_x.shape, test_y.shape


# ### Modeling

# In[17]:


RNN = keras.Sequential()


# In[18]:


RNN.add(keras.layers.LSTM(units=10, stateful=True, return_sequences=True, batch_input_shape=(1, train_x.shape[1], 1)))

RNN.add(keras.layers.LSTM(units=10, stateful=True, return_sequences=True))

RNN.add(keras.layers.LSTM(units=10, stateful=True, return_sequences=True))

RNN.add(keras.layers.LSTM(units=10))

RNN.add(keras.layers.Dense(units=1))


# In[19]:


RNN.compile(optimizer="RMSprop", loss="MSE")


# ### Training

# In[20]:


for i in range(5):
    RNN.fit(train_x, train_y, epochs=1, batch_size=1)
    RNN.reset_states()


# In[28]:


# save model
RNN.save("RNN.h5")


# ### Prediction

# In[22]:


result = RNN.predict(test_x, batch_size=1)


# In[23]:


temp = df_1[df.index>="2017-02-15"]
temp.Open = result
prediction = pd.DataFrame(scaler.inverse_transform(temp), columns=df_1.columns)


# In[25]:


plt.figure(figsize=(25, 12))

plt.plot(df.index[df.index>="2017-02-15"], prediction.Open, label="prediction", alpha=.7)
plt.plot(df.index[df.index>="2017-02-15"], df.loc[df.index>="2017-02-15", "Open"], label="real", alpha=.7)

plt.scatter(df.index[df.index>="2017-02-15"], prediction.Open, label="prediction", alpha=.7)
plt.scatter(df.index[df.index>="2017-02-15"], df.loc[df.index>="2017-02-15", "Open"], label="real", alpha=.7)

plt.legend()
plt.title("Google stock Prediction")

plt.show()


# In[34]:


# MSE
np.mean(np.square(df.loc[df.index>="2017-02-15", "Open"].values-prediction.Open))


# In[27]:


# MSE
_ = np.reshape(result, (len(result)))
np.mean(np.square(_-test_y))


# ---

# The end of analysis.
