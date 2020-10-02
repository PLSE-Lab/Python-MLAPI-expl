#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# ### Investing your money on a stock is always risky. If you want to make a profit, you have to be careful and do smart predictions.
# ### But what about if we train a RNN for making predictions for future? (Many-to-One RNN)
# ### In this Notebook I will apply very similar Sequential models. One with <a href="#rnn">SimpleRNN</a> layers and the other with <a href="#lstm">LSTM</a> layers. 
# ### Then we can test how good are they for making good predictions and <a href="#eva">Evaluate</a> the results.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# # Data

# In[ ]:


data = pd.read_csv("../input/UTX_2006-01-01_to_2018-01-01.csv")        # loading data (can be any stock)


# In[ ]:


data.head()    # EDA operations


# In[ ]:


data.info()


# In[ ]:


# I picked the "Low" column to work on

Low_data = data.iloc[:,3:4].values


# In[ ]:


Low_data           # checking the data


# In[ ]:


plt.figure(figsize=(14,10))                 # Visualizing the data
plt.plot(Low_data,c="red")
plt.title("Microsoft Stock Prices",fontsize=16)
plt.xlabel("Days",fontsize=16)
plt.ylabel("Scaled Price",fontsize=16)
plt.grid()
plt.show()


# # Data PreProcess

# In[ ]:


scaler = MinMaxScaler(feature_range=(0,1))           # Scaling the data between 1 and 0


# In[ ]:


Low_scaled = scaler.fit_transform(Low_data)


# In[ ]:


step_size = 21                      # days that are used for the following prediction

train_x = []
train_y = []


# In[ ]:


for i in range(step_size,3019):                # making feature and the label lists
    train_x.append(Low_scaled[i-step_size:i,0])
    train_y.append(Low_scaled[i,0])


# In[ ]:


train_x = np.array(train_x)                   # converting our lists to the arrays
train_y = np.array(train_y)


# In[ ]:


print(train_x.shape)                                # checking the shape


# In[ ]:


test_x = train_x[2500:]            # last 419 days are going to be used in test 
train_x = train_x[:2500]           # first 2500 days are going to be used in training
test_y = train_y[2500:]  
train_y = train_y[:2500]


# In[ ]:


train_x = np.reshape(train_x, (2500, step_size, 1))           # reshaping them for the Keras model
test_x = np.reshape(test_x, (498, step_size, 1))


# In[ ]:


print(train_x.shape)
print(test_x.shape)                             # checking the shapes again


# In[ ]:


# Now we can start to create our models, starting with RNN


# <a id="rnn"></a>
# # RNN

# In[ ]:


rnn_model = Sequential()


# In[ ]:


rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True, input_shape=(train_x.shape[1],1)))
rnn_model.add(Dropout(0.15))

rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
rnn_model.add(Dropout(0.15))

rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
rnn_model.add(Dropout(0.15))

rnn_model.add(Dense(1))


# In[ ]:


rnn_model.compile(optimizer="adam",loss="MSE")


# In[ ]:


rnn_model.fit(train_x,train_y,epochs=20,batch_size=25)


# In[ ]:


rnn_predictions = rnn_model.predict(test_x)

rnn_score = r2_score(test_y,rnn_predictions)


# In[ ]:


# Now with the next model, LSTM


# <a id="lstm"></a>
# # LSTM

# In[ ]:


lstm_model = Sequential()


# In[ ]:


lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(train_x.shape[1],1)))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))
lstm_model.add(Dropout(0.15))

lstm_model.add(Dense(1))


# In[ ]:


lstm_model.compile(optimizer="adam",loss="MSE")


# In[ ]:


lstm_model.fit(train_x,train_y,epochs=20,batch_size=25)


# In[ ]:


lstm_predictions = lstm_model.predict(test_x)

lstm_score = r2_score(test_y,lstm_predictions)


# In[ ]:


# Trainings are done. Now we can continue with Evaluation results


# <a id="eva"></a>
# # Evaluation

# In[ ]:


print("R^2 Score of RNN",rnn_score)
print("R^2 Score of LSTM",lstm_score)


# In[ ]:


lstm_predictions = scaler.inverse_transform(lstm_predictions)
rnn_predictions = scaler.inverse_transform(rnn_predictions)
test_y = scaler.inverse_transform(test_y.reshape(-1,1))


# In[ ]:


plt.figure(figsize=(16,12))

plt.plot(test_y, c="blue",linewidth=2, label="original")
plt.plot(lstm_predictions, c="green",linewidth=2, label="LSTM")
plt.plot(rnn_predictions, c="red",linewidth=2, label="RNN")
plt.legend()
plt.title("COMPARISONS",fontsize=20)
plt.grid()
plt.show()


# ### So after all, I think I am not able to be the wolf of the wall street (YET). But even this simple RNN model can help a lot to make better predicitons.
# Thanks for reading.
# 
# ## Reference
# https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru

# In[ ]:




