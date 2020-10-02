#!/usr/bin/env python
# coding: utf-8

# # **LSTM implentation in Keras**

# We are given the number of passengers travelling from a flight for 144 months. We want to predict the number of passengers. This is a typical time-series prediction problem. 

# We start by oading necessary data pre-processing and computation libraries.

# In[ ]:


import numpy as np
import pandas as pd


# Loading the dataset

# In[ ]:


dataset = pd.read_csv("/kaggle/input/air-passengers/AirPassengers.csv")


# Checking if there are any null values

# In[ ]:


dataset.isnull().sum()


# Check dataset shape

# In[ ]:


dataset.shape


# Viewing the dataset

# In[ ]:


dataset.head()


# Now we need to take the number of passengers and store it into an array

# In[ ]:


data = np.array(dataset[["#Passengers"]])


# Viewing the data

# In[ ]:


print(data[:5])


# Checking the shape of the data

# In[ ]:


data.shape


# Now we scale the data into a range of -1 and 1 using the scikit-learn library

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)


# Now we convert the data into sequences. This is done as the LSTM layer only accepts in this form.
# We first take 60 passenger sample into X_train and the next passenger sample is stored in y_train. We repeatedly performed this and store them into a numpy arrays X_train and y_train. 
# 

# In[ ]:


X_train = []
y_train = []
for i in range(60, 144):
    X_train.append(data[i-60:i, 0])
    y_train.append(data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# We check the shape of X_train

# In[ ]:


X_train.shape


# But the LSTM does not accept this type of shape. So we need to reshape this so that it is suitable for LSTM.

# In[ ]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# So this is the new shape.

# In[ ]:


X_train.shape


# Now we start the building our model.
# We start by importing the necessary libraries.

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


# Now we start building our model. 

# 1. We first create an instance of the **Sequential** class, which is the class of Keras, the Sequential model is a linear stack of layers.

# Then we add to the model which named as regressor an LSTM layer. This layer has 50 units or cells of LSTM, we set **return_sequences
# TRUE** to **return_sequences
# TRUE** to tell that the LSTM cell need to return the last state, so that it can be used in the next cell. Then we tell the shape of thee input sequence that we would be giving to the layer.

# Then we add a **dropout** in order to prevent overfitting.[](http://)

# We stack up these layers.

# Then we add the last LSTM layer. Since this is last layer we do not want to take into consideration the last state of the cell, so we skip it as the default is FALSE. 

# Next we add a Dropout layer and finally add a Dense Layer to get the output.

# In[ ]:


regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))


# Now we get the summary of the model that we just built.

# In[ ]:


regressor.summary()


# Now we will compile the model. We used the adam optimizer and the loss function is mean_squared_error as we have a regression problem.

# In[ ]:


regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')


# Now we train or fit the model. We would train for 100 epochs, with a batch size of 32.

# In[ ]:


regressor.fit(X_train,y_train,epochs = 100, batch_size = 32)


# In[ ]:





# 

# 

# 

# 

# In[ ]:


X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))


# In[ ]:


X_train.shape


# In[ ]:


import torch.nn as nn
import torch
from torch.autograd import Variable


# In[ ]:


INPUT_SIZE = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1


# In[ ]:


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state):
        r_out, hidden_state = self.LSTM(x, h_state)
        
        hidden_size = hidden_state[-1].size(-1)
        r_out = r_out.view(-1, hidden_size)
        outs = self.out(r_out)

        return outs, hidden_state

LSTM = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

optimiser = torch.optim.Adam(LSTM.parameters(), lr=0.01)
criterion = nn.MSELoss()

hidden_state = None

for epoch in range(100):
    inputs = Variable(torch.from_numpy(X_train).float())
    labels = Variable(torch.from_numpy(y_train).float())

    output, hidden_state = LSTM(inputs, hidden_state) 

    loss = criterion(output.view(-1), labels)
    optimiser.zero_grad()
    loss.backward(retain_graph=True)                     # back propagation
    optimiser.step()                                     # update the parameters
    
    print('epoch {}, loss {}'.format(epoch,loss.item()))

