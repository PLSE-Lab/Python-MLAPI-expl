#!/usr/bin/env python
# coding: utf-8

# # Keras Binary Classification
# I'm new to ML and this is my first notebook, It would be nice if anyone point out my mistakes.

# In[ ]:


# Importing libraries

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


dataset = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")

# Dropping redundant gameID
dataset.pop("gameId")
dataset.head()


# # Scaling
# Before we split our dataset and fit it to neural network let's scale our data. I'm going to use sklearn StandartScaler.

# In[ ]:


scaler = StandardScaler()

# Taking labels
labels = dataset['blueWins'].values

# Dropping wins column because we don't want our network to know correct answers.
dataset.pop("blueWins")

# Scaling our data
features = dataset.values
features = scaler.fit_transform(features)


# # Splitting data
# Let's split our data to train and test set, I prefer sklearn train_test_split method.

# In[ ]:


train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size=0.2,random_state=18)


# # Creating model
# I have tried different activation functions and nodes so this is the best combination.

# In[ ]:


model = Sequential()
model.add(Dense(8, input_dim=38))
model.add(Dense(16, activation='relu',))
model.add(Dense(1, activation='sigmoid'))
# We're going to use binary_crossentropy as our loss function because we have two states win and lose. I tried different optimizers but Rmsprop proved to be the best for this tas.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


# # Training and Test
# After training we get almost 74% accuracy. Evaluation says that we have 72% accuracy not far from Linear Regression.

# In[ ]:


model.fit(train_features,train_labels, batch_size=32, epochs=20)


# In[ ]:


model.evaluate(test_features,test_labels)

