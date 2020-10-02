#!/usr/bin/env python
# coding: utf-8

# # How to resume training a model
# Traning a model may take a very long time. This kernel shows how to train a keras model, visualize it's improvement and continue training it. We track training sessions in the `histories` variable.

# In[ ]:


from keras import models
from keras import layers
import matplotlib.pyplot as plt


# In[ ]:


from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()


# In[ ]:


mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# In[ ]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                        input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


# In[ ]:


# track histories across training sessions
histories = []
histories.append(
    model.fit(train_data, train_targets,
              batch_size=2, epochs=20, verbose=0))


# In[ ]:


def plot_histories(histories):
    plt.clf()
    mae = []
    for history in histories:
        for error in history.history['mean_absolute_error']:
            mae.append(error)
    epochs = range(1, len(mae) + 1)

    plt.plot(epochs, mae, 'b', label='Training mae')
    plt.title('Training mean absolute error')
    plt.legend()
    plt.show()


# In[ ]:


plot_histories(histories)


# In[ ]:


# continue training


# In[ ]:


histories.append(
    model.fit(train_data, train_targets,
              batch_size=2, epochs=20, verbose=0))


# In[ ]:


len(histories)


# In[ ]:


plot_histories(histories)

