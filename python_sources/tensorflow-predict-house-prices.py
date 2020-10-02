#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
print(os.listdir("../input"))


# In[ ]:


def load_data(test_split=0.2):
    with np.load("../input/boston_housing.npz") as f:
        x = f['x']
        y = f['y']
    
    num = len(x)
    indices = np.arange(num)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    
    test_start = int((1-test_split)*num)
    train_x = np.array(x[:test_start])
    train_y = np.array(y[:test_start])
    
    test_x = np.array(x[test_start:])
    test_y = np.array(y[test_start:])
    
    return (train_x,train_y),(test_x,test_y)
    


# In[ ]:


(train_data, train_labels), (test_data, test_labels) = load_data()
# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]


# In[ ]:


print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)


# In[ ]:


print(train_data[0])


# In[ ]:


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data,columns=column_names)
df.head()


# In[ ]:


print(train_labels[0:10])


# In[ ]:


mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


# In[ ]:


def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    
    return model

model = build_model()
model.summary()


# In[ ]:


class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
        else:
            print('.', end='')


# In[ ]:


EPOCHS = 500
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])


# In[ ]:


plot_history(history)


# In[ ]:


model1 = build_model()
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)

history1 = model1.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[earlyStopping,PrintDot()])


# In[ ]:


plot_history(history1)


# In[ ]:


[loss, mae] = model1.evaluate(test_data, test_labels, verbose=0)


# In[ ]:


print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))


# In[ ]:


test_predictions = model1.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


# In[ ]:


error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")


# In[ ]:





# In[ ]:




