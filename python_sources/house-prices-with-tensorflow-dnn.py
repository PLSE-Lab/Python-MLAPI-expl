from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

#Import our training data and clean it up
training_file = pd.read_csv('../input/train.csv', encoding='utf-8')
testing_file = pd.read_csv('../input/test.csv', encoding='utf-8')
training_file = training_file.drop(['Id'], axis=1)
testing_file = testing_file.drop(['Id'], axis=1)

#Create two sets of data, splitting continous and categorical data
train_num = training_file.select_dtypes(exclude=['object'])
train_cat = training_file.select_dtypes(include=['object'])
test_num = testing_file.select_dtypes(exclude=['object'])
test_cat = testing_file.select_dtypes(include=['object'])

#Fill missing data with something, NONE for objects, 0 for integers
train_num.fillna(0, inplace=True)
train_cat.fillna('NONE', inplace=True)
test_num.fillna(0, inplace=True)
test_cat.fillna('NONE', inplace=True)

#Convert categorical data to one-hot
train_cat = pd.get_dummies(train_cat, dummy_na=False, sparse=True)
test_cat = pd.get_dummies(test_cat, dummy_na=False, sparse=True)

#Split SalePrice from train_num and assign it to label var
label = train_num[["SalePrice"]]
train_num = train_num.drop("SalePrice", axis=1)

#Normalise train_num using MinMax Scaler
train_num_cols = list(train_num.columns)
test_num_cols = list(test_num.columns)
x = train_num.values
z = test_num.values
minmax_scaler = MinMaxScaler()
x_scaled = minmax_scaler.fit_transform(x)
z_scaled = minmax_scaler.fit_transform(z)
train_num = pd.DataFrame(x_scaled, columns=train_num_cols)
test_num = pd.DataFrame(z_scaled, columns=test_num_cols)

#Merge our two datasets
dataset = pd.merge(train_num, train_cat, left_index=True, right_index=True)
testdata = pd.merge(test_num, test_cat, left_index=True, right_index=True)

#Remove columns from training and test datasets that do not appear in both
for col in dataset.columns:
  if col not in testdata.columns:
    dataset = dataset.drop([col], axis=1)

for col in testdata.columns:
  if col not in dataset.columns:
    testdata = testdata.drop([col], axis=1)

dataset = dataset.reindex(sorted(dataset.columns), axis=1)
testdata = testdata.reindex(sorted(testdata.columns), axis=1)

#Split dataset into training and validation sets
dataset = dataset.values
x_train, x_val, y_train, y_val = train_test_split(dataset, label, test_size=0.10, random_state=42)

#Build our DNN using leaky_relu and drop out layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu, input_shape=[x_train.shape[1]]),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(loss='mean_squared_logarithmic_error',
              optimizer=optimizer,
              metrics=['mean_squared_logarithmic_error'])

model.summary()

#Define our callback function to halt training when we see no more improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=20)

#Train our model
history = model.fit(x_train,
                   y_train,
                   batch_size=100,
                   epochs=1000,
                   steps_per_epoch=x_train.shape[0] // 100,
                   callbacks=[early_stop],
                   validation_data=(x_val, y_val))


#Evaluate model against test data
mse, _ = model.evaluate(dataset, label)
rmse = np.sqrt(mse)
print('Root Mean Square Error on test set: {}'.format(round(rmse, 3)))

#Show some statistics of our training
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch  
  plt.figure(figsize=(12,8))
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_logarithmic_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_logarithmic_error'],
           label = 'Val Error')
  plt.ylim([0,.1])
  plt.legend()
  plt.show()

plot_history(history)

# #Make predictions against test data and save for submission
testdata = testdata.values
test_predictions = model.predict(testdata).flatten()
submission = pd.read_csv('../input/sample_submission.csv')
submission['SalePrice'] = test_predictions
submission.isnull().any().any()
submission.to_csv("my_submission.csv",index=False)