import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

import os
import glob

files = glob.glob("/kaggle/input/*/*/*.npz")
print(len(files))

print(a['MobileNetV2_bottleneck_features'].shape)
print(a['yahoo_nsfw_output'].shape)

training = np.load(open(files[0], 'rb'))
train_data = training['MobileNetV2_bottleneck_features']
train_labels = training['yahoo_nsfw_output']

validation = np.load(open(files[1], 'rb'))
validation_data = validation['MobileNetV2_bottleneck_features']
validation_labels = validation['yahoo_nsfw_output']

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=train_data.shape[1:]))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=50,
          validation_data=(validation_data, validation_labels))
model.save_weights('bottleneck_fc_model.h5')
