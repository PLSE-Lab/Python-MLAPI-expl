#!/usr/bin/env python
# coding: utf-8

# # Load Dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

data_dir = Path('../input/digit-recognizer')


# In[ ]:


# Load training data.
train = pd.read_csv(data_dir/'train.csv')
train


# In[ ]:


# Load test data.
test = pd.read_csv(data_dir/'test.csv')
test


# In[ ]:


# Split training data for learning and validation
from sklearn.model_selection import train_test_split

X_train = train.drop(columns='label').values
y_train = train['label'].values

X_learn, X_valid, y_learn, y_valid = train_test_split(X_train, y_train, random_state=0)

X_test = test.values


# In[ ]:


# Reshape from 784px to (28px, 28px, 1channel).
X_learn_gray = X_learn.reshape((-1, 28, 28, 1))
X_valid_gray = X_valid.reshape((-1, 28, 28, 1))
X_test_gray = X_test.reshape((-1, 28, 28, 1))


# In[ ]:


# ResNet50 handles 3channels, so convert from gray to rgb.
import tensorflow as tf

X_learn_rgb = tf.image.grayscale_to_rgb(tf.constant(X_learn_gray))
X_valid_rgb = tf.image.grayscale_to_rgb(tf.constant(X_valid_gray))
X_test_rgb = tf.image.grayscale_to_rgb(tf.constant(X_test_gray))


# In[ ]:


# ResNet50 needs at least (32px, 32px), so resize.
X_learn_large = tf.image.resize(X_learn_rgb, [32, 32], method='nearest')
X_valid_large = tf.image.resize(X_valid_rgb, [32, 32], method='nearest')
X_test_large = tf.image.resize(X_test_rgb, [32, 32], method='nearest')


# In[ ]:


# Scale the range of values to [0, 1]
X_learn_scale = X_learn_large / 255
X_valid_scale = X_valid_large / 255
X_test_scale = X_test_large / 255


# In[ ]:


# Show the first 16 images of X_learn_scale
fig, ax = plt.subplots(4, 4, figsize=(15, 15))
for row in range(4):
    for col in range(4):
        ax[row, col].imshow(X_learn_scale[row*4+col])


# # Create a Model using ResNet50

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping

n_classes = 10


# In[ ]:


# Create a new model using ResNet50.
# Remove the top layer by specifying `include_top=False`.
# Add a Dense layer on the top to classification.
model = Sequential([
    ResNet50(input_shape=(32, 32, 3), include_top=False, pooling='avg'),
    Dense(n_classes, activation='softmax'),
])

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# # Train the Model

# In[ ]:


# Early stopping.
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train.
get_ipython().run_line_magic('time', 'history = model.fit(X_learn_scale, y_learn, epochs=20, validation_split=0.2, callbacks=[early_stop])')


# In[ ]:


# Show history in table.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist


# In[ ]:


# Show history in graph.
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.plot(hist['epoch'], hist['loss'], label='Train Error')
ax.plot(hist['epoch'], hist['val_loss'], label = 'Valid Error')
ax.legend();


# # Predict with the Model

# In[ ]:


# The method `predict` returns likelihood,
# so use the most likelihood argument for label.
likelihood = model.predict(X_test_scale)
predict = np.argmax(likelihood, axis=1)

# Create subumission file.
result = pd.DataFrame({'ImageId': test.index + 1, 'Label': predict}, dtype='int') # ImageId is 1, 2, ...
result.to_csv('resnet50_submission.csv', index=False)

