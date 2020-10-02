#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the backend
import tensorflow as tf

# Data preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

# Model architecture
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Model evaluation
from sklearn.metrics import accuracy_score


# In[ ]:


# Load the data
train_dataset = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
test_dataset = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")


# In[ ]:


# Read data
X_train = train_dataset.iloc[:, 1:].values.astype("float32")
Y_train = train_dataset.iloc[:, 0].values.astype("int32")
X_test = test_dataset.iloc[:, 1:].values.astype("float32")
NumberID = test_dataset.iloc[:, 0].values.astype("float32")


# In[ ]:


# Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# In[ ]:


# Feature Scaling
X_train /= 255.0
X_test /= 255.0


# In[ ]:


# One hot encoding
Y_train = to_categorical(Y_train, num_classes = 10)


# In[ ]:


# Divide the dataset into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 0)


# In[ ]:


# Setting hyperparemeters
epochs = 50
batch_size = 64
input_dim = (28, 28, 1)


# In[ ]:


# Model architecture (normal)
model = Sequential([
    Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = input_dim),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Dropout(0.2),
    Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Dropout(0.3),
    Conv2D(128, (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Dropout(0.4),
    Flatten(),
    #Dense(512, activation = 'relu'),
    #Dropout(0.25),
    Dense(10, activation = 'softmax')
])


# In[ ]:


# Data Augmentation
datagen = ImageDataGenerator(rotation_range = 10,  
                             zoom_range = 0.1, 
                             width_shift_range = 0.1, 
                             height_shift_range = 0.1,  
                             horizontal_flip = False, 
                             vertical_flip = False)
datagen.fit(X_train)


# In[ ]:


# Early Stopping
early_stopping = EarlyStopping(monitor = 'val_acc',
                               patience = 10,
                               verbose = 1,
                               mode = 'auto',
                               restore_best_weights = True)


# In[ ]:


# Reduce LR
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 
                                            patience = 3, 
                                            verbose = 1, 
                                            factor = 0.5, 
                                            min_lr = 0.00001)


# In[ ]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])


# In[ ]:


model.summary()


# In[ ]:


# With data augmentation on training set
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = batch_size),
                              epochs = epochs,
                              steps_per_epoch = X_train.shape[0] // batch_size,
                              validation_data = (X_val, Y_val), 
                              callbacks = [learning_rate_reduction, early_stopping])


# In[ ]:


# Submitting Predictions to Kaggle
preds = model.predict_classes(X_test)

def write_preds(preds, fname):
    pd.DataFrame({"id": np.arange(preds.shape[0]), "label": preds}).to_csv(fname, index=False)

write_preds(preds, "submission.csv")

