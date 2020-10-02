#!/usr/bin/env python
# coding: utf-8

# Hi there!
# 
# If you look for a way to beat MNIST challenge or you got stuck at lower score and you want to find a better, simple solution - here it is. This kernel uses very simple CNN model + data augmentation and can be understood and implemented even by a total beginner (like me). Feel free to read my code, comment it and give me feedback about my solution. This solution is inspired by [another](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist) existing Kernel which made me realize that ensembling CNNs makes sense and made me wanna try it.

# In[ ]:


import numpy as np
import pandas as pd 
    
df_test  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_train.sample(n=1)


# In[ ]:


# Removing data
# After fitting a model several times, I noticed that there are some digits that could not be classified even by a human. I want to delete this noisy data to ease learning.

df_train = df_train.drop(index=[335, 445, 666, 737, 881, 1314, 59, 131, 170, 302, 844, 1725, 1820])


# In[ ]:


# Split data frames into training/test sets and normalize pixel values

X_train = np.array(df_train.loc[:, df_train.columns != 'label'])
X_test  = np.array(df_test)
y_train = np.array(df_train['label'])

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32") / 255
X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32") / 255

print(f"X_train: {X_train.shape}\nX_test: {X_test.shape}\ny_train: {y_train.shape}")


# In[ ]:


# One-hot encoding for labels

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
print("Sample one-hot encoded label:", y_train[0])


# In[ ]:


# Creating many convolutional neural networks which I'll use for predictions

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

num_models = 8
net_models = [0] * num_models

for i in range(num_models):
    net_models[i] = Sequential()

    net_models[i].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    net_models[i].add(BatchNormalization())
    net_models[i].add(Conv2D(32, kernel_size = 3, activation='relu'))
    net_models[i].add(BatchNormalization())
    net_models[i].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    net_models[i].add(BatchNormalization())
    net_models[i].add(Dropout(0.4))

    net_models[i].add(Conv2D(64, kernel_size = 3, activation='relu'))
    net_models[i].add(BatchNormalization())
    net_models[i].add(Conv2D(64, kernel_size = 3, activation='relu'))
    net_models[i].add(BatchNormalization())
    net_models[i].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    net_models[i].add(BatchNormalization())
    net_models[i].add(Dropout(0.4))

    net_models[i].add(Conv2D(128, kernel_size = 4, activation='relu'))
    net_models[i].add(BatchNormalization())
    net_models[i].add(Flatten())
    net_models[i].add(Dropout(0.4))
    net_models[i].add(Dense(10, activation='softmax'))

    net_models[i].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Adding some additional data by applying zoom, rotation and x/y shifts

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

datagen.fit(X_train)


# In[ ]:


# Ensemble models
# I train each model on a differently split data (80% for training, 20% for validation) and store training histories.
# After each step I also create a confusion matrix to see what kind of error my networks made after training.

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

histories = [0] * num_models
total_confusion_matrix = np.zeros((10, 10))

for i in range(num_models):
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    histories[i] = net_models[i].fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                                               epochs=50, verbose=0, validation_data=(X_valid, y_valid),
                                               callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, min_delta=0.0001)])
    
    print(f"Trained model no. {i+1}, val_acc: {max(histories[i].history['val_acc'])}, val_loss: {max(histories[i].history['val_loss'])}")
    
    y_valid = [np.argmax(x) for x in y_valid]
    y_valid_pred = [np.argmax(x) for x in net_models[i].predict(X_valid, batch_size=16)]
    total_confusion_matrix += confusion_matrix(y_valid, y_valid_pred)


# In[ ]:


# Plot combined confusion matrix (sum from all models) to see what errors have been made

import seaborn as sn
import matplotlib.pyplot as plt

labels = range(0, 10)
plt.figure(figsize=(20, 6))
df_cm = pd.DataFrame(total_confusion_matrix.astype(int), labels, labels)
sn.set(font_scale=1.4)
ax = sn.heatmap(df_cm, annot=True, fmt='d')
plt.yticks(rotation=0)
plt.show()


# In[ ]:


# Plot validation accuracy and loss for all models

plt.figure(figsize=(15,3))
plt.subplot(121)
for hist in histories:
    plt.plot(hist.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'])

plt.subplot(122)
for hist in histories:
    plt.plot(hist.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'])
plt.show()


# In[ ]:


# Here I sum all softmax outputs from models and use np.argmax to retrieve final prediction!

y_test = np.zeros((X_test.shape[0], 10))
for i in range(num_models):
    y_test = y_test + net_models[i].predict(X_test)
    
y_test = np.argmax(y_test,axis = 1)

pd.DataFrame({'ImageId': range(1, 28001), 'Label': y_test}).to_csv(r'submission.csv', index=False)

