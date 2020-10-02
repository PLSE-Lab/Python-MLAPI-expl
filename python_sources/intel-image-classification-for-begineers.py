#!/usr/bin/env python
# coding: utf-8

# # **Imports and Image Preprocessing**

# In[ ]:


# setup
import pandas as pd

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping


# In[ ]:


# data path
train_path = '../input/intel-image-classification/seg_train/seg_train/'
test_path = '../input/intel-image-classification/seg_test/seg_test/'


# In[ ]:


# data augmentation
train_augment = ImageDataGenerator(rescale=1./255, rotation_range=35, 
                                  width_shift_range=0.2, height_shift_range=0.2,
                                  shear_range=0.25, zoom_range=0.2,vertical_flip=True)
test_augment = ImageDataGenerator(rescale=1./255)


# In[ ]:


# data
train_data = train_augment.flow_from_directory(directory=train_path, target_size=(150, 150),
                                              color_mode='rgb', class_mode='categorical', batch_size=32)
test_data = test_augment.flow_from_directory(directory=test_path, target_size=(150, 150),
                                              color_mode='rgb', class_mode='categorical', batch_size=32)


# # **Model No. 01**

# In[ ]:


# model
model_1 = Sequential()

# layers
model_1.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(150, 150, 3)))
model_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model_1.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model_1.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model_1.add(Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model_1.add(Convolution2D(filters=528, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model_1.add(Flatten())
model_1.add(Dropout(rate=0.25))

model_1.add(Dense(units=350, activation='relu'))
model_1.add(Dropout(rate=0.2))

model_1.add(Dense(units=6, activation='softmax'))

# compile
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summary
model_1.summary()


# In[ ]:


# Early Stopping
earlystop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')


# In[ ]:


# Training
model_1.fit(x=train_data, validation_data=(test_data),
           epochs=60, verbose=1, callbacks=[earlystop])


# In[ ]:


# Report
report_1 = pd.DataFrame(model_1.history.history)
report_1.head()


# In[ ]:


# Report Visulation
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
report_1[['loss', 'val_loss']].plot(ax=axes[0])
report_1[['accuracy', 'val_accuracy']].plot(ax=axes[1])


# In[ ]:


# Evaluation on Test Data
val_loss, val_accuracy = model_1.evaluate(test_data)
print('Validation loss : {}'.format(val_loss))
print('Validation accuracy : {}'.format(val_accuracy))


# # **Model No. 02**

# In[ ]:


# model
model_2 = Sequential()

# layers
model_2.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(150, 150, 3)))
model_2.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model_2.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
model_2.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model_2.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
model_2.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model_2.add(Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
model_2.add(Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model_2.add(Flatten())
model_2.add(Dropout(rate=0.2))

model_2.add(Dense(units=550, activation='relu'))
model_2.add(Dropout(rate=0.15))

model_2.add(Dense(units=150, activation='relu'))
model_2.add(Dropout(0.1))

model_2.add(Dense(units=6, activation='softmax'))

# compile
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summary
model_2.summary()


# In[ ]:


# Early Stopping
earlystop = EarlyStopping(monitor='val_loss', patience=16, verbose=1, mode='min')


# In[ ]:


# Training
model_2.fit(x=train_data, validation_data=(test_data),
           epochs=60, verbose=1, callbacks=[earlystop])


# In[ ]:


# Report
report_2 = pd.DataFrame(model_2.history.history)
report_2.head()


# In[ ]:


# Report Visulation
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
report_2[['loss', 'val_loss']].plot(ax=axes[0])
report_2[['accuracy', 'val_accuracy']].plot(ax=axes[1])


# In[ ]:


# Evaluation on Test Data
val_loss, val_accuracy = model_2.evaluate(test_data)
print('Validation loss : {}'.format(val_loss))
print('Validation accuracy : {}'.format(val_accuracy))

