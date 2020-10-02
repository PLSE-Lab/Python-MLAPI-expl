#!/usr/bin/env python
# coding: utf-8

# # Welcome to the Areal Cactus Identification Competition
# This notebook is a starter code for all beginners and easy to understand. To predict the test data a simple CNN with a pretrained model is used.

# # Load libraries and pretrained model

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
print(os.listdir("../input/aerial-cactus-identification"))
print(os.listdir('../input/models'))


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import RMSprop,Adam
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator


# # Define parameters

# In[ ]:


q_size = 32
img_channel = 3 
num_classes=2
id_code = 'id'


# # Read the input csv files

# In[ ]:


path = '../input/aerial-cactus-identification/'
train_data = pd.read_csv(path+'train.csv')
sub_org = pd.read_csv(path+'sample_submission.csv')


# # Define some functions

# In[ ]:


def plot_bar(data):
    """Simple function to plot the distribution of the classes."""
    dict_data = dict(zip(range(0, num_classes), (((data.value_counts()).sort_index())).tolist()))
    names = list(dict_data.keys())
    values = list(dict_data.values())
    plt.bar(names, values)
    plt.grid()
    plt.show()


# In[ ]:


def read_images(filepath, data, file_list, size):
    """Read and edit the images of a given folder."""
    for file in file_list:
        img = cv2.imread(filepath+file)
        img = cv2.resize(img, (size, size))
        data[file_list.index(file), :, :, :] = img


# # Initialize the original train and test data

# In[ ]:


X_train_org = np.empty((len(train_data), q_size, q_size, img_channel), dtype=np.uint8)
X_test = np.empty((len(sub_org), q_size, q_size, img_channel), dtype=np.uint8)


# # Read the image data

# In[ ]:


read_images(path+'train/train/', X_train_org, train_data[id_code].tolist(), q_size)
read_images(path+'test/test/', X_test, sub_org[id_code].tolist(), q_size)


# In[ ]:


plot_bar(train_data['has_cactus'])


# In[ ]:


class_weight = dict(zip(range(0, num_classes), (((train_data['has_cactus'].value_counts()).sort_index())/len(train_data)).tolist()))


# In[ ]:


class_weight


# In[ ]:


y_train_org = train_data['has_cactus'].tolist()
y_train_org = to_categorical(y_train_org, num_classes = num_classes)


# # Convert and scale image data

# In[ ]:


mean = X_train_org.mean(axis=0)
X_train_org = X_train_org.astype('float32')
X_train_org -= X_train_org.mean(axis=0)
std = X_train_org.std(axis=0)
X_train_org /= X_train_org.std(axis=0)
X_test = X_test.astype('float32')
X_test -= mean
X_test /= std


# # Split train and validation data

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org, test_size = 0.2, random_state=0)


# # Create the model

# In[ ]:


conv_base = VGG16(weights='../input/models/model_weights_vgg16.h5',
                  include_top=False,
                  input_shape=(q_size, q_size, img_channel))
conv_base.trainable = True


# In[ ]:


model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))


# In[ ]:


model.compile(optimizer = Adam(lr=1e-6),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])


# In[ ]:


model.summary()


# In[ ]:


epochs = 10
batch_size = 16


# # Define the ImageDataGenerator

# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=False,
        vertical_flip=False)
datagen.fit(X_train)


# # Train model

# In[ ]:


history = model.fit(datagen.flow(X_train, y_train,
                                 batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    class_weight=class_weight)


# # Predict the test data

# In[ ]:


y_test = model.predict(X_test)


# In[ ]:


y_test_classes = np.argmax(y_test, axis = 1)


# # Write output for submission

# In[ ]:


output = pd.DataFrame({'id': sub_org['id'],
                       'has_cactus': y_test_classes})
output.to_csv('submission.csv', index=False)


# In[ ]:


plot_bar(output['has_cactus'])


# # Analyse the results

# In[ ]:


loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='Loss_Train')
plt.plot(epochs, loss_val, 'b', label='Loss_Val')
plt.title('value loss funtion')
plt.xlabel('epochs')
plt.ylabel('value loss funtion')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


acc = history.history['binary_accuracy']
acc_val = history.history['val_binary_accuracy']
epochs = range(1, len(loss)+1)
plt.plot(epochs, acc, 'bo', label='Accuracy_Train')
plt.plot(epochs, acc_val, 'b', label='Accuracy_Val')
plt.title('value of accuracy')
plt.xlabel('epochs')
plt.ylabel('value of accuracy')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


del model

