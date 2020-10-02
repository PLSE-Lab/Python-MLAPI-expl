#!/usr/bin/env python
# coding: utf-8

# Hello again, my second kernel here. Best, Behrouz

# ## 1. Import Python libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
from skimage import io
from PIL import Image

# from sklearn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# from keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils import to_categorical


# ## 2. Image labels as a DataFrame

# In[ ]:


path = '../input/intel-image-classification/seg_train/seg_train'

# Labels as a dictionary
image_labels = {'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}

# Labels as a dictionary (inverse)
image_labels_inv = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}


# create an empty list
labels = []

# Loop over images
for folder in os.listdir(path):
    for file in os.listdir(Path(path, folder)):
        img = Image.open(Path(path, folder, file))
        labels.append([(int(os.path.splitext(file)[0])), image_labels.get(folder)])
        
labels = pd.DataFrame(labels, columns=['id','label'])
labels = labels.set_index('id')
labels.head()


# ## 3. Normalize image data

# In[ ]:


# initialize standard scaler
ss = StandardScaler()

image_list = []
for i in labels.index:
    # load image
    filename = '{}.jpg'.format(i)
    folder = image_labels_inv.get(labels.loc[i, 'label'])
    filepath = Path(path, folder, filename)
    img = Image.open(filepath)
    img = np.array(img.resize((150, 150))).astype(np.float32)
    
    # for each channel, apply standard scaler's fit_transform method
    for channel in range(img.shape[2]):
        img[:, :, channel] = ss.fit_transform(img[:, :, channel])
        
    # append to list of all images
    image_list.append(img)
    
# convert image list to single array
X = np.array(image_list)

# print shape of X
print(X.shape)


# In[ ]:


# assign the label values to y
y = labels.label.values
y = to_categorical(y)
print(y.shape)


# ## 4. Split into train and test sets
# 

# In[ ]:


# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

# examine number of samples in train and test ets
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# ## 5. Model building

# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='tanh', input_shape=(150, 150, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))    

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.summary()


# ## 6. Compile and train model

# In[ ]:


model.compile(
    # set the loss
    loss = keras.losses.categorical_crossentropy,
    # set the optimizer
    optimizer = keras.optimizers.SGD(),
    # set the metric as accuracy
    metrics = ['accuracy']
)


# In[ ]:


# mock-train the model using the train and test sets
trained_cnn = model.fit(X_train, y_train, epochs=150, verbose=1, validation_data=(X_test, y_test))


# ## 7. Visualize model training history
# 

# In[ ]:


# print keys for pretrained_cnn_history dict
print(trained_cnn.history.keys())

plt.figure(1)

#plt.subplot(211)
# plot the accuracy
plt.plot(trained_cnn.history['val_acc'], 'r')
plt.plot(trained_cnn.history['acc'], 'b')
plt.title('Validation accuracy and loss')
plt.legend(['val_acc', 'acc'])
plt.ylabel('Accuracy')

plt.figure(2)
#plt.subplot(212)
# plot the loss
plt.plot(trained_cnn.history['val_loss'], 'r')
plt.plot(trained_cnn.history['loss'], 'b')
plt.legend(['val_loss', 'loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss value');

