#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')


# In[ ]:


y = train_data['label']
X = train_data.drop(['label'],axis=1)


# In[ ]:


X_test = test_data.drop(['label'],axis=1)
y_test = test_data['label']


# In[ ]:


del train_data


# In[ ]:


del test_data


# In[ ]:


X.isnull().all().unique()


# In[ ]:


y.isnull().any()


# In[ ]:


X_test.isnull().all().unique()


# In[ ]:


y_test.isnull().all()


# Therefore we dont have any null values in both our test and train data

# Lets plot the distribution of labels

# # Visualization

# In[ ]:


label_val = y.value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=label_val.index,y=label_val.values)


# Distribution of labels is relatively similar

# Right now we have flat pixels but our image should be a 2d image with single channel ( since we have grayscale images)
# So lets reshape the arrays

# In[ ]:


X = X.values.reshape(-1,28,28,1)


# In[ ]:


X_test = X_test.values.reshape(-1,28,28,1)


# Plotting images

# In[ ]:


plt.imshow(X[0][:,:,0])


# In[ ]:


plt.imshow(X[7][:,:,0])


# In[ ]:


X = X / 255
X_test = X_test / 255


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# Now we have the data to work with , so lets start building the model using Keras

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPool2D
from keras.utils.np_utils import to_categorical


# In[ ]:


y = to_categorical(y,num_classes=10)


# Lets add a checkpoint too , which can be used to save best weights

# # Model Building

# In[ ]:


from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('BWeight.md5',monitor='val_loss',
                            save_best_only=True)


# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# Compiling the model

# In[ ]:


from keras import optimizers
sgd = optimizers.SGD(lr=0.001, decay=1e-8, momentum=0.9, nesterov=True)


# In[ ]:


model.compile(optimizer=sgd,loss = 'categorical_crossentropy', metrics=['accuracy'])


# Now that we have our model ready , lets apply some data augmentation

# # Data Preparation

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zoom_range = 0.2,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=False)


# In[ ]:


datagen.fit(X)


# Splitting the train data in train and validation

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1)


# In[ ]:


size_batch = 28


# # Training

# In[ ]:


history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=size_batch),
                              epochs = 10,
                              validation_data = (X_val,y_val),
                              verbose = 2,
                              steps_per_epoch = X_train.shape[0] // size_batch,
                              callbacks=[checkpoint],
                             use_multiprocessing=True)


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# Loading best weights

# In[ ]:


model.load_weights('BWeight.md5')


# Predicting labels of our test data

# # Predictions

# In[ ]:


y_final_preds = model.predict_classes(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,y_final_preds))


# In[ ]:




