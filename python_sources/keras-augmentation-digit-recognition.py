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


# Loading data

# In[ ]:


train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


y = train_data['label']
X = train_data.drop(['label'],axis=1)


# In[ ]:


del train_data


# Checking for null values

# In[ ]:


X.isnull().all().unique()


# In[ ]:


y.isnull().any()


# In[ ]:


test_data.isnull().all().unique()


# Therefore we dont have any null values in both our test and train data

# # Visualization

# Lets plot the distributiom of labels

# In[ ]:


label_val = y.value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=label_val.index,y=label_val.values)


# Distribution of labels is relatively similar

# Right now we have flat pixels but our image should be a 2d image with single channel ( since we have grayscale images)

# So lets reshape the arrays

# In[ ]:


X = X.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)


# Plotting a single digit

# In[ ]:


plt.imshow(X[7][:,:,0])


# In[ ]:


X = X / 255
test_data = test_data / 255


# Now we have the data to work with , so lets start building the model using Keras

# In[ ]:


import warnings


# In[ ]:


warnings.filterwarnings('ignore')


# # Model building

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from keras.models import Sequential')


# In[ ]:


from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPool2D


# In[ ]:


from keras.utils.np_utils import to_categorical


# In[ ]:


y = to_categorical(y,num_classes=10)


# lets add a checkpoint too , which can be used to save best weights

# In[ ]:


from keras.callbacks import ModelCheckpoint


# In[ ]:


checkpoint = ModelCheckpoint('BWeight.md5',monitor='val_loss',
                            save_best_only=True)


# Designing the model

# In[ ]:


model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))


# Compiling the model

# In[ ]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Now that we have our model ready , lets apply some data augmentation

# # Data Preparation

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=True,
        zca_whitening=False,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)


# In[ ]:


datagen.fit(X)


# Let's fit the model

# In[ ]:


size_batch = 86


# Splitting the train data in train and validation splits

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42)


# # Training

# In[ ]:


history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=size_batch),
                              epochs = 10,
                              validation_data = (X_val,y_val),
                              verbose = 2,
                              steps_per_epoch = X_train.shape[0] // size_batch,
                              callbacks=[checkpoint])


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# # Predictions

# Loading best weights

# In[ ]:


model.load_weights('BWeight.md5')


# In[ ]:


FINAL_PREDS = model.predict_classes(test_data)


# In[ ]:


results = pd.Series(FINAL_PREDS,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("mnist_cnn_keras.csv",index=False)


# In[ ]:




