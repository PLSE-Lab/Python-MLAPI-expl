#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ### Prepare data
# 
# 1. Read Data with pandas DataFrame
# 2. Parse the "label" column(axis=1) as y_train
# 3. Check digits distribution

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


X_train = train.drop(labels=['label'], axis=1)
y_train = train['label']


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


print(y_train.value_counts(sort=False))
sns.countplot(y_train)


# ### Reshape data as an image matrix
# 
# Reshape data array to image matrix
#          [-1, 784] => [-1, 28, 28, 1]
#          
# Shape of input data for keras.Conv2D should be [samples, heigth, width, channels]

# In[ ]:


X_train = X_train.values.astype('float32').reshape(-1, 28, 28, 1)
X_test = test.values.astype('float32').reshape(-1, 28, 28, 1)


# ### Data visualization
# 
# Display the digit image

# In[ ]:


for cnt, i in enumerate([10, 20, 30]):
    plt.subplot(330 + (cnt+1))
    plt.imshow(X_train[i][:, :, 0], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i])


# ### Normalization
# Normalize data from [0..255] to [0..1]

# In[ ]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# ### One hot encoding
# Encode digit labels to one hot vector, e.g. 6 => [0,0,0,0,0,1,0,0,0,0]

# In[ ]:


from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)


# In[ ]:


y_train[0]


# ### Split data
# Split data to be train set and validation set
# 
# Set validation set to be 10%

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)


# ### CNN
# 1. Design model
# 2. Set optimizer

# In[ ]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', 
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', 
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', 
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', 
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))          


# In[ ]:


model.summary()


# In[ ]:


optimizer = RMSprop(lr=0.001, epsilon=1e-8)


# In[ ]:


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# ### Data augmentation

# In[ ]:


from keras.preprocessing import image
batch_size = 64

datagen = image.ImageDataGenerator(
     rotation_range=8,
     width_shift_range=0.08,
     shear_range=0.3,
     height_shift_range=0.08,
     zoom_range=0.08
)


# In[ ]:


train_batches = datagen.flow(X_train, y_train, batch_size=batch_size)
val_batches = datagen.flow(X_val, y_val, batch_size=batch_size)


# ### Learning rate annealer

# In[ ]:


from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.5,
                                            patience=5,
                                            min_lr=1e-5)


# ### Fit model

# In[ ]:


epochs = 30


# In[ ]:


history = model.fit_generator(
    generator=train_batches,
    steps_per_epoch=X_train.shape[0]//batch_size,
    validation_data=val_batches,
    validation_steps=X_val.shape[0]//batch_size,
    epochs=epochs, 
    verbose=2,
    callbacks=[learning_rate_reduction])


# In[ ]:


plt.figure(figsize=[12, 10])
plt.subplot(211)
plt.plot(history.history['loss'], color='b', label='Train loss')
plt.plot(history.history['val_loss'], color='r', label='Valid loss')
plt.subplot(212)
plt.plot(history.history['acc'], color='b', label='Train accuracy')
plt.plot(history.history['val_acc'], color='r', label='Valid accuracy')
legend = plt.legend()


# In[ ]:


y_pred = model.predict(X_val)
y_pred_num = np.argmax(y_pred, axis=1)
y_val_num = np.argmax(y_val, axis=1)
confusion_matrix(y_val_num, y_pred_num)


# ### Submit

# In[ ]:


# predict results
results = model.predict(X_test)

# select the indix with the maximum probability
results = np.argmax(results, axis = 1)

results = pd.Series(results, name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), results],axis = 1)
submission.to_csv("MNIST-CNN.csv", index=False)

