#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import random


# In[ ]:


import tensorflow as tf

from sklearn.model_selection import train_test_split


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU


# # EDA Phase

# In[ ]:


path = "../input/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
sample_submission = pd.read_csv(path + "sample_submission.csv")


# In[ ]:


# plot a random sample image file
num = random.randint(0, len(train))

label_sample, image = train.iloc[num,0], train.iloc[num,1:]
shape = int(np.sqrt(784))
plt.imshow(image.values.reshape(shape, shape))
print("The image should show: {}".format(label_sample))


# In[ ]:


train.isnull().values.sum()


# As shown int the above training set. The pixels in the image are in the range from 0 to 255. We should normalize it for CNN

# In[ ]:


y_train = train.label
x_train = train.drop(['label'], axis=1)


# In[ ]:


x_train = (x_train / 255) - 0.5
test = (test / 255) - 0.5


# Let's see how many digits are in the training set.

# In[ ]:


sns.countplot(y_train)


# Split train and validation data.
# Convert labels to categorical matrix

# In[ ]:


# reshape x_train and x_test to 3D image
x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)


# In[ ]:


x_part, x_val, y_part, y_val = train_test_split(
    x_train, y_train, random_state=42)


# In[ ]:


# to_categorical for y_part and y_val
y_part_oh = to_categorical(y_part, 10)
y_val_oh = to_categorical(y_val, 10)


# CNN for MNIST digits classification based on AlexNet (5 CONV layers and 3 FC layers)

# In[ ]:


def make_model():
    # Feed-forward network 
    model = Sequential()

    # CNN layers
    # First two CNN layers with maxpooling and dropout
    model.add(Conv2D(16, (3, 3), 
                     padding='same',
                     input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    #2nd two CNN layers with maxpooling and dropout
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    # FC layers with dropout
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    
    # FC Layers
    model.add(Dense(10, activation='softmax'))
    
    return model


# In[ ]:


s = tf.keras.backend.clear_session()
model = make_model()
model.summary()


# In[ ]:


#Optimizer

model.compile(
    loss='categorical_crossentropy', # this is our cross-entropy
    optimizer='adam',
    metrics=['accuracy']  # report accuracy during training
)


# In[ ]:


# Training with model
epochs=40
model_log = model.fit(
    x_part, 
    y_part_oh,
    batch_size=32, 
    epochs=epochs,
    validation_data=(x_val, y_val_oh),
    verbose=10,
)


# # Accuracy Plot and Confusion Matrix

# In[ ]:


plt.figure()
# Accuracy Plot
plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

# Loss Function Plot
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()


# In[ ]:


# Confusion Matrix
y_pred_val = model.predict_proba(x_val)
y_pred_classes = np.argmax(y_pred_val, axis=1)
y_pred_max_probas = np.max(y_pred_val, axis=1)

digit_classes = [i for i in range(10)]


# In[ ]:


# confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score

plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_val, y_pred_classes))
plt.xticks(np.arange(10), digit_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), digit_classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_val, y_pred_classes))


# # Data Augmentation
# 
# To improve the accuracy score, we can preprocess the image with image augmentation such as rotation, zoom-in and slice, etc
# 
# https://keras.io/preprocessing/image/

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10, # 15 degree rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
)

datagen.fit(x_part)
model.fit_generator(datagen.flow(x_part, y_part_oh, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)


# # Submission

# In[ ]:


test_pred = model.predict(test)
test_result = np.argmax(test_pred, axis=1)
test_result = pd.Series(test_result,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),test_result],axis = 1)
submission.to_csv("submission.csv",index=False)


# In[ ]:




