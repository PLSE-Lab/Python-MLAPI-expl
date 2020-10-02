#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


# In[ ]:


train_files = glob.glob("../input/fingers/fingers/train/*.png")
test_files = glob.glob("../input/fingers/fingers/test/*.png")


# ### Show an image and convert to array

# In[ ]:


train_files[0]


# In[ ]:


from PIL import Image
im = Image.open(train_files[0])
plt.imshow(im);


# In[ ]:


im_array = np.array(im)


# In[ ]:


im_array.shape


# ### Load train and test data

# In[ ]:


X_train = np.zeros((len(train_files), 128, 128))
Y_train = np.zeros((len(train_files), 6))


# In[ ]:


for i, trf in enumerate(train_files):
    im = Image.open(trf)
    X_train[i, :, :] = np.array(im)
    Y_train[i, int(trf[-6:-5])] = 1


# In[ ]:


X_test = np.zeros((len(test_files), 128, 128))
Y_test = np.zeros((len(test_files), 6))


# In[ ]:


for i, tsf in enumerate(test_files):
    im = Image.open(tsf)
    X_test[i, :, :] = np.array(im)
    Y_test[i, int(tsf[-6:-5])] = 1


# In[ ]:


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# ### Train model

# In[ ]:


import tensorflow as tf


# In[ ]:


tf.__version__


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


# In[ ]:


model = Sequential()
model.add(Conv2D(64, (3,3), strides=(1, 1), input_shape = (128, 128, 1), padding='same', activation = 'relu'))
model.add(MaxPool2D((8,8)))
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(6, activation = 'softmax'))
model.summary()


# #### Scale data

# In[ ]:


X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)/255
X_test = X_test.reshape(X_test.shape[0], 128, 128, 1)/255


# In[ ]:


X_train[0]


# In[ ]:


model.compile('SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = X_train, y = Y_train, batch_size = 128, epochs = 10, validation_split=0.2)


# In[ ]:


Y_pred_test = model.predict_classes(X_test)


# #### Check model perfomance by looking at confusion matrix and classification report on the test data

# In[ ]:


from sklearn import metrics


# In[ ]:


print(metrics.confusion_matrix(np.argmax(Y_test, axis=1), Y_pred_test))


# In[ ]:


print(metrics.classification_report(np.argmax(Y_test, axis=1), Y_pred_test, digits=3))


# #### Convert and save model

# In[ ]:


cnn_model = model


# In[ ]:


converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)


# In[ ]:


tflite_model = converter.convert()
open("converted_model_fingers_cnn.tflite", "wb").write(tflite_model)

