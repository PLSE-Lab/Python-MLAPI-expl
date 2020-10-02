#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer
# 
# 2 Models mentioned below will be used for training:
# 
# 1. NN - Simple 3 layer Neural Network - Keras
# 2. CNN - Convolutional Neural Network - Keras

# In[ ]:


import numpy as np
import pandas as pd

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# ### 1. Importing Dataset

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# ### 2. Preprocessing

# In[ ]:


X_train = train.drop(labels = ["label"], axis=1)
Y = train["label"]
Y_train = keras.utils.to_categorical(Y)


# ### 2.1 Normalize features

# In[ ]:


X_train = X_train / 255.
test = test / 255.
# convert Y_train to one_hot
del train


# ### 2.2 Check if there is any missing data

# In[ ]:


print(np.sum(np.isnan(X_train.values)))
print(np.sum(np.isnan(test.values)))


# ### 2.3 Plot some examples

# In[ ]:


fig = plt.figure(figsize=(10,10))

for i in range(20):
    plt.subplot(5,4,i+1)
    plt.imshow(X_train.iloc[i].values.reshape(28,28),cmap=plt.cm.binary)


# ### 2.4 Analyse the class distributions

# In[ ]:


sns.countplot(Y)


# # Models

# ## Model 1 - Simple Neural Network (Keras)
# 
# Accuracy on Kaggle learderboard: **%97.2**
# 
# Model has the features mentioned below:
# * 3 layers (2 hidden + 1 output) - 784 --> (64,64) -->  (10)
# * Learning Rate Reduction - LR will decrease when validation accuracy is not improved for 2 epochs in a row
# * Regularization - 2 hidden layers will have L2 regularization

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


from keras import regularizers
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[ ]:


# trainining
history = model.fit(X_train, Y_train, epochs=40, batch_size=128, validation_split=0.1, callbacks = [learning_rate_reduction])


# In[ ]:


# evaluate model on training data
res = model.evaluate(X_train, Y_train)
print("Model accuracy on training data is: %{}".format(res[1]*100))


# In[ ]:


# Create submission
submission_df = pd.read_csv("../input/sample_submission.csv", index_col=0)
probs = model.predict(test)
submission_df['Label'] = np.argmax(probs,axis=1)
submission_df.head()


# ## Model 2 - Convolutional Neural Network (Keras)
# 
# Accuracy on Kaggle learderboard: **%99.3** 
# 
# Model used is a VGG-like convnet from Keras website:
# 
# 

# In[ ]:


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

model.summary()


# In[ ]:


X_train_CNN = X_train.values.reshape((-1,28,28,1))
X_train_CNN.shape


# In[ ]:


model.fit(X_train_CNN, Y_train, batch_size=128, epochs=30)


# In[ ]:


# evaluate model on training data
res2 = model.evaluate(X_train_CNN, Y_train)
print("Model accuracy on training data is: %{}".format(res2[1]*100))


# #### Alternatively augmentation can be used to train the model.

# In[ ]:


# datagen = ImageDataGenerator(
#     rotation_range = 15,
#     zca_whitening = True,
#     width_shift_range=0.15,
#     height_shift_range=0.15,
#     )
# datagen.fit(X_train_CNN)

# history = model.fit_generator(datagen.flow(X_train_CNN,Y_train, batch_size=32),
#                               epochs = 10,
#                               verbose = 1, steps_per_epoch=steps_per_epoch=X_train_CNN.shape[0] // 32)


# #### Create Submission

# In[ ]:


submission_df = pd.read_csv("../input/sample_submission.csv", index_col=0)
probs = model.predict(test.values.reshape((-1,28,28,1)))
submission_df['Label'] = np.argmax(probs,axis=1)
submission_df.head()

