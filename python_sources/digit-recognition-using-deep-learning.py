#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# This Notebook models a simple but fast and accurate Convolutional Neural Networks for recognizing handwritten digits of MNIST dataset.
# ## 1.1. Import libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau


# # 2. Data Prepration
# At the beginning I try to prepare the dataset and do some preprocessing in order to get the maximum desirable output.
# ## 2.1. Load Data

# In[2]:


train = pd.read_csv("../input/train.csv").values
test = pd.read_csv("../input/test.csv").values

X_train = train[:, 1:].astype('float32')
Y_train = train[:, 0].astype('int32')
X_test = test[:, :].astype('float32')


# ## 2.2. Data Preprocessing
# The recommended preprocessing is to center the data to have mean of zero, and normalize its scale to [-1, 1] along each feature. I skipped the standardization process since has particular effect but it is good practice to use it.
# 
# **Normalization** refers to normalizing the data dimensions so that they are of approximately the same scale. I normalized each feature scale to [0, 1] since it was simpler.
# 
# **Reshape:** for every input we have 784 feature in 1-dimension and we should reshape it into 2-dimensional images with 1 channel (gray image) so i reshaped the 1x784 input vector into 28x28x1.
# 
# **Label One Hot Encoding:** our labels varies between a single integer from 0 to 9 and we should convert it to one hot vector. a one hot encoded vector has a single 1 in one dimension and 0 in other dimensions. for example a label 7 in one hot vector is [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] in this process.
# 
# **Train/Validation Data Split:** for splitting data into train and validation first we have to check the distribution of our classes.

# In[3]:


sns.countplot(Y_train)


# Knowing that we have almost the same proportion of data for each class leads us to use train validation random split without using stratify option.
# 
# **Data Augmentation:** in order to improve our model generalization it is common practice to use data augmentation. This means generating more data for training process. This data is generated based on actual training data with random small manipulation such as zoom, shift, rotation, crop, flip, etc.

# In[4]:


# Normalization
X_train = X_train/255.
X_test = X_test/255.

# Reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Label One Hot Encoding
Y_train = to_categorical(Y_train, num_classes=10)

# Train/Validation Data Split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=23)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15)
datagen.fit(X_train)


# # 3. CNN Model
# ## 3.1. Model Definition
# I used the keras Sequential API to define the model.
# 
# For the first Conv2D layer i choosed 16 filters each use 25 weights. I used other layers such as BachNormalization, MaxPooling2D and Dropout.
# 
# **Batch Nomralization** is a technique [developed by Ioffe and Szegedy](http://arxiv.org/abs/1502.03167). In practice networks that use Batch Normalization are significantly more robust to bad initialization.
# 
# **Dropout** is a Regularization technique that controls the capacity of Neural Networks to prevent overfitting [introduced by Srivastava et al](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf).
# 
# **Max Pooling** is another layer to prevent overfitting. The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned.
# 
# **Dense** is fully connected layer.
# 
# All layers use **RelU** (Rectified Linear Unit) activation function.
# 
# For optimizer i choosed **RMSprop** which is very effective adaptive learning rate method.
# ![Contours of a loss surface and time evolution of different optimization algorithms](http://cs231n.github.io/assets/nn3/opt2.gif)

# In[5]:


# Model Definition
model = Sequential()

model.add(Conv2D(16, kernel_size=(5, 5),
                activation='relu',
                input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=RMSprop(lr=1e-3),
              metrics=['accuracy'])


# ## 3.2. Train
# For the training process i choosed batch size of 70 and 20 number of epoches. also to converge faster i reduced learning rate to 0.0002 when validation loss does not improve after 2 epochs and kept it for 2 more epochs.

# In[6]:


# Learning Rate Reducer
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=2)
# Train
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=70),
                              epochs=20,
                              validation_data=(X_val, Y_val),
                              verbose=2,
                              callbacks=[reduce_lr])


# ## 3.3. Evaluation
# To evaluate the model i plot the confusion matrix on validation data.

# In[7]:


Y_val_pred = model.predict_classes(X_val)
Y_val_true = np.argmax(Y_val, axis=1)

print(confusion_matrix(Y_val_true, Y_val_pred))


# ## 3.4. Submission
# Finally we use this model to predict test set labels and submit.

# In[9]:


Y_test_pred = model.predict_classes(X_test)

submission = pd.DataFrame({ 'ImageId': range(1, 28001), 'Label': Y_test_pred })
submission.to_csv("submission.csv", index=False)

