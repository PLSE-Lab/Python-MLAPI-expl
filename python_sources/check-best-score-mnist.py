#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

# Any results you write to the current directory are saved as output.


# **Data Preparation**

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# Converting the Labels to 10 different classes
Y_train = train['label']
Y_train = to_categorical(Y_train, num_classes=10)


# In[33]:


# Preparing the data for X_train and reshaping it to (-1, 28, 28, 1) -> where -1 represents new shape should be compatible with original shape and
# 1 represents the channel 
# 28 and 28 represents the height and width of the resized images
# Dividing by 255 will normalize the values between 0 and 1
X_train = train.iloc[:,1:]
X_train = X_train/255
X_train = X_train.values.reshape(-1, 28, 28, 1)


# In[ ]:


# Preparing the data for X_test

X_test = test/255
X_test = X_test.values.reshape(-1,28,28,1)


# In[ ]:


# Check the shape of train and test data
print(X_train.shape)
print(X_test.shape)


# **Plot Figures**

# In[ ]:


plt.figure(figsize=(15,5))
for i in range(30):
    plt.subplot(3,10, i+1)
    plt.axis('off')
    plt.imshow(X_train[i].reshape(28,28))


# **Generate more images using Data Augmentation**

# In[ ]:


datagenerate = ImageDataGenerator(rotation_range=10, zoom_range=0.10, width_shift_range=0.1, height_shift_range=0.1)


# **Visualize Augmented Images**

# In[ ]:


X_train3 = X_train[9,].reshape((-1,28,28,1))
Y_train3 = Y_train[9,].reshape((1,10))
plt.figure(figsize=(15,4.5))

for i in range(30):
    plt.subplot(3,10,i+1)
    xt2, yt2 = datagenerate.flow(X_train3, Y_train3).next()
    plt.imshow(xt2[0].reshape((28,28)))
    plt.axis('off')
    if i==9:
        X_train3 = X_train[i].reshape(-1, 28, 28, 1)
    if i==19:
        X_train3 = X_train[i].reshape(-1, 28, 28, 1)


# ** 1.
# Let's see whether one, two, or three pairs is best. We are not doing four pairs since the image will be reduced too small before then. The input image is 28x28. After one pair, it's 14x14. After two, it's 7x7. After three it's 4x4 (or 3x3 if we don't use padding='same'). It doesn't make sense to do a fourth convolution. **

# In[ ]:


nets = 3
model = [0] *nets

for j in range(3):
    model[j] = Sequential()
    model[j].add(Conv2D(24,kernel_size=5,padding='same',activation='relu',
            input_shape=(28,28,1)))
    model[j].add(MaxPool2D())
    if j>0:
        model[j].add(Conv2D(48,kernel_size=5,padding='same',activation='relu'))
        model[j].add(MaxPool2D())
    if j>1:
        model[j].add(Conv2D(64,kernel_size=5,padding='same',activation='relu'))
        model[j].add(MaxPool2D(padding='same'))
    model[j].add(Flatten())
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(10, activation='softmax'))
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# Validation Set
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.333)


# In[ ]:


# Train Networks
history = [0] * nets
names = ["(C-P)x1","(C-P)x2","(C-P)x3"]
epochs = 20
for j in range(nets):
    history[j] = model[j].fit(X_train2, Y_train2, batch_size=80, epochs = epochs, validation_data = (X_val2, Y_val2), callbacks = [annealer], verbose=0)
    print("CNN {0}: Epoch={1:d}, Train_accuracy={2:.5f}, Validations={3:.5f}".format(names[j],epochs, max(history[j].history['acc']),max(history[j].history['val_acc'])))


# In[ ]:


# Plot accuracy
styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']
# print(history[0].history['val_acc'])
plt.figure(figsize=(15,7))
for i in range(nets):
    plt.plot(history[i].history['val_acc'], linestyle=styles[i])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(names, loc='upper left')
plt.show()


# > So we see that (C-P)x3 is better than others, so for me I am going to use this, despite efficiency being the most important parameter and we need to think about the computation costs too.

# **2. Lets check what should be the appropriate filter value? 
# [8,16,24,32,48,64]**

# In[ ]:


nets = 6
model = [0] * nets
for j in range(nets):
    model[j] = Sequential()
    model[j].add(Conv2D(j*8+8, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    model[j].add(MaxPool2D())
    model[j].add(Conv2D(j*16+16, kernel_size=(5,5), activation='relu'))
    model[j].add(MaxPool2D())
    model[j].add(Flatten())
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(10, activation='softmax'))
    model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Validation Set
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.333)


# In[ ]:


# Train Networks
history = [0] * nets
names = ["8 maps","16 maps","24 maps","32 maps","48 maps","64 maps"]
epochs = 20
for j in range(nets):
    history[j] = model[j].fit(X_train2, Y_train2, batch_size=80, epochs=epochs, validation_data = (X_val2, Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[j],epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))


# In[ ]:


# Plot accuracy
plt.figure(figsize=(15,6))
for i in range(nets):
    plt.plot(history[i].history['val_acc'], linestyle = styles[i])
plt.title('Model Accuracy for number of filters')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()


# > **So we see that 32 maps in first layer and 64 in second will be the best fit?**

# **3. How large a dense layer ?**

# In[ ]:


nets = 8
model = [0]*nets
for j in range(nets):
    model[j] = Sequential()
    model[j].add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    model[j].add(MaxPool2D())
    model[j].add(Conv2D(32, kernel_size=(5,5), activation='relu'))
    model[j].add(MaxPool2D())
    model[j].add(Flatten())
    if(j>0):
        model[j].add(Dense(2**(j+4), activation='relu'))
    model[j].add(Dense(10, activation='softmax'))
    model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Create validation 
history = [0]*nets
names = ["0N","32N","64N","128N","256N","512N","1024N","2048N"]
epochs = 20
for j in range(nets):
    history[j] = model[j].fit(X_train2, Y_train2, batch_size = 80, epochs = epochs, validation_data = (X_val2, Y_val2), callbacks = [annealer], verbose = 0)
    print("CNN {0}: Epochs={1:d} Training Accuracy={2:.5f}, Validation Accuracy={3:.5f}".format(names[j], epochs, max(history[j].history['acc']), max(history[j].history['val_acc'])))


# In[ ]:


# Plot Accuracy
plt.figure(figsize=(15,5))
for i in range(nets):
    plt.plot(history[i].history['val_acc'],linestyle=styles[i])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(names, loc='upper left')


# > **From this experiment, it appears that 128 units is the best**

# **4. How much dropout?
# **
# Dropout prevents the CNN from overfitting and helps us to classify better, so we add dropouts

# In[ ]:


nets = 8
model = [0]*nets

for j in range(nets):
    model[j] = Sequential()
    model[j].add(Conv2D(32, kernel_size=(5,5), input_shape = (28,28,1), activation='relu'))
    model[j].add(MaxPool2D())
    model[j].add(Dropout(j * 0.1))
    model[j].add(Conv2D(64,kernel_size=5,activation='relu'))
    model[j].add(MaxPool2D())
    model[j].add(Dropout(j*0.1))
    model[j].add(Flatten())
    model[j].add(Dense(128, activation='relu'))
    model[j].add(Dropout(j*0.1))
    model[j].add(Dense(10, activation='softmax'))
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


history = [0] * nr

