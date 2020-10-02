#!/usr/bin/env python
# coding: utf-8

# This is a modified version of my Keras CNN for the MNIST dataset. I saw my score drop on the leaderboard and decided to tweak things a bit. Unfortunately, the LB seems to be dominated by suspicious 100% result nowadays, so this might not help ... 
# 
# Here goes:

# In[ ]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[ ]:


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# In[ ]:


train_file = "../input/train.csv"
test_file = "../input/test.csv"
output_file = "submission.csv"


# ## Load the data

# In[ ]:


raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    raw_data[:,1:], raw_data[:,0], test_size=0.1)


# In[ ]:


x_train = x_train.astype("float32")/255.0
x_val = x_val.astype("float32")/255.0
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)


# In[ ]:


n_train = x_train.shape[0]
n_val = x_val.shape[0]
x_train = x_train.reshape(n_train, 28, 28, 1)
x_val = x_val.reshape(n_val, 28, 28, 1)
n_classes = y_train.shape[1]


# ## Train the model

# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(MaxPool2D(strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(MaxPool2D(strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# If you run this on a strong machine, over hundreds of epochs, augmentation will improve your performance. This means randomly perturbing the images to prevent overfitting, and Keras has a simple function for this. 
# 
# Here in the Kernel, we will only look at each image 4-5 times, so augmentation will only add a little 

# In[ ]:


datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)


# These parameters here again chosen for the Kaggle kernel. For better performance, try reducing the learning rate and increase the number of epochs. I was able to reach 99.5% with a similar net on longer training.

# In[ ]:


#Start with a slightly lower learning rate
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=3e-5), metrics = ["accuracy"])

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 16),
                           steps_per_epoch = 1000, 
                           epochs = 1, verbose = 2,
                           validation_data = (x_val[:400,:], y_val[:400,:]))


# We will use a small learning rate for the first epoch:

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=3e-5), metrics = ["accuracy"])


# Then we speed things up, only to reduce it by 10% each epoch. This Keras function does all this for us:

# In[ ]:


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9**x)


# In[ ]:


hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 16),
                           steps_per_epoch = 1000, #Increase this when not on Kaggle kernel
                           epochs = 9, #Increase this when not on Kaggle kernel
                           verbose = 2,  #verbose=1 outputs ETA, but doesn't work well in the cloud
                           validation_data = (x_val[:400,:], y_val[:400,:]), #To evaluate faster
                           callbacks = [annealer])


# ## Evaluate

# We only used a subset of the validation set during training, to save time. Now let's check performance on the whole validation set.

# In[ ]:


model.evaluate(x_val, y_val, verbose=0)


# In[ ]:


plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()


# In[ ]:


y_hat = model.predict(x_val)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)


# Not too bad, considering the minimal amount of training so far. In fact, we have only gone through the training data approximately five times. With proper training we should get really good results, but then we might need to add more Dropout to prevent overfitting.

# ## Submit

# In[ ]:


mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
x_test = mnist_testset.astype("float32")/255.0
n_test = x_test.shape[0]
x_test = x_test.reshape(n_test, 28, 28, 1)


# In[ ]:


y_hat = model.predict(x_test, batch_size=64)


# y_test consists of class probabilities, I now select the class with highest probability

# In[ ]:


y_pred = np.argmax(y_hat,axis=1)


# In[ ]:


with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(0, n_test) :
        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))


# Submitting from this notebook usually gives slightly lower than 99%, but I achieved 99.3% by averaging over 5 good runs. And you can get higher than that if you train overnight.

# In[ ]:




