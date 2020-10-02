#!/usr/bin/env python
# coding: utf-8

# **Pleae look at the version 4 for the correct kernal**
# * Hi, here we will Develop a keras way model from scratch.
# 1. First we will use the normal Keras Sequential way, then will move on to the, FCs, CNNs.

# In[ ]:


import os
print(os.listdir("../input")) # looking at the dicrectory


# We then load all the libraries for the model.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# fix random seed for reproducibility
np.random.seed(43)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,Activation
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train=pd.read_csv("../input/train.csv")
train.head() # looking at the trainning dataset


# In[ ]:


train.shape


# In[ ]:


test=pd.read_csv("../input/test.csv")
test.head() # looking at the trainning dataset


# **DATA PREPARATION**

# By looking at the trainning set, we should separate the label column which is the value set (target values) of our trainning dataset
# 
# In the given data
# 1. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. 
# 2. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. 
# 3. This pixel-value is an integer between 0 and 255, inclusive.
# 
# * Now we sperate the labels from the tranning set 
# * Hot encoding for the labels to represent as vector.
# * We Normalize the data to centre the data around zero mean and unit variance.

# In[ ]:


"""It is important preprocessing step. It is used to centre the data around zero mean and unit variance."""
img_rows, img_cols = 28, 28 # height and width pf pixels
num_classes = 10 # total number of labels
import keras
def data_prep(raw):
    train_y = keras.utils.np_utils.to_categorical(raw.label, num_classes) # hot encoding for the labels 
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    train_x = (x_shaped_array.astype('float32') / 255.) #normalizing the data 
    return train_x, train_y
train_x, train_y = data_prep(train)


#  **Building Models**

# In[ ]:



from keras.utils.np_utils import to_categorical
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)


# In[ ]:


# Set the random seed
random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


# **Lets start with simple Linear model**
# 1.  In building a keras layer we need to define the input dimensions of our data in the 1st layer
# 1. Now add Desne layer which is fully connected layer with N(desired) number of neurons
# 2. In the last Dense layer we need to define the output dimensions/classes of the model.

# In[ ]:



model=Sequential()
model.add(Dense(64,input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Flatten()) #converts into a vector 
model.add(Dense(10, activation="softmax"))
model.summary()


# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(train_x, train_y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)


# Wow we got an accuracy of 91% for simple dense layer. But its a kind of low. lets move on to the Convolutional Neural Network and see how it could help us.

# Now we Split the tranning data randomly with seed=42

# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size = 0.1, random_state=2)


# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


model.summary()


# In[ ]:


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


model.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])


# **Data Agumentation**
# Here we change the image rotation, width shift, height, we zoom a little bit, by doing all this we can increase over data almost double to the present and make our model more roboust.
# Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.

# In[ ]:


from keras.preprocessing import image
gen = image.ImageDataGenerator()
#datagen=ImageDataGenerator(rotation_range=10, width_shift_range=0.1, shear_range=0.1,
 #                              height_shift_range=0.1, zoom_range=0.1)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
#batches = datagen.flow(X_train, Y_train, batch_size=64)
#val_batches = datagen.flow(X_val, Y_val, batch_size=64)


# In[ ]:


from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)


# In[ ]:


batch_size = 86
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 30, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# In[ ]:


model.evaluate(X_val, Y_val)


# In[ ]:


test = test.values.reshape(-1,28,28,1)
train_x = (test.astype('float32') / 255.)


# In[ ]:



# predict results
results = model.predict(test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)


# In[ ]:




