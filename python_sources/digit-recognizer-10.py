#!/usr/bin/env python
# coding: utf-8

# Digit recognition system is the working of a machine to train itself or recognizing the digits from different sources like emails, bank cheque, papers, images, etc. and in different real-world scenarios for online handwriting recognition on computer tablets or system, recognize number plates of vehicles, processing bank cheque amounts, numeric entries in forms filled up by hand (say-tax forms) and so on. The handwritten digits are not always of same size, width, orientation and justified to margins as they differ from writing of person to person, so the general problem would be while classifying the digits due to the similarity between digits such as 1 and 7,5 and 6,3 and 8,2 and 5,2 and 7,etc. This problem is faced more when many people write a single digit with a variety of different handwritings. Lastly, the uniqueness and variety in the handwriting of different individuals also influence the formation and appearance of the digits.
# 
# This dataset includes handwriting digits total of 60,000 images consisting of 49,000 examples in training set with labelled images from 10 digits (0 to 9) and 21,000 examples in testing set which are unlabelled images.
# 
# Handwritten digits are images in the form of 28*28 gray scale intensities of images representing an image. The size of an image is 28 by 28, so there are 784 (28*28) values for the label. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 to 255 , inclusive. Our goal is to correctly identify digits from a dataset of tons of thousands of handwritten images.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')


# In[ ]:


#loading the dataset
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[ ]:


#viewing the image of the digit at index 35
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
image_index=35
print(y_train[image_index])
plt.imshow(x_train[image_index],cmap='Greys')
plt.show()


# In[ ]:


#checking the shape of the training data
print(x_train.shape)
print(x_test.shape)


# In[ ]:


#checking the type of the data
type(x_train)
x_train.dtype


# The data type is unsigned 8-bit integer.

# In[ ]:


#checking for null values in training data
np.isnan(x_train).any()


# In[ ]:


#checking for null values in test data
np.isnan(x_test).any()


# In[ ]:


#checking for null values in training labels data
np.isnan(y_train).any()


# In[ ]:


#checking for null values in test labels data
np.isnan(y_test).any()


# Hence,there are no missing values in the whole dataset

# In[ ]:


#reshaping 3-D array to 2-D array
x_train1=x_train.reshape(60000,784)
x_train1.shape


# In[ ]:


#converting the numpy array into pandas Datframe for further analysis
df=pd.DataFrame.from_records(x_train1)
df.shape


# In[ ]:


df.head()


# In[ ]:


df.describe()


# **Conclusions from the description**
# 1. Each figure is a 28x28 sized image,so there are 784 columns,each depicting the intensity value of a pixel.
# 2. Many pixels have 0 intensity value.
# 3. Some pixels have values other than 0 as in the last few columns,the mean,std and max are not 0.
# 4. Most of the pixels in each column have 0 value as 25%,75% and the median for all the columns are 0.
# 

# In[ ]:


#checking the index of a pixel with intensity 254 
df[df.iloc[:,774]==254].index


# In[ ]:


#checking the shape of the training labels data 
y_train.shape


# In[ ]:


#converting the training labels data into a dataframe
df_y=pd.DataFrame.from_records(y_train.reshape(60000,1))
df_y[0].head()


# In[ ]:


#displaying the count of each value (0,1,2,3,4,5,6,7,8,9) in the labels
df_y[0].value_counts()


# In[ ]:


#plotting a bar plot to show the frequency of each digit in the training data
df_y[0].value_counts().plot(kind='bar')
plt.show()


# All the digits occur with almost the same frequency in the dataset.

# In[ ]:


x_test1=x_test.reshape(10000,784)
df_2=pd.DataFrame.from_records(x_test1)
df_2.head()


# In[ ]:


#Preprocessing 
#Reshaping the images to a single color channel
x_train = df.values.reshape((-1, 28, 28, 1))
x_test = df_2.values.reshape((-1, 28, 28, 1))


# In[ ]:


#Checking the shape after re-shaping
print(x_train.shape)
print(x_test.shape)


# In[ ]:


#One-hot encoding on target
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes = 10)
y_train.shape
y_test = to_categorical(y_test, num_classes = 10)
y_test.shape


# In[ ]:


#preparing pixel data
#converting integers into float
train_norm_x = x_train.astype('float32')
test_norm_x = x_test.astype('float32')


# In[ ]:


#normalizing by dividing by the highest value i.e. 255
train_norm_x=train_norm_x/255.0
test_norm_x=test_norm_x/255.0


# In[ ]:


#splitting into training and validation data
X_train, X_val, Y_train, Y_val = train_test_split(train_norm_x, y_train, test_size = 0.1, random_state=2)


# In[ ]:


#making a cnn model
## my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

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


epochs = 30
batch_size = 86


# In[ ]:


#compiling the model using adam optimizer
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


#Data Augmentation 
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


# In[ ]:


# Annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# In[ ]:


# imports

from keras.models import model_from_json 

# opening and store file in a variable

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

