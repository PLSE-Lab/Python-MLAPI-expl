#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# *** 1. READING IN THE FILES**

# In[ ]:


#read the training data file
df = pd.read_csv('../input/train.csv')
print(df.head())


# > Here we have two columns. One colum corresponds to the images for the whales while the other column corresponds to the image id for the whale.

# In[ ]:


#let's look at the unique values in the ID column
print(df['Id'].describe())


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))
sns.countplot(y = df['Id'] == 'new_whale', palette = 'Dark2')


# >From the above plot we can see that the count of ID's corresponding to the pics that were labelled as new_whale is maximum. The images with new whales amount to a total of about 9700 which is more than half of total available images.

# > **Let's prepare our data so that we can train our CNN model onto it. The images here are in the form of string. We know that our CNN model takes images in the form of array as input. So we need to convert our string into array format so that we can feed them to our CNN.**

# In[ ]:


#dimension of our original training dataframe
print(df.shape)


# In[ ]:


#Let's get our x_train and y_train from our dataframe
x_train = df['Image']
y_train = df['Id']


# **2. PREPARING OUR TRAINING IMAGE DATA**

# >**while obtaining our data into an array format our data has to follow a specific format. The usual format for the input data that is to be feed to the CNN model is of the form --> (batch_size, height, width, channels). Here the batch size is nothing but the total number of rows in our train dataframe i.e 25361. The height and width is something that we can arbitrarily choose. It's always a good practice to choose the height and width in such a way as it should be as minimum as possible but not as small that it would become difficult for us to interpret the image itself. So the values should be such that we should be able to tell the content of the image by looking at it. The main idea is to reduce the total number of parameters as far as possible to reduce the computational time.**

# In[ ]:


#import all the necessary libraries from the keras API
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input


# In[ ]:


#define a function to prepare our trianing images
def PrepareTrainImages(dataframe, shape, path):
    
    #obtain the numpy array filled with zeros having the format --> (batch_size, height, width, channels)
    x_train = np.zeros((shape, 100, 100, 3))
    count = 0
    
    for fig in dataframe['Image']:
        
        #load images into images of size 100x100x3
        img = load_img("../input/" + path + "/" + fig, target_size = (100, 100, 3))
        
        #convert images to array
        x = img_to_array(img)
        x = preprocess_input(x)

        x_train[count] = x
        count += 1
    
    return x_train
    


# In[ ]:


x_train = PrepareTrainImages(df, df.shape[0], 'train')


# **3. NORMALIZE THE DATA**

# **Once we got the training data the next step would be to normalize the data so that all the pixel values lie in the same range.**

# In[ ]:


print(x_train.shape) #we got the data in the format that we need for the CNN model


# In[ ]:


#let's normalize the data.
x_train[0] # we can see that the pixel values in the following array have large differene in their values
#so it's always better the obtain all the values in the same range
x_train = x_train.astype('float32') / 255 #data normalized


# **4. DATA VISUALIZATION**

# In[ ]:


#Let's visualize some of our taining images
plt.figure(figsize = (12,8))
plt.subplot(2, 2, 1)
plt.imshow(x_train[0][:,:,0], cmap = 'gray') #the first image
plt.title(df.iloc[0,0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(x_train[100][:,:,0], cmap = 'gray')
plt.title(df.iloc[100,0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(x_train[1000][:,:,0], cmap = 'gray')
plt.title(df.iloc[1000,0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(x_train[4000][:,:,0], cmap = 'gray')
plt.title(df.iloc[4000,0])
plt.xticks([])
plt.yticks([])


# **5. LABLE ENCODING AND ONE HOT ENCODING THE ID COLUMN  VALUES**

# In[ ]:


from keras.utils import np_utils #to obtain the one hot encodings of the id values
from sklearn.preprocessing import LabelEncoder #to obtain the unique integer values for each id values


# In[ ]:


le = LabelEncoder()
y_train = np_utils.to_categorical(le.fit_transform(y_train))


# In[ ]:


print(y_train[:10])
print(y_train.shape)


# **6. BUILDING THE CNN MODEL**

# **Now that we have our training dataset prepared we are ready to build our CNN model. Let's start by building a simple CNN model
# This model will have the following layer arrangements**
# 
# ***(CONV2D -> ACTIVATION -> MAXPOOLING) --- (CONV2D -> ACTIVATION -> MAXPOOLING) --- (FLATTEN)---(DENSE -> ACTIVATION)***

# In[ ]:


#let's start by importing all the necessary libraries for building the CNN model
import keras
from keras.layers import Conv2D
from keras.layers import Activation, BatchNormalization
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam


# >**We will start by building the CNN model without implementing any regularization techniques such as dropout or batch normalization just to analyze how important it is to use some of the regularization techniques almost everytime to avoid overfitting of your model.

# In[ ]:


#start building the model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (x_train.shape[1:]), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =  (2,2)))

model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =  (2,2)))

model.add(Conv2D(128, (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =  (2,2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))


# In[ ]:


#looking at the summary for our model
model.summary()


# In[ ]:


#compile the model
optim = Adam(lr = 0.001) #using the already available learning rate scheduler
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])


# In[ ]:


#fit the model on our dataset
history = model.fit(x_train, y_train, epochs = 30, batch_size = 64)


# In[ ]:


#let's look how our model performed by plotting the accuracy and loss curves
sns.set(style = 'darkgrid')
plt.figure(figsize = (18, 14))
plt.subplot(2, 1, 1)
plt.plot(range(30), history.history['acc'])
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING ACCURACY')
plt.title('TRAINING ACCURACY vs EPOCHS')

plt.figure(figsize = (18, 14))
plt.subplot(2, 1, 1)
plt.plot(range(30), history.history['loss'])
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING LOSS')
plt.title('TRAINING LOSS vs EPOCHS')


# **CNN WITH BATCH NORMALIZATION**

# >Here I haven't used maxpooling for downsampling. Instead I have increased the amount of stride each (5,5) filter will take while moving over the input image. Also Batch normalization is implemented as a regularization technique to avoid overfitting.

# In[ ]:


#start building the model
model1 = Sequential()
model1.add(Conv2D(32, (5,5), strides = (2,2), input_shape = (x_train.shape[1:]), padding = 'same'))
model1.add(Activation('relu'))
model1.add(BatchNormalization())
#model1.add(MaxPooling2D(pool_size =  (2,2)))

model1.add(Conv2D(32, (3,3), strides = (2,2), padding = 'same'))
model1.add(Activation('relu'))
model1.add(BatchNormalization())
#model1.add(MaxPooling2D(pool_size =  (2,2)))

# model1.add(Conv2D(128, (3,3), padding = 'same'))
# model1.add(Activation('relu'))
# model1.add(BatchNormalization())
# model1.add(MaxPooling2D(pool_size =  (2,2)))

model1.add(Flatten())

model1.add(Dense(32))
model1.add(Activation('relu'))
model1.add(BatchNormalization())

model1.add(Dense(y_train.shape[1]))
model1.add(Activation('softmax'))

model1.summary()


# In[ ]:


#compile the model
optim = Adam(lr = 0.001) #using the already available learning rate scheduler
model1.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])


# In[ ]:


#fit the model on our dataset
history1 = model1.fit(x_train, y_train, epochs = 25, batch_size = 64)


# In[ ]:


#let's look how our model performed by plotting the accuracy and loss curves
sns.set(style = 'darkgrid')
plt.figure(figsize = (18, 14))
plt.subplot(2, 1, 1)
plt.plot(range(25), history1.history['acc'])
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING ACCURACY')
plt.title('TRAINING ACCURACY vs EPOCHS')

plt.figure(figsize = (18, 14))
plt.subplot(2, 1, 1)
plt.plot(range(25), history1.history['loss'])
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING LOSS')
plt.title('TRAINING LOSS vs EPOCHS')


# **CNN WITH BATCH NORMALIZATION AND DROPOUT**

# > Here instead of increasing the strides I have used Maxpooling for downsampling. Also dropout and Batch normalization is also been used as regularization techniques.

# In[ ]:


#start building the model
model2 = Sequential()
model2.add(Conv2D(32, (5,5), input_shape = (x_train.shape[1:]), padding = 'same'))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size =  (2,2), strides = (2,2)))
model2.add(Dropout(0.2))

model2.add(Conv2D(32, (3,3), padding = 'same'))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size =  (2,2), strides = (2,2)))
model2.add(Dropout(0.2))

# model1.add(Conv2D(128, (3,3), padding = 'same'))
# model1.add(Activation('relu'))
# model1.add(BatchNormalization())
# model1.add(MaxPooling2D(pool_size =  (2,2)))

model2.add(Flatten())

model2.add(Dense(128))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.5))

model2.add(Dense(y_train.shape[1]))
model2.add(Activation('softmax'))

model2.summary()


# In[ ]:


#compile the model
optim = Adam(lr = 0.001) #using the already available learning rate scheduler
model2.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])


# In[ ]:


#fit the model on our dataset
history2 = model2.fit(x_train, y_train, epochs = 100, batch_size = 64)


# In[ ]:


#let's look how our model performed by plotting the accuracy and loss curves
sns.set(style = 'darkgrid')
plt.figure(figsize = (18, 14))
plt.subplot(2, 1, 1)
plt.plot(range(100), history2.history['acc'])
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING ACCURACY')
plt.title('TRAINING ACCURACY vs EPOCHS')

plt.figure(figsize = (18, 14))
plt.subplot(2, 1, 1)
plt.plot(range(100), history2.history['loss'])
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING LOSS')
plt.title('TRAINING LOSS vs EPOCHS')


# > If we try to analyze the above three different CNN models, we can clearly see some difference between the model using a regularization technique and the one which isn't. The first model in which no regularization technique was implemented resulted in a good accuracy but to reach that accuracy we required very little time. This is not always bad but also not ideal. The model must take it's time to learn all the features from our dataset and as the epochs progresses the model starts to learn more and more about the dataset and the loss starts to gradually decrease. In the first model the loss did decrease but the decrease was very steep. While in the third model where we implemented both dropout as well as batch normalization the decrease in loss was gradual which is what it should be. While for the model one reached good accuracy in less number of epochs it took more number of epochs for model three to match the performance of the model one and two.
# 
# >We can train the model three for even more number of epochs to achieve the accuracy close to 99% on the training dataset.

# **USING MOBILENET ARCHITECTURE WITH IMAGE DATA GENERATOR**

# In[ ]:


import keras
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


datagen = ImageDataGenerator(rescale = 1 / 255.,
                            horizontal_flip = True,
                            rotation_range = 10,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1)


# In[ ]:


mobilenet_model = MobileNet(weights = None, input_shape = (100, 100, 3), classes = 5005)
mobilenet_model.summary()


# In[ ]:


#compile the model
optim = Adam(lr = 0.001)
mobilenet_model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])


# In[ ]:


#fit the model on our data
h2 = mobilenet_model.fit_generator(datagen.flow(x_train, y_train, batch_size = 64), epochs = 300, steps_per_epoch = len(x_train) // 64)


# In[ ]:


#let's look how our model performed by plotting the accuracy and loss curves
sns.set(style = 'darkgrid')
plt.figure(figsize = (18, 14))
plt.subplot(2, 1, 1)
plt.plot(range(300), h2.history['acc'])
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING ACCURACY')
plt.title('TRAINING ACCURACY vs EPOCHS')

plt.figure(figsize = (18, 14))
plt.subplot(2, 1, 1)
plt.plot(range(300), h2.history['loss'])
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING LOSS')
plt.title('TRAINING LOSS vs EPOCHS')


# From the above plot we can observe that after about 90 epochs the loss became stagnant. 

#  **MAKING PREDICTIONS ON TEST DATA**

# In[ ]:


test_data = os.listdir("../input/test/")
print(len(test_data))


# In[ ]:


test_data = pd.DataFrame(test_data, columns = ['Image'])
test_data['Id'] = ''


# In[ ]:


x_test = PrepareTrainImages(test_data, test_data.shape[0], "test")
x_test = x_test.astype('float32') / 255


# In[ ]:


predictions = mobilenet_model.predict(np.array(x_test), verbose = 1)


# In[ ]:


for i, pred in enumerate(predictions):
    test_data.loc[i, 'Id'] = ' '.join(le.inverse_transform(pred.argsort()[-5:][::-1]))


# In[ ]:


test_data.to_csv('model_submission4.csv', index = False)


# In[ ]:




