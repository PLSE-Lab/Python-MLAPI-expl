#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Here are some standard libraries that are loaded when you 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualize satellite images
from skimage.io import imshow # visualize satellite images

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout # components of network
from keras.models import Sequential # type of model


# ## Get Input Data
# The input data was encoded into CSV files. The X_test_sat4.csv flattened the images that were 28 x 28 x 4 that were taken from space. The first three channels are the standard red, green, and blue channels in normal images. The 4th is a near-infrared band. We are using the smaller test set because the training set is too big.
# After extracting the data from the csv files, we can reshape it into the original images. Then, we can see the images before we train on them.
# The second file we are loading are the labels for each image. They can be one of 4: barren land, trees, grassland and other. Each row in the file looks like this [1,0,0,0], where only one of the 4 value is 1. If it is one, then it is that class respective to the order I showed above. If it was the above values, the image is a picture of barren land. If it was [0,1,0,0], then it would be trees. If it was [0,0,1,0], then it would be grassland and so on.

# In[ ]:


x_train_set_fpath = '../input/X_test_sat4.csv'
y_train_set_fpath = '../input/y_test_sat4.csv'
print ('Loading Training Data')
X_train = pd.read_csv(x_train_set_fpath)
print ('Loaded 28 x 28 x 4 images')
Y_train = pd.read_csv(y_train_set_fpath)
print ('Loaded labels')


# ## The values are in a pandas(data library) DataFrame. We need them as a numpy array
# You can convert pandas dataframes to numpy arrays like this:

# In[ ]:


X_train = X_train.as_matrix()
Y_train = Y_train.as_matrix()
print ('We have',X_train.shape[0],'examples and each example is a list of',X_train.shape[1],'numbers with',Y_train.shape[1],'possible classifications.')


# In[ ]:


#First we have to reshape each of them from a list of numbers to a 28*28*4 image.
X_train_img = X_train.reshape([99999,28,28,4]).astype(float)
print (X_train_img.shape)


# In[ ]:


#Let's take a look at one image. Keep in mind the channels are R,G,B, and I(Infrared)
ix = 54643#Type a number between 0 and 99,999 inclusive
imshow(np.squeeze(X_train_img[ix,:,:,0:3]).astype(float)) #Only seeing the RGB channels
plt.show()
#Tells what the image is
if Y_train[ix,0] == 1:
    print ('Barren Land')
elif Y_train[ix,1] == 1:
    print ('Trees')
elif Y_train[ix,2] == 1:
    print ('Grassland')
else:
    print ('Other')


# ## Let's now define our model
# There are 2 different types of models we can choose from: A 'vanilla' artificial neural network we have been learning about, and a special Convolutional Neural Network we will learn about, which is very, very good at image recognition. For now we will use the simpler, vanilla artificial neural network. The network will only have one layer: the output one. This network will not be expected to be very powerful, and pretty slow.

# In[ ]:


model = Sequential([
    Dense(4, input_shape=(3136,), activation='softmax')
])


# Now that we have the data and model ready, there is one more thing we have to do. In neural networks, it is very important we normalize training data. This means we make the mean 0, and the standard deviation 1 for the best results. However, dividing the image by 255 is good enough. We will just divide the array by 255:

# In[ ]:


X_train = X_train/255


# ## Now lets fit our model to the training data

# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train,Y_train,batch_size=32, epochs=5, verbose=1, validation_split=0.01)


# Lets try to see what the model can do on a few images. Let's first get the predictions:

# In[ ]:


preds = model.predict(X_train_img[-1000:].reshape(1000, 28*28*4), verbose=1)


# In[ ]:


ix = 20 #Type a number between 0 and 999 inclusive
imshow(np.squeeze(X_train_img[99999-(1000-ix),:,:,0:3]).astype(float)*255) #Only seeing the RGB channels
plt.show()
#Tells what the image is
print ('Prediction:\n{:.1f}% probability barren land,\n{:.1f}% probability trees,\n{:.1f}% probability grassland,\n{:.1f}% probability other\n'.format(preds[ix,0]*100,preds[ix,1]*100,preds[ix,2]*100,preds[ix,3]*100))

print ('Ground Truth: ',end='')
if Y_train[99999-(1000-ix),0] == 1:
    print ('Barren Land')
elif Y_train[99999-(1000-ix),1] == 1:
    print ('Trees')
elif Y_train[99999-(1000-ix),2] == 1:
    print ('Grassland')
else:
    print ('Other')


# Ehh. 72% accuracy is pretty bad. Here's a model that can do lot better(there is some error):

# In[ ]:


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,4)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train_img,Y_train,batch_size=32, epochs=5, verbose=1, validation_split=0.01)


# That should take a good 20-25 minutes to train. There are definitely going to be better, faster architectures you could use, but this network is much better at image recognition, then the other type. OR it should be, but it is not, and i am working on it=)

# In[ ]:




