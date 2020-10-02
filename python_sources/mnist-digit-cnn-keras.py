#!/usr/bin/env python
# coding: utf-8

# This kernel is a quick dive into the MNIST data set and shows some simple usage of Keras to first train a fully connected network followed by a more appropriate CNN. It's still work in progress and it's mainly to get myself up to speed with the Kaggle Kernel, but it may be of use to people looking for a quick tutorial (There are many much more detailed tutorials out there)
# 
# Any feedback is more than welcome :)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop


import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#     First we read the input sets from cvs files

# In[ ]:


train = pd.read_csv( '../input/train.csv')
test = pd.read_csv( '../input/test.csv' )


# As keras expects our data in the form of numpy arrays, we first convert the data and split it into the inputs X and the output labels Y

# In[ ]:


train_matrix = train.as_matrix()
x_train = train_matrix[:,1:]
y_train = train_matrix[:,0]

x_train = x_train.reshape( ( 42000, 28, 28, 1 ) ).astype( 'float32' ) / 256.0
x_val = x_train[ 40000:, :, :, :]
x_train = x_train[ :40000, :, :, :]
x_test = test.as_matrix().reshape( 28000, 28,28,1).astype( 'float32' ) / 256.0

y_train = to_categorical( y_train, 10 )
y_val = y_train[40000:]
y_train = y_train[:40000]
x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape


# To check our input data we take the first 10 images and render them as in image using plt.imshow. We need to do a bit of rearrangment on the dimensions in order to have them shown in a horizontal row

# In[ ]:


image_count = 10
images = [ x.reshape( 28,28 ) for x in np.split(x_train[:image_count,:,:,0], image_count ) ]
plt.imshow( np.hstack( images ) )


# For our first attempt we use a fully connected neural net with a single hidden layer of size 512. We have the following three layers:
# * Flatten: Since we have prepared our data to be of size 28x28x1 to serve as an input to a CNN, we need to flatten the data
# * Dense: This is our hidden layer. It contains 512 hidden units with a relu activation (max(0,x))
# * Dense: Our output layer. We have 10 units as there are 10 outputclasses. We use the standard softmax activation for multi-class classification
# 
# Once we've constructed the model we compile it so that we can fit it to the data. We use categorical_crossentropy which is the standard loss function for multi classification (with the classes encoded as one-hot vectors). RMSprop will give us a good speed on our fitting and we output accuracy as an additional metric as it is more informative than the straight loss

# In[ ]:


simple = Sequential()
simple.add( Flatten( input_shape=(28,28,1)))
simple.add( Dense( 512, activation='relu' ))
simple.add( Dense( 10, activation='softmax' ))

simple.compile( loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
simple.summary()


# Now we are ready to fit the data using our prepared training data and validation data. We limit our optimisation to 5 epochs so it completes in a reasonable time

# In[ ]:


simple.fit( x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val,y_val))


# In[ ]:


cnn = Sequential()

cnn.add( Conv2D( 16, 3, input_shape =( 28,28, 1)))
cnn.add( MaxPooling2D())
cnn.add( Conv2D( 32, 3, ))
cnn.add( Flatten())
cnn.add( Dense( 128, activation='relu' ))
cnn.add( Dense( 128, activation='relu' ))
cnn.add( Dense( 10, activation='softmax' ))

cnn.compile( loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
cnn.summary()


# In[ ]:


cnn.fit( x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val,y_val))
                    


# In[ ]:


results = np.argmax( cnn.predict( x_test ), axis=1)


# In[ ]:


output = pd.DataFrame( {'Label': results, 'ImageId' : range( 1, len( results ) + 1) } )
output


# In[ ]:


plt.hist( output)


# In[ ]:


output.to_csv( 'results.csv', index=False)


# In[ ]:




