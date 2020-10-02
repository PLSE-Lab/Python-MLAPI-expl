#!/usr/bin/env python
# coding: utf-8

# # ****Digit Recognizer****

# If you are interested to know how to make ML WebApps which you can delpoy, go through this medium article. [Sentiment analysis with Bi-LSTM and delpoy as Web App](https://medium.com/@mirzamujtaba10/sentiment-analysis-642b935ab6f9)

# In this notebook, I am going to demonatrate how to make model that can recognize digits ranging from 0 to 9 .The dataset available has just pixel values,that too in a csv file.So our first step is to store this data in a pandas dataframe and then using numpy , we ll trasform them into 28 x 28 matrix.We can feed this matrix to a convolutional network.

# We use convolutional neural networks in order recognize digits from 0-9.Convolutional neural networks are very much suitable for recognizing images.Although when dealing with numbers such as 6 and 9 which have just different orientation,this type of network may struggle.

# Let us first understand how convolutional neural networks are different from normal neural nets.Convolution neural networks consists of two layers viz- convolution layer and pooling layer.To know what these layers actually do, I suggest you go through this blog.This is also an image recognition task, but here we classify images of animals [Recognizing HD images of animals](https://www.mygreatlearning.com/blog/image-recognition/)

# If you did not go though the blog, let me explain briefly. In Convolution layer, we use filters of our desired shape such as (2 x 2) or (3 x 3).After applying these filters on the image we get an image which is enhanced in a certain way. e.g. After using a certain type of filter, it may be that horizontal lines in the image will be highlighted more as compared to rest of shapes. So when using multiple filters (e.g x filters), we get x different images and we use all these filtered images to get features to train the model.Also the  using the convolution we also consider the information in the surrounding pixels of images whih is essential when dealing with images.Here is an example that shows four different images oobtained after applying four filters to a single image.

# ![convolution ](https://lh6.googleusercontent.com/qxCnvVWYxtxvyJBYOyC40q4PhYo2MKeu4iXlP4_Cymj55BO2F-Q99lYrLmvZ86KFsg9IwgXe8c-9B6I1aGCqv3WGaUeqg2c7mCyD5FEnSqqREWg2i8i0E6Z82cczGpL2rKGovNS8)

# Using the Pooling layer, we actually reduce the size of our image without loosing much image.Here we also use a filter usually (2 x 2) or (3 x 3).In this model we have used maxpooling, which selects the highest value pixel in filter.This image describes the process much easily 

# ![Pooling operation](https://lh6.googleusercontent.com/4vtME_1uqHyaJ5jcj8VGBJVBgDNnwh0YcIlMlYB0MSE9p2eJ6o9JeaMUA-n__ExzjAip82Sw8s4cqjfLC32LUR6p6acyp68C7SOInvf7pdQw7CogbWQVb6Kgo8fqME2ndEnL7yc3)

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


#importing require libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import ReLU,BatchNormalization
from keras import models
from keras.layers import MaxPool2D
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D


# load the downloaded dataset
dataset_train  =pd.read_csv('../input/digit-recognizer/train.csv')
dataset_test   =pd.read_csv('../input/digit-recognizer/test.csv')
submission     =pd.read_csv('../input/digit-recognizer/sample_submission.csv')

#splitting the training data
dataset_train_X  =np.asarray(dataset_train.iloc[:,1:]).reshape([-1, 28, 28, 1])
dataset_train_Y  =np.asarray(dataset_train.iloc[:,:1]).reshape([len(dataset_train), 1])

#splitting the test data
dataset_test_X  =np.asarray(dataset_test.iloc[:,:]).reshape([len(dataset_test), 28, 28, 1])
# dataset_test_Y  =np.asarray(dataset_test.iloc[:,:]).reshape([len(dataset_test), 1])

#converting pixel value in the range 0 to 1
dataset_train_X  =dataset_train_X/255
dataset_test_X   =dataset_test_X/255

#initilizing model
model = models.Sequential()

# Block 1
model.add(Conv2D(64,3, padding  ="same",kernel_initializer='he_uniform',input_shape=(28,28,1)))
model.add(ReLU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Block 2
model.add(Conv2D(64,3, padding  ="same",kernel_initializer='he_uniform'))
model.add(ReLU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

#Block 3
model.add(Conv2D(64,3, padding  ="same",kernel_initializer='he_uniform'))
model.add(ReLU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

#Now we tranfoem thw 2d matriz into a 1 d vector to feed into a dense layer
model.add(Flatten())

model.add(Dense(576,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(264,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(10,activation="softmax"))

model.compile(Adam(lr=0.001), loss="sparse_categorical_crossentropy" ,metrics=['accuracy'])
model.summary()

#train the model
# history=model.fit(dataset_train_X,dataset_train_Y,epochs=50,batch_size=32)


# In[ ]:


# prediction=model.predict(dataset_test_X)
# label = np.argmax(prediction, axis=1)
# label


# In[ ]:


# submission['Label'] = label


# In[ ]:


# submission.to_csv('submission.csv', index=False)

