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


# **Data Preparation**
# 1. Load Data

# In[ ]:


#Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


Y_train = train["label"]
#Drop 'label' column
X_train = train.drop(labels = ["label"],axis=1)
del train
g = sns.countplot(Y_train)
Y_train.value_counts()


# **2. Check for null and missing values**

# In[ ]:


X_train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# **3. Normalization**
# 
# Converting data from [0...255] value to [0...1]

# In[ ]:


X_train = X_train/255.0
test = test/255.0


# **4. Reshape**
# Reshaping images from 784px to [28,28,1] 3D Matrices
# 
# Here, if RGB images would have been present then we would have reshaped into [28,28,3] 3D matrices. 
# 
# Extra dimenion is channel which is used in Keras.

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# **5. Label Encoding **

# In[ ]:


Y_train = to_categorical(Y_train, num_classes = 10)


# **6. Split training and Validation set**

# In[ ]:


random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


# Model will be trained on 90% of the data and randomly chosen 10% of the data has been kept aside for testing the model.

# In[ ]:


g = plt.imshow(X_train[0][:,:,0])


# **CNN**
# 1.  Convoultional Layer : Here the model learns features from the images, we have to set filters 32 or 64 depending upon the usecase.
# 2. Pooling : MaxPool 2D is the next layer which picks the maximal value in a region.
# 3. Dropout : Randomly shutting off some nodes of the network to prevent repetition (here regularization) and thus reduces overfiting (mugging up of results xD).
# 4. Activation Function : Here we have used "relu" [max(0, x)]. It is added to provide non-linearity to the network.
# 5. Flatten Layer : It is used to convert final feature map into a one single vector.
# 
# 

# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same',
                activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same',
         activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',
                activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',
                activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

