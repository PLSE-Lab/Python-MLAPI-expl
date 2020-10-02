#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **This is  a redo of the boelow code which I did some weeks ago. I have gained some new knowlegde since then. I will be using a CNN in this Kernel to predict the hand sign in an image. The hand signs are of course of the ASL.**
# 
# First thing will be to load the data and convert it into pandas dataframes.

# In[ ]:


file_train = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")
file_test = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test.csv")

train_dataframe = pd.DataFrame(file_train)
test_dataframe = pd.DataFrame(file_test)


# **Next, I will do a just have a look at the data
# 1)Print the head for both the train and test dataframe
# 2)Do some minor visializations. This seems like a cleaned dataset so I will prin the info just to clarify 
# 3)**

# In[ ]:


train_dataframe.head(10)


# In[ ]:


test_dataframe.head(10)


# **Asyou can see, our label is the first column. Our data is a 28*28 pixels hence the 784 columns
# I suspect there are no null values but just to make sure I will print the data info**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

train_dataframe.info()


# In[ ]:


test_dataframe.info()


# As you can see. It doesn't look like there are any inconsistances in the data. There are 785 columns in both dataframes and they are all int64 types. Now will move on to preprocessing

# In[ ]:


Y_train = train_dataframe.label
X_train = train_dataframe.drop(['label'],axis=1)

Y_test = test_dataframe.label
X_test = test_dataframe.drop(['label'],axis=1)
X_train.head()


# In[ ]:


Y_train.head()


# In[ ]:


from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


Y_train_final = to_categorical
Y_test_final = to_categorical


X_train = X_train.values


# In[ ]:


X_train_final = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.values
X_test_final = X_test.reshape(X_test.shape[0],28,28,1)


# **Our data is already split betweet train and test so no need to split, will just go on and build our model. We will use a CNN **

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense, Dropout, Flatten

model = Sequential



# We are working with 28 by 28 images 

# In[ ]:


file = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")
dataframe = pd.DataFrame(file)
X = dataframe.drop(['label'],axis=1)
Y = dataframe.label
print(X.columns[0])
print("======")
print(Y.unique())


# In[ ]:


#"""
import cv2 as cv

imag = cv.imread("/kaggle/input/sign-language-mnist/amer_sign2.png")
#When reading pixel values from a csv, dont forget to reshape
img = np.array(X.iloc[6])
img = img.reshape((28,28))
print(np.array(X.iloc[6]))
#plt.imshow(img)


#"""


# In[ ]:




x_train = X.values

x_train = x_train.reshape(X.shape[0],28,28,1 )
x_train = x_train/255
y_train = to_categorical(Y)



# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(32, (3,3), padding = 'same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (3,3), padding = 'same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3,3), padding = 'same', activation='relu'))
model.add(Conv2D(64, (3,3), padding = 'same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(25,activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


hist = model.fit(x_train,y_train, batch_size=15, epochs=5, validation_split=0.2)

