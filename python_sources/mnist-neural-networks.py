# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing.imputation import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import pandas as pd 
import numpy as np 
import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_data =pd.read_csv("../input/train.csv")


# In[3]:


test =pd.read_csv("../input/test.csv")


# In[4]:


submission = pd.read_csv("../input/sample_submission.csv")


# In[5]:


Y = train_data['label']


# In[6]:


train_data = train_data.drop(['label'], axis=1)


# In[7]:


X = train_data


# In[8]:

x_train, x_val, y_train, y_val = train_test_split(X.as_matrix(), Y.as_matrix(), test_size=0.10, random_state=42)

# network parameters 
batch_size = 128
num_classes = 10
epochs = 5 # Further Fine Tuning can be done

# input image dimensions
img_rows, img_cols = 28, 28

# preprocess the train data 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_train /= 255

# preprocess the validation data
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
x_val = x_val.astype('float32')
x_val /= 255

input_shape = (img_rows, img_cols, 1)

# convert the target variable 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# preprocess the test data
Xtest = test.as_matrix()
Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols, 1)

# Model Creation:

model = Sequential()

# add first convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# add second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# add one max pooling layer 
model.add(MaxPooling2D(pool_size=(2, 2)))

# add one dropout layer
model.add(Dropout(0.25))

# add flatten layer
model.add(Flatten())

# add dense layer
model.add(Dense(128, activation='relu'))

# add another dropout layer
model.add(Dropout(0.5))

# add dense layer
model.add(Dense(num_classes, activation='softmax'))

# complile the model and view its architecur
model.compile(loss=keras.losses.categorical_crossentropy,  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.summary()

#Fitting in model and calculate accuracy

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
accuracy = model.evaluate(x_val, y_val, verbose=0)
print('Test accuracy:', accuracy[1])


# preprocess the test data
Xtest = test.as_matrix()
Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols, 1)

pred = model.predict(Xtest)
y_classes = pred.argmax(axis=-1)
res = pd.DataFrame()
res['ImageId'] = list(range(1,28001))
res['Label'] = y_classes
res.to_csv("output.csv", index = False)