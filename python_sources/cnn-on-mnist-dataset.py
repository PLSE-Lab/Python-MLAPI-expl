# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
np.random.seed(2)
from sklearn.model_selection import train_test_split


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

y_train = train["label"]

# Drop 'label' column
x_train = train.drop(labels = ["label"],axis = 1) 


# Normalizing the data as CNN will work better or converge faster on [0..1] rather than [0..255]
x_train = x_train / 255.0
test = test / 255.0

#reshape the data
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

y_train = to_categorical(y_train, num_classes = 10)  #encode labels to one hot vector

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)

#CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

CNN_model = Sequential()

CNN_model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
CNN_model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
CNN_model.add(MaxPool2D(pool_size=(2,2)))
CNN_model.add(Dropout(0.50))


CNN_model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
CNN_model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
CNN_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
CNN_model.add(Dropout(0.50))


CNN_model.add(Flatten())
CNN_model.add(Dense(256, activation = "relu"))
CNN_model.add(Dropout(0.5))
CNN_model.add(Dense(10, activation = "softmax"))

# Defining the optimizer as it will iteratively improves the parameters to minimize the loss
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
CNN_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 2
batch_size = 86

result = CNN_model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val), verbose = 2)