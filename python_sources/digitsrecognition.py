# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading and exploring the MNIST data set
train_data_path = '../input/train.csv'
test_data_path = '../input/test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

#print(train_data.head()) # just to make sure I know the structure of the data

X_train = train_data.values[:,1:]
y_train = train_data.values[:,0]

img_rows = 28
img_cols = 28
num_classes = 10

#print(train_data.values[:,1:].shape)
num_images = X_train.shape[0]
x_shaped_array = X_train.reshape(num_images, img_rows, img_cols, 1)
out_x = x_shaped_array / 255
#print(out_x.shape)

# Defining the model
my_model = Sequential()
my_model.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1),data_format="channels_last"))
my_model.add(Conv2D(32, kernel_size=2, activation='relu', data_format="channels_last"))
my_model.add(Conv2D(32, kernel_size=2, activation='relu', data_format="channels_last"))
my_model.add(Conv2D(16, kernel_size=2, activation='relu', data_format="channels_last"))
my_model.add(Conv2D(16, kernel_size=2, activation='relu', data_format="channels_last"))
my_model.add(Flatten())
my_model.add(Dense(128, activation='relu'))
my_model.add(Dense(num_classes, activation='softmax'))

my_model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
my_model.fit(out_x, y_train,
          batch_size=128,
          epochs=16,
          validation_split = 0.1)
          
X_test = test_data.values

          
num_images_test = X_test.shape[0]
x_shaped_array_test = X_test.reshape(num_images_test, img_rows, img_cols, 1)
out_x_test = x_shaped_array_test / 255


predictions = my_model.predict(out_x_test)
predicted_labels = np.argmax(predictions, axis=1)
d = {'ImageId': np.arange(1, 28001), 'Label': predicted_labels}
final_predictions = pd.DataFrame(data=d, index=np.arange(1, 28001))

predictions_path = 'submission1.csv'
pd.DataFrame(final_predictions).to_csv(predictions_path, index=None)