# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

x_train = images
y3_train = train["Distance"]
y3_train = np.array(y3_train)
input_shape = images[0].shape

#Modelling a Sequential Model
classifier = Sequential()

classifier.add(Conv2D(16, (5, 5),input_shape = input_shape, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 1 - Convolution
classifier.add(Conv2D(16, (3, 3),input_shape = input_shape,activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3),input_shape = input_shape, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), input_shape = input_shape,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a fourth convolutional layer
classifier.add(Conv2D(64, (3, 3), input_shape = input_shape,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'linear'))

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
classifier.fit(x_train, y3_train, epochs = 50, validation_split=0.27)

#Predicting for validation set
predictions = classifier.predict(images_test)
print(predictions)