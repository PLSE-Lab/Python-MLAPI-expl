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
dataset=pd.read_csv('../input/train.csv')
dataset1=pd.read_csv('../input/test.csv')
x=dataset.iloc[:,[2,4,5,6,7,9]].values
x_test=dataset1.iloc[:,[1,3,4,5,6,8]].values
y=dataset.iloc[:,1].values


#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,1]=labelencoder_x.fit_transform(x[:,1])
labelencoder_x_test=LabelEncoder()
x_test[:,1]=labelencoder_x.fit_transform(x_test[:,1])

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
x= my_imputer.fit_transform(x)
x_test= my_imputer.fit_transform(x_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
x_test = sc.fit_transform(x_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 6))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x, y, batch_size = 10, nb_epoch = 500)

# PREDICTING  THE TEST RESULTS 
y_pred = classifier.predict(x_test)


y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0








