# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
dataset = pd.read_csv('../input/oasis_cross-sectional.csv')
dataset = dataset.fillna(method='ffill')
X = dataset.iloc[:,4:11].values
y = dataset.iloc[:, 3].values

# Making Age binary using assumption people over 65 having Alzheimers
ys = []
for x in y:
   if x >= 65:
     
       x = 1
   else: 
       x = 0
   ys.append(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, ys, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense 

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation = 'relu',  input_dim = 7, units = 12, kernel_initializer = 'uniform'))

# Adding the second hidden layer
classifier.add(Dense( activation = 'relu', units = 12, kernel_initializer = 'uniform'))

# Adding the third hidden layer
classifier.add(Dense( activation = 'relu', units = 12, kernel_initializer = 'uniform'))

# Adding the output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform' ))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   
print(cm)

# Any results you write to the current directory are saved as output.