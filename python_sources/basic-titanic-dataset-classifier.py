# Custom Artificial Neural Network

# Data Pre-processing

# Importing libraries
import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, [2, 4, 5, 6, 7]].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_gender = LabelEncoder()
X[:, 1] = labelencoder_X_gender.fit_transform(X[:, 1])
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean')
X = imp.fit_transform(X)

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating the ANN

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU

# Initialising ANN
classifier = Sequential()

# Adding input layer and hidden layer one
classifier.add(Dense(5, input_shape = (5, )))
classifier.add(Dropout(rate = 0.1))
classifier.add(LeakyReLU())

# Adding hidden layer two
classifier.add(Dense(3))
classifier.add(Dropout(rate = 0.1))
classifier.add(LeakyReLU())

# Adding output layer
classifier.add(Dense(1, activation = 'sigmoid'))

# Compiling ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN to training set
classifier.fit(X_train, y_train, batch_size = 2, epochs = 500)

'''
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''

# Testing
tests = pd.read_csv('../input/test.csv')
input_X = tests.iloc[:, [1, 3, 4, 5, 6]].values
input_X[:, 1] = labelencoder_X_gender.fit_transform(input_X[:, 1])
input_X = imp.fit_transform(input_X)
input_X = sc.transform(input_X)
output_y = classifier.predict(input_X)
output_y = output_y[:, 0]
results = []
for pred in output_y:
    if pred > 0.5:
        results.append(1)
    else:
        results.append(0)
tests = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
    'PassengerId':tests['PassengerId'],
    'Survived':results
    })
submission.to_csv('submissionNQTitanic.csv', index = False)