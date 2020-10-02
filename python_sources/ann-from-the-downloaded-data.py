# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/Interview.csv')
dataset.pop('Candidate_ID')
dataset = dataset.replace('Scheduled Walkin', 'Scheduled Walk In')
dataset = dataset.drop(dataset.index[1233])
X = dataset.iloc[:, 0:21]
X.pop('Observed_Attendance')
X = pd.get_dummies(X , columns=X , drop_first=False).values
y = dataset.loc[:, 'Observed_Attendance'].map({'Yes': 1, 'No' : 0, 'No ': 0}).values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle=True)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 256))

# Adding the second hidden layer
classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the forth hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Compiling the ANN
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, epochs=500, batch_size=25)

# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results

loss, accuracy = classifier.evaluate(X_test, y_test)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)