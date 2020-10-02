# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Part1 - Data preprocessing

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('../input/atm_data_m2.csv')
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 10].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_0 = LabelEncoder()
X[:,0] = labelencoder_X_0.fit_transform(X[:, 0])
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:,3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:,4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
X[:,5] = labelencoder_X_5.fit_transform(X[:, 5])
# Encode variable with one hotencoder
onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4,5])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

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


# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu', input_dim = 33))

# Adding the 2 hidden layer
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

# Adding the 3 hidden layer
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))

# Adding the 4 hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mse')

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Print the predicted vs actual results
for i in range(len(y_test)):
    print("Y=%s, Predicted=%s" % (y_test[i], y_pred[i]))


# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix

from sklearn import metrics

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))


print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
(np.sqrt(metrics.mean_squared_error(y_test, y_pred))/
np.mean(y_test))



