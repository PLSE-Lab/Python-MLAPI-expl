import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Importing the dataset

dataset = pd.read_csv("../input/Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y =dataset.iloc[:, 13].values

# Handling the categorical Variables : One hot Ecoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:, 1]=labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 =LabelEncoder()
X[:, 2]= labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features= [1])
X= onehotencoder.fit_transform(X).toarray()
X=X[: , 1:]

#spliting the data set into training set and test set

from sklearn.cross_validation import train_test_split
X_train , X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train =sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)

# Importing the keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim= 6, init="uniform", activation= "relu", input_dim=11))

# Adding the Second hidden layer
classifier.add(Dense(6, init="uniform", activation="relu")) #Home Work- explore the libraries and Functions like Dense()

#Adding the output layer
classifier.add(Dense(1,init="uniform", activation="sigmoid"))

# Compling ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fitting the ANN to the training sets
classifier.fit(X_train,y_train,batch_size= 10, nb_epoch =111  )

#Predicting the test set results
y_pred= classifier.predict(X_test)
y_pred =(y_pred> 0.5)

#Confusion Metrics
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)




