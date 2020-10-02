
#importing necessary libraries
import numpy as np                                                                              
import pandas as pd
import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler               

dataset = pd.read_csv('Students Performance in Exams.csv')    #reading the dataset
X = dataset.iloc[:,0:5].values  #reading the set based on which predictions are made
y = dataset.iloc[:,5:9].values  #reading the results 

#creating LabelEncoders for categorical values
labelEncoderX1 = LabelEncoder() 
labelEncoderX2 = LabelEncoder() 
labelEncoderX3 = LabelEncoder()
labelEncoderX4 = LabelEncoder()
labelEncoderX5 = LabelEncoder()

#changing categorical values to numbers
X[:,0] = labelEncoderX1.fit_transform(X[:,0])
X[:,1] = labelEncoderX2.fit_transform(X[:,1])
X[:,2] = labelEncoderX3.fit_transform(X[:,2])
X[:,3] = labelEncoderX4.fit_transform(X[:,3])
X[:,4] = labelEncoderX5.fit_transform(X[:,4])

#using OneHotEncoder to change every single value to binary values by creating new columns
oneHotEncoder = OneHotEncoder()
X = oneHotEncoder.fit_transform(X).toarray()

#getting rid of dummy variables 
X = X[:, np.r_[1:6,7:12,13:14,15:16]]

#applying standart scaler to inputs and modifying results to be in the range between 0 and 1 
sc = StandardScaler()
X = sc.fit_transform(X)
y = y / 100

#splitting the dataset to subsets 
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#creating machine learning model using keras
model = Sequential()
model.add(Dense(units = 32, activation = 'sigmoid'))
model.add(Dropout(0.1)) #adding dropout to prevent model from overfitting
model.add(Dense(units = 16, activation = 'sigmoid'))
model.add(Dropout(0.1)) #adding dropout to prevent model from overfitting
model.add(Dense(units = 3, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 32, epochs = 1500)

#predicting results for new values
prediction = model.predict(X_test)
result = y_test - prediction

#this model reaches a mean square error of around 0.0192 which is around 13% eror per each