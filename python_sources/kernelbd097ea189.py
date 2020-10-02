
#Installing Theano Tensorflow Keras

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


#Importing the dataset
dataset=pd.read_csv('../input/Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values


#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
#splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#creating and Fitting classifiers to the Training set
#part-2 Making ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#Adding the third output layer & last output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the Ann to the Training set
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)


#part-3 Making the prediction
#predicting the test set result
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)
"""
Predict if the single customer with following infomation is Fraud or not
Geography: France
Credit Score:600
Gender:Male
Age:40
Tenure:3
Balance:60000
Number of Products:2
Has Credit Card:Yes
Is active Member:Yes
Estimated Salary:50000
"""
new_prediction=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,6000,2,1,1,50000,]])))

new_prediction=(new_prediction>0.5)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

print(cm)
print(new_prediction)


































































