# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Import Pandas
import pandas as pd
import numpy as np

#Read csv dataset for training and validation
dataset=pd.read_csv('../input/train.csv')

#Read dataset for Kaggle submission
dataset_submit=pd.read_csv('../input/test.csv')

#X=independant variables=the image data + basic feature scaling
X=dataset.iloc[:,1:].values / 255.0
X_submit=dataset_submit.iloc[:,0:].values / 255.0

#y=dependant variable= the written digit label 0-9
y=dataset.iloc[:,:1].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

#Grab the automatic data splitter from sklearn
from sklearn.model_selection import train_test_split

#Create the Training/Testing split for my cross validation (No cross validation at present)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.001,random_state=0)


#Import keras to create the sequential network structure
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initialize the network
classifier = Sequential()

#Add the first hidden layer, and specifying #inputs mean(10,784)=397)
classifier.add(Dense(units=382,kernel_initializer='uniform',activation='relu',input_dim=784))
classifier.add(Dropout(0.3))

classifier.add(Dense(units=191,kernel_initializer='uniform',activation='relu',input_dim=784))
classifier.add(Dropout(0.3))

#Add the output layer, an analog digit value
classifier.add(Dense(units=10,kernel_initializer='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 10,verbose=2) #500

#Predicting the Kaggle submission
y_pred_submit=classifier.predict(X_submit)

#Create the Kaggle submission
y_pred_submit_int=[['ImageId','Label']]
for i in range(len(X_submit)):
    summedVal=int(round((y_pred_submit[i][0]*0)+(y_pred_submit[i][1]*1)+(y_pred_submit[i][2]*2)+(y_pred_submit[i][3]*3)+(y_pred_submit[i][4]*4)+(y_pred_submit[i][5]*5)+(y_pred_submit[i][6]*6)+(y_pred_submit[i][7]*7)+(y_pred_submit[i][8]*8)+(y_pred_submit[i][9]*9),0))
    if(summedVal>9):
        summedVal=9
    pair=[i+1,summedVal]
    y_pred_submit_int.append(pair)

#Write submission to storage    
raw_data = y_pred_submit_int
df = pd.DataFrame(raw_data)
df.to_csv(path_or_buf = 'submission01.csv', index=None, header=False)
