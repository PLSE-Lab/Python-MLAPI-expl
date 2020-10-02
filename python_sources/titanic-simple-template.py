# ML Data preprocessing template 
# importing libarary 


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 



#Importing database 
dataset= pd.read_csv('train.csv')
#As u know i have imported the Data set

X=dataset.drop(['Name','Ticket','Cabin','Embarked','Survived'],axis=1).values
#Now here is a catch i have eliminated all the unwanted coulmns fron the data and made our 
#Dependent varible dataset

Y=dataset.iloc[:,1].values
# Here i created the vector my dependent varible note : indecates the coulmn here 


#Handling the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
#Now above code is for trasforming the male female into 1 and 0 , its called categorical Data handling 
#dont worry abt oneHotEncoder u will understand it after wards 


# Handling the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, [3]])
X[:,[3]] = imputer.transform(X[:,[3]])

#As u can see i have replaced some missing ages with most frequent ages  

#Data spliting 
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#Here i have split my dataset in the traing and test set in 20 - 80 ratio that means i 
# wiil feed 80 % data to our machine to learn and we will use 20% data to test our model 


# Implementing the random forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion = 'entropy',random_state=0)
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



























