# %% [code]
"""
@script_author: Jyothi A. Tom
@script_name: Prediction of iris flower species 
@script_description: Predicting the species of iris flower based on the characteristics given, using the KNN algorithm
@script_packages_used: numpy, pandas, sklearn
"""

#importing the necessary packages
import numpy as np                                  #linear algebra
import pandas as pd                                 #data processing
from sklearn import metrics                         #used to compute accuracy of the model
import sklearn.model_selection as ms                #used to split the data into train and test data
from sklearn.neighbors import KNeighborsClassifier  #used to build knn model

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

iris = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")   #inputs the iris dataset
iris.shape                      #prints the number of rows and columns of the dataset 'iris'
iris.dtypes                     #gives the data type present in each column of 'iris'
y=iris['species'].to_list()     #assigns the 'species' attribute as the target variable
x = iris.drop("species",axis=1) #x contains all the predictors of y.hence the 'id' and 'species' columns are dropped from iris

x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.3,random_state=1) #the dataset is split into train and test data

#build the knn model for k=7
knn1 = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
knn1.fit(x_train,y_train)
y_pred = knn1.predict(x_test)
print("The accuracy of the model: ",metrics.accuracy_score(y_test,y_pred))

#To predict the species of 2 unknown iris flowers
unknown = [[76,34,5,2.4],       
           [5,3.9,2,0.3]] #assigning the characteristics of 2 iris flowers of unknown species to unknown
pred = knn1.predict(unknown)  #predicting the species of the unknown iris flowers
print("Predicted species: ", pred)   #print the predicted species

# %% [code]
