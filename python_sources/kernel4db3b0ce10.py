#@script-Author: Salman
#@script-Date  : 13/1/2020
#@script-Description : Using KNN on the Iris dataset to predict the species.
#@external-packages-used: Sklearn

#import all necessary librabries
import pandas as pd
import numpy as np
import sklearn as sk
import warnings 
warnings.filterwarnings('ignore')

#load the data from the dataset
a=pd.read_csv("../input/iris/Iris.csv")
a
X=a.drop(['Species','Id'],axis=1)
X
Species=a['Species']
Species

#convert the categorical data under "Species" column into libraries
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
species_encoded=le.fit_transform(Species)
a['Species']=species_encoded
y=a.drop(['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],axis=1)
y

#Splitting the data into training and tests(60:40)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=5)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Find the best k value
k_range=range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
classes={0:'setosa',1:'versicolor',2:'virginica'}
classes[0]

#Fit and predict
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X,y)
x_new=[[3,4,5,2],[1,2,3,1]]
y_predict=knn.predict(x_new)
y_predict[0]
print(classes[y_predict[0]])
print(classes[y_predict[1]])