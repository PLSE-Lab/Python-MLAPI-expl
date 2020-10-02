# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""
@script-author:Srija Srinivasan
@script-name: k-NN Classifier using Iris data
@script-description:K-Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning.
Here it is applied to Iris dataset to classify the different species of Iris based on their Sepal and Petal measurements.
@external-packages used: sklearn.
"""
#import sklearn library
import sklearn.model_selection as ms
import sklearn.preprocessing as pre
#import metrics module to check accuracy
import sklearn.metrics as sms
#import load_iris function from dataset module
from sklearn.datasets import load_iris
#create an object containing iris dataset and it's attributes
iris =load_iris()
#print the iris dataset
data=iris.data
target=iris.target
X=data.tolist()
Y=target.tolist()
# split the datset into test and train using train_test_split function 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=ms.train_test_split(X,Y,test_size=0.25,random_state=0)
# import KNeighborsClassifier class from sklearn
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
sms.accuracy_score(y_test,y_pred)
