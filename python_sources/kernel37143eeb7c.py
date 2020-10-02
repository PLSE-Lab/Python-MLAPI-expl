"""
   @script-author: Sachin Selvan
   @script-date: 13/01/2020
   @script-description: Prediciting the species of iris dataset using KNN Classifier
   @external-packages used: sklearn
"""

#import the load_iris function from the dataset module
from sklearn.datasets import load_iris
#creating a variable to hold the iris dataset and its attributes
iris = load_iris()
type(iris)
#printing the iris data
iris.data
#getting the features of the dataset (column names)
print(iris.feature_names)
#to get the target names of the dataset
print(iris.target_names)
#Integers representing the target species where 0=sentosa, 1=versicolor and 2=virginica
print(iris.target)
#checkinh the number of observations and columns
print(iris.data.shape)
# defining x and y
x = iris.data
y = iris.target
print(x.shape)
print(y.shape)

#Training the Model
#splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)
#shape of train and test objects
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#importing KNeighborsClassifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

#importing a metrics model to check the accuracy 
from sklearn import metrics

#running k=1 till 25 and recording it
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
#testing the accuracy for each k value in the specified range
scores

#importing package
import matplotlib.pyplot as plt
#plotting the relationship between k and values of the testing accuracy
plt.plot(k_range,scores_list)
plt.xlabel('Value of k')
plt.ylabel('Testing Accuracy')

#choosing k as 5 and training the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

#0 = setosa, 1=versicolor, 2=virginica
classes = {0:'setosa',1:'versicolor',2:'virginica'}

#Making prediction on some unseen data 
#predict for the below two random observations
x_new = [[3,4,5,2],
         [5,4,2,2]]
y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])