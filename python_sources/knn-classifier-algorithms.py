#!/usr/bin/env python
# coding: utf-8

# # Creating Our Own K-Nearest Neighbours Classifiers
# ***
# ## The Dataset
# ***
# We are going to use the Iris data set.
# 
# Iris is a type of flower with three types of species namely
# 
# * *Setosa*
# * *Versicolor*
# * *Virginica*
# 
# Each Species has four measurments :
# 
# * *Sepal Length*
# * *Sepal Width*
# * *Petal Length*
# * *Petal Width*
# 
# To Study More about the Iris Dataset please refer [Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set).

# ## The Algorithm
# ***
# The basic idea of this algorith is to separate(classify) the data into categories from the training data and then use that category to predict the category of the new data.
# 
# To do this we have to compare the Eucledian distance of New Data and with the Old data categries and then predict the  Category of the new data.
# 
# This can be More understanded with this graph
# 
# ![](https://pbs.twimg.com/media/DmVRIqrXcAAOvtH.jpg:large)
# 

# ## Prerequisite 
# ***
# The important Python Packages needed are 
# 
# 1. *Scikit-Learn(sklearn)*
# 2. *Scipy*
# 

# **In the below code cell we have imported all the needed libraries.**

# In[41]:


# importing iris from scikit-learn
from sklearn.datasets import load_iris
# importing accuracy_score to check the accuracy of the prediction
from sklearn.metrics import accuracy_score  
# importing train_test_split to split the data in training and testing parts
from sklearn.model_selection import train_test_split
# importing the distance method to measure the Eucledian distance
from scipy.spatial import distance
# loading iris dataset to a local object
iris = load_iris()


# In[42]:


# taking the iris features in variable X 
X = iris.data
# taking the iris lables in variable y
y = iris.target
# splitting the features and lables in training and testing part
# Size of the testing data is the 20% of the total data i.e 30 examples out of 150 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)


# ## Creating the algorithms
# ***
# ** In the below cell we will write the code for the KNN classifier **
# 
# 1. To create this we will start with cresting an class named **** KNNClassifier ****.
# 2. Then we need three basic methods to run our classifier : 
#     - **fit()** method.
#         - In this method we take the _TRAINING DATASET_.
#     - **predict()** method.
#         - Here we will take the data for which this method return<br> 
#           the predicted output.
#     - **close()** method
#         - This is a method used at the backend of the classifier.<br>
#           In the close method we measure the distance of the _TEST CASE_<br>
#           with each row of the Training Dataset and return the index of <br>
#           the closest row of the Training Dataset.<br>
# 
# 

# In[43]:


# creating the KNNClassifier class
class KNNClassifier:
    def fit(self, X_train, y_train):                # creating the fit method
        self.X_train = X_train                      # getting the features of the training dataset 
        self.y_train = y_train                      # getting the lables of the training dataset 
        
    def predict(self, X_test):                      # creating the pridict method
        pre = []
        for i in X_test:
            res = self.close(i)                     # getting the closeset value from the close method
            pre.append(y_train[res])                # apppending the prediction list with the lable of closest feature
        return pre                                  # returning the predicted array
            
    def close(self, i):                             # creating the close method
        bestdist = distance.euclidean(self.X_train[0],i)   # assuming that first index is the best index
        bestinx = 0
        for j in range(len(self.X_train)):
            if distance.euclidean(self.X_train[j],i) < bestdist:   #measuring Euclidean distance 
                bestinx = j
                bestdist = distance.euclidean(self.X_train[j],i)
            
        return bestinx                              # returning the closest index
    
    
        


# ## Checking the algorithm
# ***
# In the below cell we will fit the data to the algorithm.
# Also we will predict the values of the _TESTING DATA_.
# In the last line will check the accuracy of the predicted
# output with the original values.

# In[44]:


# creating the classifier object clf.
clf = KNNClassifier()
# fitting the Training Dataset in the object.
clf.fit(X_train, y_train)
# storing the predicted output in ans.
ans = clf.predict(X_test)
# checking the Accuracy Score of the prediction 
print(str(accuracy_score(ans, y_test)*100) +"%")


# __As we can see the accuracy is about 96 % which means our classifier is working perfectly.__

# # Something Extra ......
# 
# ***
# 
# In last I am sharing some commands to view the data in the notebook itself.<br>
# - To view Feature names<br><br>
#     > __print(iris.feature_names)__
#     
#     
# - To view Target names<br><br>
#     > __print(iris.target_names)__
#     
#     
# To view the compelete dataset at once using Pandas Dataframe here is the code

# In[45]:


from sklearn.datasets import load_iris                # importing the iris
import pandas as pd                                   # importing the pandas library

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)      # creating the Pandas DATAFRAME
df.head()                                             # printing the first 5 values of the data
# to view the compelete dataset just uncomment the below line
#df

