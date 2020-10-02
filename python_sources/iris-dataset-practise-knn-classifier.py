#!/usr/bin/env python
# coding: utf-8

# Iris Dataset (ongoing)
# This is trying to call a KNN classifier on the Iris data step. This has been created on the Spyder development environment

# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[ ]:


iris=load_iris()


# In[ ]:


#Split the data into independent variables(X) and dependent variable (Y)
X=iris.data
Y=iris.target


# In[ ]:


# Split Data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(X,Y)


# In[ ]:


# Call K nearest neighbours function (for now)
knn = KNeighborsClassifier(n_neighbors=5)


# In[ ]:


#Fit the model with training data
knn.fit(x_train,y_train)


# In[ ]:


# Predict the dependent variable
y_pred=knn.predict(x_test)


# In[ ]:


# Check the accuracy score with the y_test data
AccuScore= metrics.accuracy_score(y_test, y_pred)
print(AccuScore)

