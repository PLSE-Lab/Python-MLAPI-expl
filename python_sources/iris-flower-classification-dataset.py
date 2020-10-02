#!/usr/bin/env python
# coding: utf-8

# We import all the necessary libraries and algorithms,Also we load our source file

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing,cross_validation
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("../input/Iris.csv")
df.head(3)


# In[ ]:


df.head(5)


# 'Id' column seems to be insignificant in classifying the flower,so we simply drop it

# In[ ]:


df.drop(['Id'], axis=1 ,inplace=True)


# Now we need to check for the null values present im our dataset

# In[ ]:


df.info()


# since Species column contains categorical data lets see what all categories are present

# In[ ]:


df['Species'].value_counts()


# Let us see how does sepal and petals seperately classify the flower

# In[ ]:


figure = df[df.Species == "Iris-versicolor"].plot(kind='scatter',x ='SepalLengthCm',y = 'SepalWidthCm' , label = 'Versicolor' , color = 'Blue')
df[df.Species == 'Iris-setosa'].plot(kind = 'scatter',x='SepalLengthCm' , y= 'SepalWidthCm' , label = 'Setosa' , color = 'Red' , ax = figure)
df[df.Species == 'Iris-virginica'].plot(kind='scatter' , x='SepalLengthCm' , y= 'SepalWidthCm' , label = 'Virginica' , color = 'Brown' , ax = figure)
figure.set_xlabel('Sepal Length')
figure.set_ylabel('Sepal Width')
figure.set_title("Sepal length vs sepal width")
plt.show()


# In[ ]:


fig = df[df.Species == "Iris-versicolor"].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',label='versicolor',color='Blue')
df[df.Species == 'Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',label='setosa',color='Red',ax=fig)
df[df.Species == 'Iris-virginica'].plot(kind='scatter' , x='PetalLengthCm' , y = 'PetalWidthCm',label='virginica',color='Brown',ax=fig)
fig.set_xlabel('PetalLengthCm')
fig.set_ylabel('PetalWidthCm')
fig.set_title('Petal Length vs Petal Width')
plt.show()


# Seeing the graph it is pretty ovious that the classification of these flowers on basis of petals is more clear.

# In[ ]:


df.head(5)


# In[ ]:


y = df.Species


# In[ ]:


X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]


# In[ ]:


X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25)


# In[ ]:


X_train.tail(5)


# In[ ]:


X_train.shape


# In[ ]:


df.tail(5)


# Lets train our data using Logistic Regression and check its accuracy against testing data

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)
accuracy = logreg.score(X_test,y_test)
print(accuracy)


# Lets train our data using K Neighbors Classifier and check its accuracy against testing data

# In[ ]:


clf = KNeighborsClassifier(6)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)


# lets predict against some random data 

# In[ ]:


ex = np.array([8.7 , 6.5 , 3.5 , 4.6])
ex = ex.reshape(1,-1)
prediction = logreg.predict(ex)
print (prediction)


# Lets train our data using Support Vector Machine and check its accuracy against testing data

# In[ ]:


classifier = SVC()
classifier.fit(X_train,y_train)
accuracy = classifier.score(X_test,y_test)
print(accuracy)

