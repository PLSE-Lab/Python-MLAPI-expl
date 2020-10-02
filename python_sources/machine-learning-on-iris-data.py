#!/usr/bin/env python
# coding: utf-8

# # Getting Started
# 
# * I am starting a journey towards Data Science and here is the first draft.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Data Information
# * Iris Dataset is about the three species of different flowers namely **Iris-Setosa, Iris-Virginica, and Iris-Versicolor**
# * It consist of 150 rows and 6 columns.
# * Iris is supposed to be an academic dataset and considered for study purpose.
# * At the start of your Data Science Journey you are expected to predict the species of the flower using different machine learning models.

# # Loading the Data
# * Python has a ton of Datasets and can be loaded on to the Kernel easily. There is also an option to upload your personal data onto the Kernel.

# In[ ]:


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.head()


# In[ ]:


iris.info()


# In[ ]:


iris.drop('Id',axis=1,inplace=True)


# # Plot What you have.
# * Plotting helps to give you an initial idea.
# * In this cas we plot Sepal Length Vs Sepal Width in order to find put any existing relation within them.
# * Although there seems to be no relation between them Petal Length Vs Petal Width gives you a clear relation within them.

# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='Scatter',x='SepalLengthCm',y='SepalWidthCm'
                                             ,color='Orange',label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='Scatter',x='SepalLengthCm',y='SepalWidthCm'
                                           ,color='Blue',label='Versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='Scatter',x='SepalLengthCm',y='SepalWidthCm'
                                          ,color='Green',label='Virginica', ax=fig)
fig.set_xlabel('Sepal Length')
fig.set_ylabel('Sepal Width')
fig.set_title('Sepal Length Vs Width Scatter Plot')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# As we can see there is no direct relationship in the Sepal Length and Sepal Width in Iris, further we try to plot Petal Length against Petal Width to determine whether they follow any pattern or not.

# In[ ]:


fig=iris[iris.Species=='Iris-setosa'].plot(kind='Scatter',x='PetalLengthCm',y='PetalWidthCm',color='Orange',label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='Scatter',x='PetalLengthCm',y='PetalWidthCm',color='Blue',label='Versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='Scatter',x='PetalLengthCm',y='PetalWidthCm',color='Green',label='Virginica',ax=fig)
fig.set_xlabel='Petal Length'
fig.set_ylabel='Petal Width'
fig.set_title='Petal Length Vs Width Scatter Plot'
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# # Histogram

# In[ ]:


iris.hist(edgecolor='black',linewidth='1.5')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# #Violin Plot
# * Violin plot shows the density distribution of length and width in the species. The thinner area means the density is less and the broader part suggests that the density is more.

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris,label='Species Vs Petal Length')
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris,label='Species Vs Petal Width')
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris,label='Species Vs Sepal Length')
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris,label='Species Vs Sepal Width')


# # Machine Learning Models
# * Importing the requred packages to implement models.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


iris.shape


# # Correlation Plotting
# * As the name suggests it gives you a relationship score among all the columns of the Dataset.
# * A negative value means If one value increases the other decreases.
# * Plotting it into a heatmap helps in better visualization.

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(iris.corr(),annot=True,color='red')
plt.show()


# Heatmap shows higher correlation between ***Petal Length and Petal Width***, also in ***Petal Length and Sepal Length***

# # Splitting the Data
# * Splitting the data into train and test Dataframe.
# * The standard convention followed is 70% train and 30% test dataframe.

# In[ ]:


train,test=train_test_split(iris,test_size=0.3)
print(train.shape)
print(test.shape)


# In[ ]:


x_train=train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y_train=train.Species
x_test=test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y_test=test.Species


# In[ ]:


x_train.head(5)


# In[ ]:


x_test.head(5)


# In[ ]:


y_train.head(5)


# # Model Implementation
# * The Machine Learning models which we have imported in the previous stage are implemented here. Its list includes-
# 
# 1. Support Vector Machine (SVM)
# 2. Logistic Regression
# 3. Decision Tree Classifier
# 4. K Neighbors Classifier
# 5. Random Forest Classifier

# In[ ]:


model=svm.SVC()
model.fit(x_train,y_train)
prediction1=model.predict(x_test)
print('The accuracy of SVM is:' ,metrics.accuracy_score(prediction1,y_test))


# In[ ]:


model=LogisticRegression()
model.fit(x_train,y_train)
prediction2=model.predict(x_test)
print('The accuracy of Logistic Regression is:' ,metrics.accuracy_score(prediction2,y_test))


# In[ ]:


model=DecisionTreeClassifier()
model.fit(x_train,y_train)
prediction3=model.predict(x_test)
print('The accuracy score of Decision Tree is:' ,metrics.accuracy_score(prediction3,y_test))


# In[ ]:


model=KNeighborsClassifier()
model.fit(x_train,y_train)
prediction4=model.predict(x_test)
print('The model accuracy score of K Neighbors Classifier is:' ,metrics.accuracy_score(prediction4,y_test))


# In[ ]:


model=RandomForestClassifier()
model.fit(x_train,y_train)
prediction5=model.predict(x_test)
print('The accuracy of Random Forest Classifier is:' ,metrics.accuracy_score(prediction5,y_test))


# In[ ]:


df={'Model Name':['Support Vector Machine','Logistic Regression','Decision Tree','K Neighbors',
                 'Random Forest Classifier'],
   'Accuracy':[metrics.accuracy_score(prediction1,y_test),metrics.accuracy_score(prediction2,y_test),
              metrics.accuracy_score(prediction3,y_test),metrics.accuracy_score(prediction4,y_test),
              metrics.accuracy_score(prediction5,y_test)]}
model_accuracy=pd.DataFrame(df,columns=['Model Name','Accuracy'])
print(model_accuracy)


# # Plotting the Acquired Results

# In[ ]:


fig=model_accuracy.plot(kind='bar',x='Model Name',y='Accuracy')
fig.set_ylabel='Accuracy'
fig.set_title='Multiple Model Accuracy Graph'
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# # Support and Upvote
# * If you have read till here and you find my kernel helpful don't forget to upvote and if you have any suggestions plese let me know in the comments.
