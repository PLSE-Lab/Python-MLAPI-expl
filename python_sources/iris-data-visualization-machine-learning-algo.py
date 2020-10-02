#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # Linear Algebra
import pandas as pd # Data Processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Load the dataset
Iris=pd.read_csv('../input/iris/Iris.csv')
Iris.head(n=10)#Checking the first 10 rows of the dataset


# In[ ]:


Iris.info()#Checking the type of data we have in dataset
Iris.isnull().sum()#Checking for null values, as there is no missing data.Hence we can process the dataset


# In[ ]:


#Now dropping the Id column, which is unnecessary for further analysis
Iris.drop('Id',axis=1,inplace=True)#axis=1 means removing col wise
Iris.head(n=10)#Again checking for the first 10 rows of dataset


# In the above data we have removed Id col which is unnecessary, now data is all good for further analysis..
# Hope you guys enjoy while coding..

# #Data cleaning is done, now it's time for some Exploratory Data Analysis(EDA)
# #First and foremost,we are going to see how data is distributed among Sepal length and Sepal width using a Scatter plot
# 
# * **A scatter plot is a type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data. If the points are coded, one additional variable can be displayed.

# In[ ]:


#Exploratory Data Analysis
#Sepal Length VS Sepal Width
fig=Iris[Iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',marker='x',color='#fa6c33',label='Setosa')
fig=Iris[Iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',marker='*',color='#3c8991',label='Versicolor',ax=fig)
fig=Iris[Iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',marker='D',color='#d5081e',label='Virginica',ax=fig)
fig.set_xlabel('Sepal Length')
fig.set_ylabel('Sepal Width')
fig.set_title('Sepal Length VS Sepal Width')
fig=plt.gcf()
fig.set_size_inches(10,6)
sns.set_style("darkgrid")
plt.show()


# The above plot shows the relationship between Sepal Length and Sepal Width****

# Wow! The above plot looks so amazing,now i am so excited to see how data is distributed among Petal Length and Petal Width..Without any delay let see..

# In[ ]:


#Exploratory Data Analysis
#Petal Length VS Petal Width
fig=Iris[Iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',marker='x',color='#270c8c',label='Setosa')
fig=Iris[Iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',marker='o',color='#d5081e',label='Versicolor',ax=fig)
fig=Iris[Iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',marker='>',color='#45aa53',label='Virginica',ax=fig)
fig.set_xlabel('Petal Length')
fig.set_ylabel('Petal Width')
fig.set_title('Petal Length VS Petal Width')
fig=plt.gcf()
fig.set_size_inches(10,6)
sns.set_style("darkgrid")
plt.show()


# *****The above plot shows the relationship between Petal Length and Petal Width

# #Now let us see how length and width may vary according to the species
# **# using violin plot from seaborn**
# 
# * *A violin plot is a method of plotting numeric data. It is similar to a box plot, with the addition of a rotated kernel density plot on each side. Violin plots are similar to box plots, except that they also show the probability density of the data at different values, usually smoothed by a kernel density estimator*

# In[ ]:


#Violin plot

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='SepalLengthCm',palette='muted',inner='quartile',data=Iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='SepalWidthCm',palette='muted',inner='quartile',data=Iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='PetalLengthCm',palette='muted',inner='quartile',data=Iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='PetalWidthCm',palette='muted',inner='quartile',data=Iris)
plt.show()


# #The above violin plot shows density of length and width of the Species..the thinner part shows that there is lesser density..The thicker part shows that there is higher density

# #Now lets see how length and width are distributed among Sepals and Petals using a histogram
# 
# * A histogram is a graphical display of data using bars of different heights. In a histogram, each bar groups numbers into ranges. Taller bars show that more data falls in that range. A histogram displays the shape and spread of continuous sample data.
# 
# 

# In[ ]:


#Histogram
Iris.hist(edgecolor='black', bins=15,color='#C11321',linewidth=1.8)
fig=plt.gcf()
fig.set_size_inches(15,9)
plt.show()


# The above plot shows how length and width are distributed..hope you liked my presentation..let's see some more..
# Let's check Pair Plot****
# 
# * A pairs plot allows us to see both distribution of single variables and relationships between two variables . Pair plots are a great method to identify trends for follow-up analysis and, fortunately, are easily implemented in Python!

# In[ ]:


# Pair Plot using seaborn library
pairplot=sns.pairplot(Iris,hue='Species',palette='husl',diag_kind="kde",kind='scatter')


# * Area Plot : Area Plot gives us a visual representation of Various dimensions of Iris flower and their range in dataset.

# In[ ]:


# Area Plot
Iris.plot.area(y=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],alpha=0.5,figsize=(13,9));


# Heat map: Heat map is used to find out the correlation between different features in the dataset.High positive or negative value shows that the features have high correlation.This helps us to select the parmeters for machine learning.

# In[ ]:


# Heat map
fig=plt.gcf()
fig.set_size_inches(13,9)
fig=sns.heatmap(Iris.corr(),annot=True,cmap='YlGnBu',linewidths=1.5,linecolor='k', vmin=0.5,vmax=1.0,square=True,cbar_kws={"orientation": "vertical"},cbar=True)


# Everythig looks so amazing right.! Data visualization is done using different plots..hope you liked it.!
# * Now it's time to choose an machine learning algorithm,when we look up to the dataset..As the data which we have is lablled so we are using following Machine Learning Classification Algorithms for prediction..
# 

# In[ ]:


#Splitting the dataset into X and Y..for that we need below sckitlearn library
X=Iris.iloc[:,0:4]
Y=Iris.iloc[:,4]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


# Importing all the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics #To check accuracy of the model


# In[ ]:


#Logistic Regression
logistic=LogisticRegression()
logistic.fit(X_train,Y_train)
y_prediction=logistic.predict(X_test)
accuracy_logistic=metrics.accuracy_score(y_prediction,Y_test)
print('The accuracy of the Logistic Regression is',accuracy_logistic)


# In[ ]:


#Decision Tree
decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,Y_train)
y_prediction=decision_tree.predict(X_test)
accuracy_decision_tree=metrics.accuracy_score(y_prediction,Y_test)
print('The accuracy of the Decision Tree Classifier is',accuracy_decision_tree)


# In[ ]:


#KNearestNeighbors
knn=KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train,Y_train)
y_prediction=knn.predict(X_test)
accuracy_knn=metrics.accuracy_score(y_prediction,Y_test)
print('The accuracy of the KNN is',accuracy_knn)


# In[ ]:


#Support Vecotr Machine--SVM
sv=svm.SVC()
sv.fit(X_train,Y_train)
y_prediction=sv.predict(X_test)
accuracy_svm=metrics.accuracy_score(y_prediction,Y_test)
print('The accuracy of svm is',accuracy_svm)


# Below algorithm is a part of Ensemble Learning---Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier(learning_rate=0.01,random_state=10)
model.fit(X_train,Y_train)
y_prediction=model.predict(X_test)
accuracy_gradientboostingclassifier=metrics.accuracy_score(y_prediction,Y_test)
print("The accuracy of the gradientboostingclassifier is",accuracy_gradientboostingclassifier)


# In[ ]:


Models=pd.DataFrame({'Algorithm':['Logistic Regression','Decision Tree','KNearestNeighbors','Support Vector Machine','Gradient Boosting'],'Accuracy':[accuracy_logistic,accuracy_decision_tree,accuracy_knn,accuracy_svm,accuracy_gradientboostingclassifier]})
Models.sort_values(by='Accuracy',ascending=False)


# # The above Dataframe shows that high accuracy with logistic,knn and svm..If you like it kindly give me an upvote.!****

# In[ ]:




