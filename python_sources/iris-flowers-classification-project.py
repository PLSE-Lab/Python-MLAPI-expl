#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Importing the dataset

# In[ ]:


df= pd.read_csv('../input/iris/Iris.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df['Species'].value_counts()


# # Drop Unwanted Columns

# In[ ]:


df.drop('Id',axis=1,inplace=True)


# # Exploratory Data Analysis

# ### Sepal

# In[ ]:


fig = df[df.Species == 'Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='royalblue',label='Iris-setosa')
df[df.Species == 'Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='darkorange',label='Iris-versicolor',ax=fig)
df[df.Species == 'Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='forestgreen',label='Iris-virginica',ax=fig)

fig.set_xlabel('Sepal Length')
fig.set_ylabel('Sepal Width')
fig.set_title('Sepal Length vs Width')


# In[ ]:


sns.pairplot(df, hue='Species')


# In[ ]:


sns.FacetGrid(df, hue='Species',height=5).map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()


# ### Petal

# In[ ]:


fig = df[df.Species == 'Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='royalblue',label='Iris-setosa')
df[df.Species == 'Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='darkorange',label='Iris-versicolor',ax=fig)
df[df.Species == 'Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='forestgreen',label='Iris-virginica',ax=fig)

fig.set_xlabel('Petal Length')
fig.set_ylabel('Petal Width')
fig.set_title('Petal Length vs Width')


# In[ ]:


df.hist(edgecolor='black',linewidth=1.2)


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
sns.violinplot(data=df, x='Species',y='SepalLengthCm')
plt.subplot(2,2,2)
sns.violinplot(data=df, x='Species',y='SepalWidthCm')

plt.subplot(2,2,3)
sns.violinplot(data=df,x='Species', y='PetalLengthCm')
plt.subplot(2,2,4)
sns.violinplot(data=df, x='Species', y='PetalWidthCm')


# As this is a **Classification Problem**, we will use Classification Algorithms to build our model.

# ### Importing Packages for Classification algorithms

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


sns.heatmap(df.corr(), annot=True, cmap='seismic')
plt.show()


# In the above figure, we can see that Sepal Length and Width are not correlated. While, the Petal Length and Width are *highly correlated*.

# ## 1. We will use "ALL" the features to Train the algorithm and Check the accuracy.

# ### Splitting the data into Train and Test set

# In[ ]:


X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


y_train.head()


# ### Logistic Regression (LR)

# In[ ]:


model = LogisticRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
print('Logistic Regression accuracy = ', metrics.accuracy_score(prediction,y_test))


# ### Support Vector Machine (SVM)

# In[ ]:


model = svm.SVC()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
print('SVM accuracy = ', metrics.accuracy_score(prediction,y_test))


# ### Decision Tree

# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
print('Decision Tree accuracy = ', metrics.accuracy_score(prediction,y_test))


# ### K-Nearest Neighbors (KNN)

# In[ ]:


model = KNeighborsClassifier()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
print('KNN accuracy = ', metrics.accuracy_score(prediction,y_test))


# **CONCLUSION:** By applying the above 4 Machine Learning algorithms, we see that all our models give the exact same High Accuracy.

# ## 2. We will use "Sepal and Petal features separately" to Train the algorithm and Check the accuracy.

# ### Splitting the Sepal data and Petal data into Train and Test set

# ### Sepal

# In[ ]:


sepal_X = df[['SepalLengthCm','SepalWidthCm']]
sepal_y = df['Species']

sepal_X_train, sepal_X_test, sepal_y_train, sepal_y_test = train_test_split(sepal_X, sepal_y, test_size=0.3,random_state=0)


# ### Petal

# In[ ]:


petal_X = df[['PetalLengthCm','PetalWidthCm']]
petal_y = df['Species']

petal_X_train, petal_X_test, petal_y_train, petal_y_test = train_test_split(petal_X, petal_y, test_size=0.3,random_state=0)


# ### Logistic Regression (LR)

# In[ ]:


model = LogisticRegression()

model.fit(sepal_X_train,sepal_y_train)

prediction = model.predict(sepal_X_test)
print('Logistic Regression accuracy for Sepal = ', metrics.accuracy_score(prediction,sepal_y_test))

model.fit(petal_X_train,petal_y_train)

prediction = model.predict(petal_X_test)
print('Logistic Regression accuracy for Petal = ', metrics.accuracy_score(prediction,petal_y_test))


# ### Support Vector Machine (SVM)

# In[ ]:


model = svm.SVC()

model.fit(sepal_X_train,sepal_y_train)

prediction = model.predict(sepal_X_test)
print('SVM accuracy for Sepal = ', metrics.accuracy_score(prediction,sepal_y_test))

model.fit(petal_X_train,petal_y_train)

prediction = model.predict(petal_X_test)
print('SVM accuracy for Petal = ', metrics.accuracy_score(prediction,petal_y_test))


# ### Decision Tree

# In[ ]:


model = DecisionTreeClassifier()

model.fit(sepal_X_train,sepal_y_train)

prediction = model.predict(sepal_X_test)
print('Decision Tree accuracy for Sepal = ', metrics.accuracy_score(prediction,sepal_y_test))

model.fit(petal_X_train,petal_y_train)

prediction = model.predict(petal_X_test)
print('Decision Tree accuracy for Petal = ', metrics.accuracy_score(prediction,petal_y_test))


# ### K-Nearest Neighbors (KNN)

# In[ ]:


model = KNeighborsClassifier()

model.fit(sepal_X_train,sepal_y_train)

prediction = model.predict(sepal_X_test)
print('KNN accuracy for Sepal = ', metrics.accuracy_score(prediction,sepal_y_test))

model.fit(petal_X_train,petal_y_train)

prediction = model.predict(petal_X_test)
print('KNN accuracy for Petal = ', metrics.accuracy_score(prediction,petal_y_test))


# **CONCLUSION:** By applying the above 4 Machine Learning algorithms, we see that using Petal over Sepal gives us more accuracy.

# In[ ]:




