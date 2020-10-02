#!/usr/bin/env python
# coding: utf-8

# <title>IRIS  Dataset Analysis and Prediction</title>

# <h1>Iris Dataset - Data Analysis and Predictions</h1>

# In[1]:


# Import Libraries
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# <h2>Exploratory Data Analysis</h2>

# In[2]:


# Load Data
iris = datasets.load_iris()
iris_y = iris.target
iris = pd.DataFrame(iris.data,columns=iris.feature_names)
iris.sample(5)


# In[3]:


# Check Data Quality
(((iris.isna()*1).sum())/len(iris))*100


# In[4]:


# Project data into scatter plot to see the correlation
fig = plt.figure(figsize=(12,6))
ax_sp = fig.add_subplot(121)
ax_pt = fig.add_subplot(122)
ax_sp.scatter(x=iris['sepal length (cm)'],y=iris['sepal width (cm)'])
ax_pt.scatter(x=iris['petal length (cm)'],y=iris['petal width (cm)'])


# <h3>Observations:</h3>
# <ul>
#     <ol>=> There seems to be no important correlation between Sepal length and width</ol>
#     <ol>=> There is a linear correlation between Sepal length and width</ol>
# </ul>

# In[5]:


# Distribution of Sepal and Petal values
fig = plt.figure(figsize=(12,6))
ax_spl = fig.add_subplot(221)
ax_spw = fig.add_subplot(222)
ax_ptl = fig.add_subplot(223)
ax_ptw = fig.add_subplot(224)
sns.kdeplot(iris['sepal length (cm)'],ax=ax_spl,shade=True,color='blue')
sns.kdeplot(iris['sepal width (cm)'],ax=ax_spw,shade=True,color='red')
sns.kdeplot(iris['petal length (cm)'],ax=ax_ptl,shade=True,color='green')
sns.kdeplot(iris['petal width (cm)'],ax=ax_ptw,shade=True,color='orange')


# <h3>Observations:</h3>
# <ul>
#     <ol>=> Sepal Length and Width data is normally distributed</ol>
#     <ol>=> Petal Length and Width data has a bimodal distribution</ol>
# </ul>

# In[6]:


# Normalise the data into 0 - 1 range
def NormVar(data):
    return ((data - data.min())/(data.max() - data.min()))


# In[7]:


# Distribution of Sepal and Petal values after applying normalisation
fig = plt.figure(figsize=(16,6))
ax_spl = fig.add_subplot(221)
ax_spw = fig.add_subplot(222)
ax_ptl = fig.add_subplot(223)
ax_ptw = fig.add_subplot(224)
sns.kdeplot(NormVar(iris['sepal length (cm)']),ax=ax_spl,shade=True,color='blue')
sns.kdeplot(NormVar(iris['sepal width (cm)']),ax=ax_spw,shade=True,color='red')
sns.kdeplot(NormVar(iris['petal length (cm)']),ax=ax_ptl,shade=True,color='green')
sns.kdeplot(NormVar(iris['petal width (cm)']),ax=ax_ptw,shade=True,color='orange')


# In[8]:


# Apply Noramlisation into the data set to bring them between 0 and 1
for i in iris.columns:
    iris[i] = NormVar(iris[i])
iris.head()


# In[9]:


# Lets look at correlation between variables
plt.figure(figsize=(12,6))
sns.heatmap(iris.corr(),annot=True)


# In[10]:


# As there is a very high correlation between petal length and Width, lets combine to make single column.
iris['petal'] = NormVar(iris['petal length (cm)'] * iris['petal length (cm)'])
sns.kdeplot(iris['petal'],shade=True)


# In[11]:


# Drop the 2 columns as they are redundant now
iris = iris.drop(['petal length (cm)','petal width (cm)'],axis=1)


# <h2>Predictive Analytics using KNN</h2>

# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# In[81]:


# Split Data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(iris,iris_y,test_size=0.3)


# In[82]:


print("Shape of Train/Test Data:")
print("=> Train X : ",X_train.shape)
print("=> Train y : ",y_train.shape)
print("=> Test X  : ",X_test.shape)
print("=> Test y  : ",y_test.shape)


# In[85]:


# fit the model with different values of n_neighbors, and plot the accuracy score
acc=list()
for i in range(1,11):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc.append(round(metrics.accuracy_score(pred,y_test)*100,2))
plt.plot(acc)


# In[84]:


# get the final predictions
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train,y_train)
pred = model.predict(X_test)
round(metrics.accuracy_score(pred,y_test)*100,2)


# In[ ]:




