#!/usr/bin/env python
# coding: utf-8

# 

# 

# ## ****About the Notebook****
# ### This is a very basic tutorial to the Machine Learning world using the Iris Dataset.
# ### In this notebook I have applied the basic Machine Learning algorithms and tried to get the best accuracy possible.

# 

# 

# 
# 
# ## **1-1 Problem Feature**
# ### The Iris or Fisher's dataset was introduced by Ronald Fisher in 1936 paper - The use of multiple measurements in taxonomic problems.
# ### This dataset contains records of 3 species(Iris Setosa, Iris Virginica & Iris versicolor) with 
# ### 50 samples each. 4 features of each sample was measured i.e. The length and width of Sepals and Petals
# ### The iris dataset is a good dataset for beginners in the field of Machine Learning.
# 
# ## **1-2 Variables in the Dataset**
# ### The dataset contains 6 columns:
# ### * Id
# ### * SepalLengthCm
# ### * SepalWidthCm
# ### * PetalLengthCm
# ### * PetalWidthCm
# ### * Species
# 
# ## **1-3 Aim**
# ### The aim of the dataset is to get the best accuracy by applying different Machine Learning algorithms

# ## Loading the required packages.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# ## Importing the iris dataset

# In[ ]:


# The Python Pandas packages helps us work with our datasets. We start by acquiring the datasets into Pandas DataFrames
iris_dataset = pd.read_csv('../input/Iris.csv')


# In[ ]:


# Preview the data
iris_dataset.head()


# In[ ]:


# Check whether there are any missing values in the dataset
iris_dataset.info()


# In[ ]:


# Id column in the dataset does not contribute hence dropping the 'ID' field.
iris_dataset.drop(['Id'], axis=1, inplace=True)

iris_dataset.head()


# In[ ]:


# Check the unique values present in the Species column
print(iris_dataset['Species'].unique())


# ### We have 3 types of species present in the dataset - Iris-setosa, Iris-versicolor and Iris-virginica

# ## **Visualization**

# In[ ]:


# Plot the relationship between Sepal Length and Sepal width for all the species using matplotlib
fig = iris_dataset[iris_dataset.Species=='Iris-setosa'].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "orange", label='Setosa', marker='x')
iris_dataset[iris_dataset.Species=='Iris-versicolor'].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "green", label='Versicolor', ax = fig, marker='o')
iris_dataset[iris_dataset.Species=='Iris-virginica'].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "blue", label='Virginica', ax = fig, marker = 's')
fig.set_xlabel('Sepal Lenth in Cm')
fig.set_ylabel('Sepal Width in cm')
fig.set_title('Sepal Length vs Width')
fig = plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# ###  In the above scatter plot, we can easily differentiate the Setosa species but its hard to visually differentiate between Versicolor and Virginica.

# In[ ]:


# Plot the relationship between Petal Length and Petal width for all the species using seaborn package.

plt.figure(figsize=(10,6))
ax = sns.scatterplot(x=iris_dataset.PetalLengthCm, y=iris_dataset.PetalWidthCm, hue=iris_dataset.Species, style=iris_dataset.Species)
plt.title('Petal Length vs Width')


# ### In the above plot for Petal length and Petal width, we get a better clustered distribution for different types of species. We get a strong dissection for Setosa with respect to Versicolor and Virginica. Also, a better clustered distribution for Versicolor and Virginica for Petals in comparison to Sepals. Hence we should consider data for Petals for more accurate predictions.

# In[ ]:


# Plotting boxplot 
f,axis = plt.subplots(2,2, figsize = [20,20])
plt.subplot(2,2,1)
sns.boxplot(data=iris_dataset, x = 'Species', y = 'SepalLengthCm')
plt.subplot(2,2,2)
sns.boxplot(data=iris_dataset, x = 'Species', y = 'SepalWidthCm')
plt.subplot(2,2,3)
sns.boxplot(data=iris_dataset, x = 'Species', y = 'PetalLengthCm')
plt.subplot(2,2,4)
sns.boxplot(data=iris_dataset, x = 'Species', y = 'PetalWidthCm')


# In[ ]:


# Generating violin plot to provide a visual distribution of data and its probability density. This provides a combination of Box plot and Density plot.
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
sns.violinplot('Species', 'SepalLengthCm', data = iris_dataset)
plt.subplot(2,2,2)
sns.violinplot('Species', 'SepalWidthCm', data = iris_dataset)
plt.subplot(2,2,3)
sns.violinplot('Species', 'PetalLengthCm', data = iris_dataset)
plt.subplot(2,2,4)
sns.violinplot('Species', 'PetalWidthCm', data = iris_dataset)


# ### From the above boxplots and violin plots its clearly visible that setosa < versicolor < virginica. 
# ### In Violinplot, the thinner part shows there is less density and the thicker portion portrays higher density. The white dot represents the median, the thick grey bar shows the IQR and the thin grey line shows 95% confidence interval.

# In[ ]:


# Boxplot to get a better picture of how the data is distributed.
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
sns.distplot(iris_dataset['SepalLengthCm'], bins=10)
plt.subplot(2,2,2)
sns.distplot(iris_dataset['SepalWidthCm'], bins = 10)
plt.subplot(2,2,3)
sns.distplot(iris_dataset['PetalLengthCm'], bins = 10)
plt.subplot(2,2,4)
sns.distplot(iris_dataset['PetalWidthCm'], bins = 10)


# In[ ]:


# Generating Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(iris_dataset.corr(), annot=True, linewidths=0.4)   # annot displays the value of each cell in the heatmat
plt.show()


# In[ ]:


# From the above heatmap its evident that the Petal Length and Petal Width are highly correlated. On the other hand, the Sepal Length and Sepal Width are not correlated.


# In[ ]:


# Loading the ML algorithms
from sklearn.linear_model import LogisticRegression  # For Logistic Regression
from sklearn.model_selection import train_test_split  # To split the data set into training and testing 
from sklearn.neighbors import KNeighborsClassifier  # for K Nearest neighbors
from sklearn import svm  # for SVM (Support Vector Machines) algorithm
from sklearn import metrics  # for checking model accuracy
from sklearn.tree import DecisionTreeClassifier  # for using Decision Tree Algorithm


# In[ ]:


# Splitting the dataset into Training and Testing
X = iris_dataset.drop(['Species'], axis=1)
y = iris_dataset['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:





# In[ ]:


# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
prediction_log_reg = model.predict(X_test)

print('The accuracy of Logistic Regression', metrics.accuracy_score(prediction_log_reg, y_test))


# In[ ]:


# K-nearest neigbors
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train,  y_train)
prediction_KNear = model.predict(X_test)

print('The accuracy of K-nearest neighbors:', metrics.accuracy_score(prediction_KNear, y_test))


# In[ ]:


a_index = list(range(1,11))
a = pd.Series()
x = [1,2,3,4,5,6,7,8,9,10]
for i in a_index:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    a = a.append(pd.Series(metrics.accuracy_score(prediction, y_test)))
plt.plot(a_index, a)


# In[ ]:


# The above graph shows accuracy levels of KNN models for different values of n


# In[ ]:





# In[ ]:


# We had used all the features of the Iris. Now we will use Petals and Sepals seperately


# In[ ]:


petals = iris_dataset[['PetalLengthCm', 'PetalWidthCm', 'Species']]
sepals = iris_dataset[['SepalLengthCm', 'SepalWidthCm', 'Species']]


# In[ ]:


petals.head()


# In[ ]:


sepals.head()


# In[ ]:


# Splitting the Petals and Sepals dataset into Training and Testing

petals_x = petals.drop(['Species'], axis = 1)
petals_y = petals['Species']

sepals_x = sepals.drop(['Species'], axis = 1)
sepals_y = sepals['Species']

x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(petals_x, petals_y, test_size = 0.3)
x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(sepals_x, sepals_y, test_size = 0.3)


# In[ ]:


print(x_train_p.shape, x_test_p.shape, y_train_p.shape, y_test_p.shape)
print(x_train_s.shape, x_test_s.shape, y_train_s.shape, y_test_s.shape)


# In[ ]:





# In[ ]:


#Logistic Regression
model_log = LogisticRegression()
# 1. Petals
model_log.fit(x_train_p, y_train_p)
prediction = model_log.predict(x_test_p)

print('The accuracy of the model for Petals is:', metrics.accuracy_score(prediction, y_test_p))

# 2. Sepals
model_log.fit(x_train_s, y_train_s)
prediction = model_log.predict(x_test_s)

print('The accuracy of the model for Sepals is:', metrics.accuracy_score(prediction, y_test_s))


# In[ ]:


# Decision Tree 
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(criterion='gini', max_depth=6)  # by defalt it takes Gini index as critera

# 1. Petals
model_dt.fit(x_train_p, y_train_p)
prediction = model_dt.predict(x_test_p)

print('The accuracy of the model for Petals is:', metrics.accuracy_score(prediction, y_test_p))

# 2. Sepals
model_dt.fit(x_train_s, y_train_s)
prediction = model_dt.predict(x_test_s)

print('The accuracy of the model for Sepals is:', metrics.accuracy_score(prediction, y_test_s))


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()

# 1. Petals
model_rf.fit(x_train_p, y_train_p)
prediction = model_rf.predict(x_test_p)

print('The accuracy of the model for Petals is:', metrics.accuracy_score(prediction, y_test_p))

# 2. Sepals
model_rf.fit(x_train_s, y_train_s)
prediction = model_rf.predict(x_test_s)

print('The accuracy of the model for Sepals is:', metrics.accuracy_score(prediction, y_test_s))


# In[ ]:


# K-Nearest Neighbors
model_knn = KNeighborsClassifier(n_neighbors=9)

# 1. Petals
model_knn.fit(x_train_p, y_train_p)
prediction = model_knn.predict(x_test_p)

print('The accuracy of the model for Petals is:', metrics.accuracy_score(prediction, y_test_p))

# 2. Sepals
model_knn.fit(x_train_s, y_train_s)
prediction = model_knn.predict(x_test_s)

print('The accuracy of the model for Sepals is:', metrics.accuracy_score(prediction, y_test_s))


# In[ ]:





# In[ ]:




