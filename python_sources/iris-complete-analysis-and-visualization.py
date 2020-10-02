#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Importing the necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading the Iris Dataset

iris = pd.read_csv("/kaggle/input/iris/Iris.csv")


# ## **Summarizing the data**

# In[ ]:


iris.head()


# ### Removing the Unwanted columns

# In[ ]:


iris.drop('Id', axis = 1, inplace = True)


# In[ ]:


# Dimensions of te Data

iris.shape


# In[ ]:


# Statistical summary of all attributes

iris.describe()


# In[ ]:


# To know how many variety of species are present in the data

iris.groupby('Species').size()


# ## **Understanding the IRIS dataset**

# **1. Relational plot**

# In[ ]:


sb.relplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = iris, hue = 'Species')


# - -----------------------------------------------------------------------------------------------------------------
# - **setosa** has small sepal_length and high sepal_width
# - **versicolor** has medium sepal_length and medium sepal_width
# - **virginica** has high sepal_length and medium sepal_width
# - -----------------------------------------------------------------------------------------------------------------

# In[ ]:


sb.relplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = iris, hue = 'Species')


# - **setosa** has less petal_length and less petal_width
# - **versicolor** has medium petal_length and medium petal_width
# - **virginica** has more petal_length and more petal_width

# In[ ]:


sb.relplot(x = 'SepalLengthCm', y = 'SepalWidthCm', hue = 'PetalLengthCm', data = iris, col = 'Species')


# - **setosa** has less petal_length
# - **versicolor** has medium petal_length
# - **virginica** has more petal_length

# In[ ]:


sb.relplot(x = 'SepalLengthCm', y = 'SepalWidthCm', hue = 'PetalWidthCm', data = iris, col = 'Species')


# - **setosa** has less petal_width
# - **versicolor** has medium petal_width
# - **virginica** has more petal_width

# **2. Categorical plot**

# In[ ]:


sb.catplot(x = 'Species', y = 'SepalLengthCm', data = iris, kind = 'swarm')


# - ***From this we can clearly classify between setosa and virginica by means of sepal_length***

# In[ ]:


sb.catplot(x = 'Species', y = 'SepalWidthCm', data = iris, kind = 'swarm')


# - ***From this, we can classify setosa and versicolor by means of sepal_width***

# In[ ]:


sb.catplot(x = 'Species', y = 'PetalLengthCm', data = iris, kind = 'swarm')


# - **This petal_length column will be working as the major classifier which clearly classifies all the three types *setosa, versicolor and virginica***

# In[ ]:


sb.catplot(x = 'Species', y = 'PetalWidthCm', data = iris, kind = 'swarm')


# - **This petal_width column is also a major classifier, which helps us to classify setosa, versicolor and virginica**

# ### Checking for Outliers

# In[ ]:


sb.catplot(x = 'Species', y = 'SepalLengthCm', data = iris, kind = 'box')


# - **virginica** has the outliers with respect to sepal_length

# In[ ]:


sb.catplot(x = 'Species', y = 'SepalWidthCm', data = iris, kind = 'box')


# - **setosa** has two outliers and **virginica** has two outliers with respect to sepal_width

# In[ ]:


sb.catplot(x = 'Species', y = 'PetalLengthCm', data = iris, kind = 'box')


# - **setosa and versicolor** has one minor outliers with respect to petal_length

# In[ ]:


sb.catplot(x = 'Species', y = 'PetalWidthCm', data = iris, kind = 'box')


# - **setosa** has minor outlier with respect to petal_width

# ### Correlation between the columns

# In[ ]:


corr = iris.corr(method = 'pearson')
corr


# In[ ]:


sb.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, annot = True, fmt = '.0%')


# - **petal_length and petal_width** has the higest positive correlation of **96%**, so that we can drop anyone of the column inorder to increase the accuracy.
# - **petal_length and sepal_length** has the next highest positive correlation of **87%**, so that we can drop anyone of the column in order to increase the accuracy.
# - **sepal_length and petal_width** has the positive correlation of **82%**.

# In[ ]:


sb.pairplot(data = iris, hue = 'Species', height = 3)


# # Model creation

# In[ ]:


# Selecting Dependent and Independent Variables
X = iris.iloc[:, [1, 3]].values
y = iris.iloc[:, -1].values


# In[ ]:


from sklearn.preprocessing import LabelEncoder
species_encoder = LabelEncoder()
species_encoder.fit(y)

y = species_encoder.transform(y)


# In[ ]:


# Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


from sklearn import svm, tree
from sklearn.metrics import accuracy_score, confusion_matrix

classifier = svm.SVC(gamma = 'auto', kernel = 'rbf')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy Score\t:\t", accuracy_score(y_test, y_pred))
print("Confusion Matrix\t:\t")
print(confusion_matrix(y_test, y_pred))


# In[ ]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


# In[ ]:


plt.figure(figsize = (10, 7))

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('blue', 'orange', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
               c = [ListedColormap(('white', 'red', 'black'))(i)], label = j)


plt.title('IRIS-Train Set')
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()
plt.show()
plt.savefig("train.png")


# In[ ]:


plt.figure(figsize = (10, 7))

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('blue', 'orange', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
               c = [ListedColormap(('white', 'red', 'black'))(i)], label = j)
    
plt.title('IRIS - Test Set')
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()
plt.show()

plt.savefig("test.png")


# - I hope you enjoyed this quick introduction to some of the quick, simple data visualizations you can create with pandas, seaborn, and matplotlib in Python!
# 
# - I encourage you to run through these examples yourself, tweaking them and seeing what happens. From there, you can try applying these methods to a new dataset and incorprating them into your own workflow!
# 
# 

# In[ ]:




