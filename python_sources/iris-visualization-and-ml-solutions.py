#!/usr/bin/env python
# coding: utf-8

# # Contents
# 1. [Importing Libraries and Packages](#p1)
# 2. [Loading and Viewing Data Set](#p2)
# 4. [Visualization](#p3)
# 5. [Initializing, Optimizing, and Predicting](#p4)

# <a id="p1"></a>
# # 1. Importing Libraries and Packages
# We will use these packages to help us manipulate the data and visualize the features/labels as well as measure how well our model performed. Numpy and Pandas are helpful for manipulating the dataframe and its columns and cells. We will use matplotlib along with Seaborn to visualize our data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import matplotlib.pyplot as plt
import seaborn as sns

# Importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="p2"></a>
# # 2. Loading and Viewing Data Set
# With Pandas, we can load both the training and testing set that we wil later use to train and test our model. Before we begin, we should take a look at our data table to see the values that we'll be working with. We can use the head and describe function to look at some sample data and statistics.

# In[ ]:


# Importing Data
data = pd.read_csv("../input/Iris.csv")


# In[ ]:


# Showing first five columns
data.head()


# In[ ]:


# Showing last five columns
data.tail()


# In[ ]:


# Checking Null Values
data.isnull().sum()


# #### As you can see our data clean.

# In[ ]:


# We need to drop useless columns
data.drop(["Id"], axis = 1, inplace = True)


# In[ ]:


# Statistics Features
data.describe()


# <a id="p3"></a>
# # 3. Visualization
# 
# In order to visualizate the data, we are goingo to use matplotlib and seaborn.

# In[ ]:


sns.jointplot(data.loc[:,'SepalLengthCm'], data.loc[:,'PetalLengthCm'], kind="regg", color="#ce1414")


# In[ ]:


sns.set(style="white")
df = data.loc[:,['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=4)


# In[ ]:


# Histogram
# bins = number of bar in figure
data.SepalLengthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))
data.SepalWidthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))
data.PetalLengthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))
data.PetalWidthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))
plt.show()


# In[ ]:


# Correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


# Create a trace
trace = go.Scatter(
    x = data.SepalLengthCm,
    y = data.SepalWidthCm,
    mode = 'markers'
)

data_2 = [trace]
fig = dict(data = data_2)
iplot(fig)


# In[ ]:


# Trace 2
trace1 = go.Scatter(
    x  = data.PetalLengthCm,
    y  = data.PetalWidthCm,
    mode = 'markers',
    marker = dict(
        size = 16,
        colorscale = 'Viridis',
        showscale = True
    )
)

data_1 = [trace1]
fig = dict(data = data_1)
iplot(fig)


# In[ ]:


trace2 = go.Box(
    y = data.PetalLengthCm
)

trace3 = go.Box(
    y = data.PetalWidthCm
)

data_2 = [trace2, trace3]
fig = dict(data = data_2)
iplot(fig)


# <a id="p4"></a>
# # 4. Initializing, Optimizing, and Predicting
# Now that our data has been processed and formmated properly, and that we understand the general data we're working with as well as the trends and associations, we can start to build our model. We can import different classifiers from sklearn. 

# ### Splitting The Data into Training And Testing Dataset

# In[ ]:


# Split into train and test
# The attribute test_size = 0.2 splits the data into 80% and 20% ratio. train = 80% and test = 20%
train, test = train_test_split(data, test_size = 0.3)


# In[ ]:


# Four parameter going to help predict our main subject
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# Output our training data
train_x = train.Species
test_Y = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# Output our test data
test_y = test.Species


# ### Logistic Regression

# In[ ]:


# Implement our regression
model = LogisticRegression()
# Fitting the model
model.fit(train_X,train_x)
# Predict the fitting data
prediction = model.predict(test_Y)
print('The accuracy is', metrics.accuracy_score(prediction,test_y))


# ### Support Vector Machines (SVM)

# In[ ]:


# Implementing SVM
model = svm.SVC()
# Fitting the model
model.fit(train_X,train_x)
# Predict the fitting data
prediction = model.predict(test_Y) 
print('The accuracy is:',metrics.accuracy_score(prediction,test_y))


# ### Decision Tree

# In[ ]:


# Implementing Decision Tree Classifier
model = DecisionTreeClassifier()
# Fitting the model
model.fit(train_X,train_x)
prediction = model.predict(test_Y)
print('The accuracy is',metrics.accuracy_score(prediction,test_y))


# ### K-Nearest Neighbours

# In[ ]:


# This examines 3 neighbours
model = KNeighborsClassifier(n_neighbors=3) 
# Fitting the model
model.fit(train_X,train_x)
prediction = model.predict(test_Y)
print('The accuracy is',metrics.accuracy_score(prediction,test_y))


# In[ ]:


# Find Best K Value
score_list = []
for each in range(1,50):
    knn_2 = KNeighborsClassifier(n_neighbors = each)
    knn_2.fit(train_X, train_x)
    score_list.append(knn_2.score(test_Y,test_y))

plt.plot(range(1,50), score_list)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.show()


# **If you liked the kernel, please upvote or make a comment. They motivate me :)**
