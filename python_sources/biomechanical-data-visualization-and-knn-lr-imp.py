#!/usr/bin/env python
# coding: utf-8

# # Contents
# 1. [Importing Libraries and Packages](#p1)
# 2. [Loading and Viewing Data Set](#p2)
# 3. [Clean and Normalization Data](#p3)
# 4. [Visualization](#p4)
# 5. [Initializing, Optimizing, and Predicting](#p5)

# <a id="p1"></a>
# # 1. Importing Libraries and Packages
# We will use these packages to help us manipulate the data and visualize the features/labels as well as measure how well our model performed. Numpy and Pandas are helpful for manipulating the dataframe and its columns and cells. We will use matplotlib along with Seaborn to visualize our data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

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
data = pd.read_csv("../input/column_2C_weka.csv")
print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')


# In[ ]:


# Showing first five columns
data.head()


# In[ ]:


# Showing last five columns
data.tail()


# In[ ]:


data.describe()


# <a id="p3"></a>
# # 3. Clean and Normalization Data
# We need to change categorical data to numeric data and we have to normalize the data.

# In[ ]:


A = data[data.class_2 == "Abnormal"]
N = data[data.class_2 == "Normal"]


# In[ ]:


# Converting 
data.class_2 = [1 if each == "Abnormal" else 0 for each in data.class_2]
y = data.class_2.values
x_data = data.drop(["class_2"], axis = 1)


# In[ ]:


# Normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)) 


# <a id="p4"></a>
# # 4. Visualization
# 
# In order to visualizate the data, we are goingo to use matplotlib and seaborn. Before the visualization don't forget the normalize the data.

# In[ ]:


plt.scatter(A.pelvic_incidence, A.pelvic_radius, color = "red")
plt.scatter(N.pelvic_incidence, N.pelvic_radius, color = "green")
plt.xlabel("Pelvic Incidence")
plt.ylabel("Pelvic Radius")
plt.show()


# In[ ]:


sns.jointplot(x.loc[:,'pelvic_radius'], x.loc[:,'pelvic_incidence'], kind="regg", color="#ce1414")


# In[ ]:


sns.countplot(x="class_2", data=data)
data.loc[:,'class_2'].value_counts()


# In[ ]:


color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class_2']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class_2'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()


# In[ ]:


sns.set(style="white")
df = x.loc[:,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# In[ ]:


# Correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


# Histogram
# bins = number of bar in figure
data.pelvic_incidence.plot(kind = 'hist', bins = 50, figsize = (15,15))
data.sacral_slope.plot(kind = 'hist', bins = 50, figsize = (15,15))
data.pelvic_radius.plot(kind = 'hist', bins = 50, figsize = (15,15))
plt.show()


# <a id="p5"></a>
# # 5. Initializing, Optimizing, and Predicting
# Now that our data has been processed and formmated properly, and that we understand the general data we're working with as well as the trends and associations, we can start to build our model. We can import different classifiers from sklearn. 

# # KNN Implementation

# In[ ]:


# Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)


# In[ ]:


# KNN Implementation 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)


# In[ ]:


print("{} nn score: {}".format(3,knn.score(x_test,y_test)))


# In[ ]:


# Find Best K Value
score_list = []
for each in range(1,50):
    knn_2 = KNeighborsClassifier(n_neighbors = each)
    knn_2.fit(x_train, y_train)
    score_list.append(knn_2.score(x_test,y_test))

plt.plot(range(1,50), score_list)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.show()


# # Logistic Regression Implementation

# In[ ]:


x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("X Train: ", x_train.shape)
print("X Test: ", x_test.shape)
print("Y Train: ", y_train.shape)
print("Y Test: ", y_test.shape)


# In[ ]:


# Initialize 
# Let's initialize parameters

def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0
    return w,b


# In[ ]:


# Calculation of z
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# In[ ]:


# Forward and Backward Propagation
# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients


# In[ ]:


# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# In[ ]:


# Prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 100) 


# In[ ]:


# sklearn
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("Test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("Train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))

