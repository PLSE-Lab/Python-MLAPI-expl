#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION
# 
# ### In this notebook, gender will be predicted with Logistic Regression algorithm. I will apply this algorithm by hand, I won't use scikit-learn library directly.
# The steps of the work is below
# 
# **1. EDA(Expolatory Data Analysis) for the Gender Recognition by Voice dataset**
# 
#    * [Getting familiar with the dataset](#get-familiar)
#     
# **2. Normalization (If needed)**
# 
# * [Normalization](#normalization)
# 
# **3. Logistic Regression implementation with python (Scikit-learn library will be used different parts of the algorithm)**
# 
# * [Choosing train-test split randomly](#traintest)
# 
# * [Initialize Parameters (w,b)](#initparam)
# 
# * [Implementing Sigmoid function](#sigmoid)
# 
# * [Calculation of cost-loss function](#cost)
# 
# * [Forward-Backward function](#fb)
# 
# * [Purpose of the Derivation *-(Optimization)*](#derivation)
# 
# * [Updating Weight and Bias](#update)
# 
# * [Prediction](#predict)
# 
# * [Logistic Regression](#lr)
# 
# * [Interpreting the result via visualization](#visualization)
# 
# **[References](#references)**
# 

# <a id="get-familiar"></a> <br>
# ## Getting familiar with the dataset

# In[ ]:


# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data = pd.read_csv('../input/voice.csv')
data.info()


# ***IS THERE ANY MISSING FEATURE VALUE?***
# 
# We can see that, there is no missing data.

# In[ ]:


# Top 10 records in the dataset
data.head(10)


# ***TRANSLATE MALE-FEMALE OPTIONS TO COMPUTER'S LANGUAGE***
# 
# In order to make train,test, and prediction, we have to convert our binary male-female option into 0-1's. I choose male as '0' and female as '1'. 

# In[ ]:


data.label = [1 if each=='female' else 0 for each in data.label]
# Our y-axis(Outcome)
y = data.label.values
# Our features for prediction&training, x will include all of data except the outcome(label)
x_data = data.drop(["label"],axis=1)


#  <a id="normalization"></a> <br>
#  ## NORMALIZATION
# 
# Normalization have to applied to dataset, because some features can brake the balance with their proportion to other features like kurtosis and skew. These attributes should be re-scaled.

# In[ ]:


# Find the max&min value of the each column then apply the formula. This is a way to re-scaling
x = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
print(x.head().kurt)


# ## Logistic Regression Steps
# 
# Logistic Regression Algorithm's steps are represented below in computation graph way:
# 
# ![](https://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression_files/logistic_regression_schematic.png)
# *Source: [https://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression_files/logistic_regression_schematic.png](http://)*

#  <a id="traintest"></a> <br>
#  ## TRAIN TEST SPLIT
# 
# 20% of the data will be used for the test, rest of the data will be used for the training.

# In[ ]:


# Select Train-Test split randomly every-time
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T

# Features - Records
print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# <a id="initparam"></a> <br>
# ## Initializing Parameters 

# In[ ]:


# Initialize Parameters(dimensin = count of the features)
def initialize_weight_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b=0.0
    return w,b


# <a id="sigmoid"></a> <br>
# ## IMPLEMENTING SIGMOID FUNCTION
# 
# Sigmoid function is the one of the magical function from the mathland. It scales our data to probabilistic values. All of data will be represented in between 0 and 1 correctly. 
# After the implementation we can check the correctness the function f(0)=0.5. 
# Here is the graph and formula of the sigmoid function.
# 
# ![](https://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression_files/logistic_function.png)
# 
# *Figure1 : https://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression_files/logistic_function.png*

# In[ ]:


# Sigmoid function
# Calculation of z
# z = np.dot(w.T,x_train)+b
def sigmoid(z):
    return 1/(1+np.exp(-z))
# test
print(sigmoid(0))


# <a id="cost"></a> <br>
# ## Cost/Loss Function
# We want to minimize the cost, hence we will take derivative of the funvtion then, when the derivative equal to zero the root of the function gives to minimum value. 
# 
# ![](http://image.ibb.co/dFCR3H/6.jpg)
# Figure 3 : Loss function **(Cross Entropy Formula)**

# In[ ]:


# Forward and Backward Propogarion Combined
def ForwardBackwardP(w,b,x_train,y_train):
    #Forward Propogation
    z = np.dot(w.T,x_train)+b # Multiply with weight then sum each data
    y_head = sigmoid(z) # Scale the result into a probablistic value
    
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    # Cost is summation of all losses
    cost = (np.sum(loss))/x_train.shape[1] # x_train.shape[1] is count of the samples
    # Divided to sample size because of scaling
    
    # Backward Propogation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    # Save into Dictionary
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    return cost,gradients


# <a id="derivation"></a> <br>
# ## PURPOSE OF THE DERIVATION
# 
# ![](http://image.ibb.co/dAaYJH/7.jpg)
# Figure 4 : Purpose of the Derivation

# <a id="update"></a> <br> 
# # Update Learning Parameters (Weights and Bias)
# 
# Algorithm should update the weights and bias by number of iterations. While we doing this, we will use **learning_rate parameter** It is coefficient of derivatives and it is a *hyperparameter*. So it means we should tune value of the paramter by manually.
# 
# ![](http://image.ibb.co/hYTTJH/8.jpg)
# > Alpha value = learning_rate

# In[ ]:


# Updating Learning Parameters(wi,b)
def update(w,b,x_train,y_train,learning_rate,iterations):
    cost_list = []
    cost_list2 = []
    index = []
    
    # Updating learning parameters by number of iterations
    for i in range(iterations):
        # Make forward and backward propogation and find cost and gradients
        cost,gradients = ForwardBackwardP(w,b,x_train,y_train)
        cost_list.append(cost)
        #UPDATE
        w = w- learning_rate*gradients["derivative_weight"]
        b = b- learning_rate*gradients["derivative_bias"]
        
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i,cost))
            
        # Update learn parameters and bias
        parameters = {"weight":w,"bias":b}
        
    # Plot
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# <a id="predict"></a> <br>
# # Prediction

# In[ ]:


def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b) # Forward propogation for x_test
    Y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i]<=0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    return Y_prediction
    


# <a id="lr"></a> <br>
# ## Logistic Regression

# In[ ]:


def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,iterations):
    # Initialize 
    dimension = x_train.shape[0] # Feature size
    w,b = initialize_weight_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w,b,x_train,y_train,learning_rate,iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
    return y_prediction_test
    
y_pred_test = logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, iterations = 100)


# In[ ]:


# Top 20 prediction and real values
head_y_pred= ["Female" if each == 1 else "Male" for each in y_pred_test[0][:20]]
head_y_real = ["Female" if each == 1 else "Male" for each in y_test[:20]]
print(head_y_pred)
print(head_y_real)


# <a id="visualization"></a> <br>
# # Visualization
# ## Rate of the Male-Female Predictions

# In[ ]:


# Count of male-female
male_count = 0 
female_count = 0
for i in range(y_pred_test.shape[1]):
    if y_pred_test[0][i] == 0:
        male_count+=1
    else:
        female_count+=1

# Visualization
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

labels = ['Male','Female']
values = [male_count,female_count]

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='basic_pie_chart')


# ## Rate of male-female for real test data
# We can see accuracy of the our prediction

# In[ ]:


# Calculation of rate of raw test data
male_test=0
female_test=0
for i in y_test:
    if i == 0:
        male_test+=1
    else:
        female_test+=1

values = [male_test,female_test]
trace = go.Pie(labels=labels, values=values)
py.iplot([trace], filename='basic_pie_chart')


# <a id="references"></a> <br>
# # References

# *  References : This work is includes some parts of 'kanncaa1's deep-learning kernel. Thanks for sharing.*
# *  Source: https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners*
