#!/usr/bin/env python
# coding: utf-8

# # Understanding the Gradient Descent Algorithm

# This sole purpose of this notebook is to show how `Gradient Descent Algorithm` works, also it serves as a code implementation for the article `Understanding Gradient Descent Algorithm through Golf`. If you have not checked the article, click [here](https://www.linkedin.com/pulse/understanding-gradient-descent-algorithm-through-golf-amit-mittal/), I have explained the concept in layman's language using an example of Golf.

# ## Content
# 1. Importing Libraries and Loading Dataset
# 2. Feature Scaling
# 3. Defining Gradient Descent Function
# 4. Visualizing the Results
# 5. Conclusion

# ## 1. Importing Libraries and Loading Dataset

# In[ ]:


#importing libraries which will be required in the process
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#supress warnings
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#loading dataset using pandas and showing first five rows
df = pd.read_csv('/kaggle/input/advertising-data/Advertising.csv')
df.head()


# For the learning purpose, we will consider only one independent variable `TV`. It will also help in visualizing the results easily. Reason to consider `TV` as independent variable is because it has very good positive correlation of `0.90` with dependent varaiable `Sales`.

# In[ ]:


# X -> Independent variable, which is TV
X = df.iloc[:,1:2].values

# y -> Dependent varible, which is Sales
y = df.iloc[:,-1:].values


# In[ ]:


#Now we will see a scatter plot to show relation between independent and dependent varaiable
plt.scatter(X,y,alpha=0.5)
plt.xlabel('TV',size=10)
plt.ylabel('Sales',size=10)
plt.title('Scatter Plot of TV vs Sales',size=15)
plt.show()


# ## 2. Feature Scaling

# Here,I have defined a function `FeatureScaling` which maps the value of numeric data in a column between -1 to 1, this is also called as `Standard Scalar`.
# 
# The purpose of scaling the columns (variables) is to have values of every column in same range. It helps in converging the Gradient Descent Algorithm faster also. 

# In[ ]:


def FeatureScaling(X):
    """
    is function takes an array as an input, which needs to be scaled down.
    Apply Standardization technique to it and scale down the features with mean = 0 and standard deviation = 1
    
    Input <- 2 dimensional numpy array
    Returns -> Numpy array after applying Feature Scaling
    """
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i]-mean[i])/std[i]

    return X


# In[ ]:


# scaling variable TV using defined function
X = FeatureScaling(X)


# Since, I will be defining function to perform gradient descent, so we will be required to add a column of `1's` in X.

# In[ ]:


print("X before adding column of 1's:",X[:2],sep="\n")
X = np.append(arr=np.ones((X.shape[0],1)),values=X,axis=1)
print("\nX after adding column of 1's:",X[:2],sep="\n")


# ## 3. Defining Gradient Descent Function

# Now, I have defined function `ComputeCost` which computes the Mean Squared Errors, which will be optimized by gradient descent algorithm.

# In[ ]:


#ComputeCost function determines the cost (sum of squared errors) 

def ComputeCost(X,y,theta):
    """
    This function takes three inputs and uses the Cost Function to determine the cost (basically error of prediction vs
    actual values)
    Cost Function: Sum of square of error in predicted values divided by number of data points in the set
    J = 1/(2*m) *  Summation(Square(Predicted values - Actual values))
    
    Input <- Take three numoy array X,y and theta
    Return -> The cost calculated from the Cost Function
    """
    m=X.shape[0] #number of data points in the set
    J = (1/(2*m)) * np.sum((X.dot(theta) - y)**2)
    return J


# In[ ]:


#Gradient Descent Algorithm to minimize the Cost and find best parameters in order to get best line for our dataset

def GradientDescent_New(X,y,theta,alpha,no_of_iters):
    """
    Gradient Descent Algorithm to minimize the Cost
    
    Input <- X, y and theta are numpy arrays
            X -> Independent Variables/ Features
            y -> Dependent/ Target Variable
            theta -> Parameters 
            alpha -> Learning Rate i.e. size of each steps we take
            no_of_iters -> Number of iterations we want to perform
        Return -> theta (numpy array) which are the best parameters for our dataset to fit a linear line
             and Cost Computed (numpy array) for each iteration
    """
    m=X.shape[0]
    J_Cost = []
    theta_array = []
    for i in range(no_of_iters):
        error = np.dot(X.transpose(),(X.dot(theta)-y))
        theta = theta - alpha * (1/m) * error
        J_Cost.append(ComputeCost(X,y,theta))
        
        #below code is to note theta value of every 30th iteration, which we will be using further in this notebook
        if (i+1)%30 == 0:
            theta_array.append(theta)
    
    return theta, np.array(J_Cost), np.array(theta_array)


# Now, we will use the above function to minimize the `Mean Squared Error`and find the optimal value of theta's to get the best line.

# In[ ]:


#number of iterations
iters = 300

#learning rate
alpha = 0.01

#initializing theta
theta = np.zeros((X.shape[1],1))

#finally computing values using function
theta, J_Costs, theta_array = GradientDescent_New(X,y,theta,alpha,iters)


# ## 4. Visualizing the Results

# Now we will visualize, how `Gradient Descent Algorithm` minimized the error.

# In[ ]:


plt.figure(figsize=(8,5))
plt.plot(J_Costs,color="g")
plt.title('Convergence of Gradient Descent Algorithm')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.show()


# We can clearly see from above grap that, gradient descent algorithm converged after somewhere 275 iterations.

# In[ ]:


# Removing column of 1's from X in order to visualize the data.
X = X[:,1:]


# Now we will visualize, how gradient descent algoroithm helps to find the best line which describes linear relationship between how money spent on `TV` advertisements and impact `Sales` (by minimizing errors).
# Since, it is not possible to show how theta changes after every iteration. Here, I have shown changes after every 30th iteration.
# 
# **We can see that how `intercept` and `slope` changes after every 30th iteration.**

# In[ ]:


for i in range(10):
    plt.figure(figsize=(40,10))
    plt.subplot(2,5,i+1)
    b0, b1 = round(float(theta_array[i,0]),2), round(float(theta_array[i,1]),2)
    y_pred = b0 + b1 * X
    mse = round(J_Costs[30*i+30-1],2)
    plt.scatter(X,y,alpha=0.5)
    plt.plot(X,y_pred,color="r")
    plt.xlabel('TV',size=10)
    plt.ylabel('Sales',size=10)
    plt.title('Sales = {} + {} * TV (after {} iterations, MSE: {})'.format(b0,b1,30*i+30,mse),size=14)
    plt.show()


# ## 5. Conclusion

# As you we can see above, how `Gradient Descent Algorithm` helped to minimize the `Mean Squared Error` and helped to find the Best line which can help in prediciting `Sales` for a company, if we are given how mich they spend on `TV` advertisements.
# 
# This notebook delivered, how the algorithm works in attaining the best fit line for the dataset. Gradient Descent is algorithm is very popular algorithm used in Machine Learning model to minimize the errors. You can read about it more from other sourses in order to learn better.
# 
# I hope you liked the idea of explaining the concept, if any feedback, please type in comment sections.
# Still if you haven't read the article, click [here](https://www.linkedin.com/pulse/understanding-gradient-descent-algorithm-through-golf-amit-mittal/).
# 
# Many Thanks :)

# ### End Note:
# I thank, Andrew N.G. for his Machine Learning course on Coursera, which helped me to write those functions
