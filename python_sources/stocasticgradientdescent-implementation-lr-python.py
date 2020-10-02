#!/usr/bin/env python
# coding: utf-8

# # Boston House price prediction using SGD

# In this kernel we will be implementing SGD on LinearRegression from scarch using python and we will be also comparing sklearn implementation SGD and our implemented SGD.

# In[ ]:


# Importing necessary libraries

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
boston = load_boston()


# In[ ]:


print(boston.data.shape)


# In[ ]:


print(boston.feature_names)


# In[ ]:


print(boston.target.shape)


# In[ ]:


print(boston.DESCR)


# In[ ]:


# Loading data into pandas dataframe
bos = pd.DataFrame(boston.data)
print(bos.head())


# In[ ]:


#noramlization for fast convergence to minima
#bos = (bos - bos.mean())/bos.std()
#bos.head()


# In[ ]:


bos['PRICE'] = boston.target

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']


# In[ ]:


# Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


X_train.mean()


# In[ ]:


# Standardization

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)


# In[ ]:


X_train


# In[ ]:


'''
# code source:https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

print('Coefficients: \n', lm.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# Explained variance score: 1 is perfect prediction
# print("R^2 score: %.2f"  % lm.score(X_test, Y_test))
print('Variance score: %.2f' % r2_score(Y_test, Y_pred))

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()
'''


# In[ ]:


from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
clf = SGDRegressor()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Coefficients: \n", clf.coef_)
print("Y_intercept", clf.intercept_)


# **Observations**
# 
# * Overall we can say the regression line not fits data perfectly but it is okay. But our goal is to find the line/plane that best fits our data means minimize the error i.e. mse should be close to 0.
# * MSE is 28.54 means the total loss(squared difference of true/actual target value and predicted target value). 0.0 is perfect i.e. no loss.
# * coefficient of determination tells about the goodness of fit of a model and here, r^2 is 0.70 which means regression prediction does not perfectly fit the data. An r^2 of 1 indicates that regression prediction perfect fit the data.

# # Stochastic Gradient Decent(SGD) for Linear Regression

# In[ ]:


# Imported necessary libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[ ]:


# Data loaded 
bostan = load_boston()


# In[ ]:


# Data shape
bostan.data.shape


# In[ ]:


# Feature name
bostan.feature_names


# In[ ]:


# This is y value i.e. target
bostan.target.shape


# In[ ]:


# Convert it into pandas dataframe
data = pd.DataFrame(bostan.data, columns = bostan.feature_names)
data.head()


# In[ ]:


# Statistical summary
data.describe()


# In[ ]:


#noramlization for fast convergence to minima
data = (data - data.mean())/data.std()
data.head()


# In[ ]:


data.mean()


# In[ ]:



#from sklearn.preprocessing import StandardScaler
#std = StandardScaler()
#data = std.fit_transform(data)
#data


# In[ ]:


# MEDV(median value is usually target), change it to price
data["PRICE"] = bostan.target
data.head()


# In[ ]:


# Target and features
Y = data["PRICE"]
X = data.drop("PRICE", axis = 1)


# In[ ]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[ ]:


#x_train = (x_train - x_train.mean())/ x_train.std()
#x_test = (x_test - x_train.mean())/ x_test.std()


# In[ ]:


#std = StandardScaler()
#x_train = std.fit_transform(x_train)
#x_test = std.fit_transform(x_test)


# In[ ]:


#x_train[0:,0:5]


# In[ ]:


x_train["PRICE"] = y_train
#x_test["PRICE"] = y_test


# In[ ]:


#x_train["PRICE"] = y_train
#x_test["PRICE"] = y_test


# In[ ]:


def cost_function(b, m, features, target):
    totalError = 0
    for i in range(0, len(features)):
        x = features
        y = target
        totalError += (y[:,i] - (np.dot(x[i] , m) + b)) ** 2
    return totalError / len(x)


# In[ ]:


# The total sum of squares (proportional to the variance of the data)i.e. ss_tot 
# The sum of squares of residuals, also called the residual sum of squares i.e. ss_res 
# the coefficient of determination i.e. r^2(r squared)
def r_sq_score(b, m, features, target):
    for i in range(0, len(features)):
        x = features
        y = target
        mean_y = np.mean(y)
        ss_tot = sum((y[:,i] - mean_y) ** 2)
        ss_res = sum(((y[:,i]) - (np.dot(x[i], m) + b)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    return r2


# In[ ]:


def gradient_decent(w0, b0, train_data, x_test, y_test, learning_rate):
    n_iter = 500
    partial_deriv_m = 0
    partial_deriv_b = 0
    cost_train = []
    cost_test = []
    for j in range(1, n_iter):
        
        # Train sample
        train_sample = train_data.sample(160)
        y = np.asmatrix(train_sample["PRICE"])
        x = np.asmatrix(train_sample.drop("PRICE", axis = 1))
        # Test sample
        #x_test["PRICE"] = [y_test]
        #test_data = x_test
        #test_sample = test_data.sample()
        #y_test = np.asmatrix(test_sample["PRICE"])
        #x_test = np.asmatrix(test_sample.drop("PRICE", axis = 1))
        
        for i in range(len(x)):
            partial_deriv_m += np.dot(-2*x[i].T , (y[:,i] - np.dot(x[i] , w0) + b0))
            partial_deriv_b += -2*(y[:,i] - (np.dot(x[i] , w0) + b0))
        
        w1 = w0 - learning_rate * partial_deriv_m 
        b1 = b0 - learning_rate * partial_deriv_b
        
        if (w0==w1).all():
            #print("W0 are\n", w0)
            #print("\nW1 are\n", w1)
            #print("\n X are\n", x)
            #print("\n y are\n", y)
            break
        else:
            w0 = w1
            b0 = b1
            learning_rate = learning_rate/2
       
            
        error_train = cost_function(b0, w0, x, y)
        cost_train.append(error_train)
        error_test = cost_function(b0, w0, np.asmatrix(x_test), np.asmatrix(y_test))
        cost_test.append(error_test)
        
        #print("After {0} iteration error = {1}".format(j, error_train))
        #print("After {0} iteration error = {1}".format(j, error_test))
        
    return w0, b0, cost_train, cost_test


# In[ ]:


# Run our model
learning_rate = 0.001
w0_random = np.random.rand(13)
w0 = np.asmatrix(w0_random).T
b0 = np.random.rand()

optimal_w, optimal_b, cost_train, cost_test = gradient_decent(w0, b0, x_train, x_test, y_test, learning_rate)
print("Coefficient: {} \n y_intercept: {}".format(optimal_w, optimal_b))

'''
error = cost_function(optimal_b, optimal_w, np.asmatrix(x_test), np.asmatrix(y_test))
print("Mean squared error:",error)
'''

plt.figure()
plt.plot(range(len(cost_train)), np.reshape(cost_train,[len(cost_train), 1]), label = "Train Cost")
plt.plot(range(len(cost_test)), np.reshape(cost_test, [len(cost_test), 1]), label = "Test Cost")
plt.title("Cost/loss per iteration")
plt.xlabel("Number of iterations")
plt.ylabel("Cost/Loss")
plt.legend()
plt.show()

#error = cost_function(optimal_b, optimal_w, np.asmatrix(x_test), np.asmatrix(y_test))
#print("Mean squared error: %.2f" % error)


# # Comparison between sklearn SGD and implemented SGD in python 

# In[ ]:


# Sklearn SGD
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % r2_score(Y_test, Y_pred))


# In[ ]:


# Implemented SGD
# The mean squared error
error = cost_function(optimal_b, optimal_w, np.asmatrix(x_test), np.asmatrix(y_test))
print("Mean squared error: %.2f" % (error))
# Explained variance score : 1 is perfect prediction
r_squared = r_sq_score(optimal_b, optimal_w, np.asmatrix(x_test), np.asmatrix(y_test))
print("Variance score: %.2f" % r_squared)


# In[ ]:


# Scatter plot of test vs predicted
# sklearn SGD
plt.figure(1)
plt.subplot(211)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: Sklearn SGD")
plt.show()

# Implemented SGD
plt.subplot(212)
plt.scatter([y_test], [(np.dot(np.asmatrix(x_test), optimal_w) + optimal_b)])
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: Implemented SGD")
plt.show()


# In[ ]:


# Distribution of error
delta_y_im = np.asmatrix(y_test) - (np.dot(np.asmatrix(x_test), optimal_w) + optimal_b)
delta_y_sk = Y_test - Y_pred
import seaborn as sns;
import numpy as np;
sns.set_style('whitegrid')
sns.kdeplot(np.asarray(delta_y_im)[0], label = "Implemented SGD", bw = 0.5)
sns.kdeplot(np.array(delta_y_sk), label = "Sklearn SGD", bw = 0.5)
plt.title("Distribution of error: $y_i$ - $\hat{y}_i$")
plt.xlabel("Error")
plt.ylabel("Density")
plt.legend()
plt.show()


# In[ ]:


# Distribution of predicted value
sns.set_style('whitegrid')
sns.kdeplot(np.array(np.dot(np.asmatrix(x_test), optimal_w) + optimal_b).T[0], label = "Implemented SGD")
sns.kdeplot(Y_pred, label = "Sklearn SGD")
plt.title("Distribution of prediction $\hat{y}_i$")
plt.xlabel("predicted values")
plt.ylabel("Density")
plt.show()


# **observations**
# * The mean squared error(mse) is quite high means the regression line does not fit the data properly. i.e. average squared difference between the actual target value and predicted target value is high. lower value is better.
# * r-squared score is 0.88, means the fit explain 88% of the total variation in the data about the average.
# * After looking at the error graph we can say +ve side of the graph, error is more.
# * By looking at the distribution of predicted value graph, It is clear that prediction of implemented SGD and sklearn SGD both are ovelapping(not perfectly) but the density of sklearn SGD is ~58% whereas implemented SGD is ~62% means the implemented SGD is predicting high but in actual it is not.

# **Conclusions**
# * While comparing scikit-learn implemented linear regression and explicitly implemented linear regression using optimization algorithm(sgd) in python we see there are not much differences between both of them.
# * Both of the model are not perfect but okay.
