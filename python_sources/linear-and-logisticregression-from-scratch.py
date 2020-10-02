# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:13:42 2020

@author: Sourav


Here I have implemented linear regression and logistic regression from scratch. I have written this script
mainly for beginners who are facing trouble to write the codes of algrothims from scratch.These two algorithm are
one of the basic algorithms all the data scientists starts with. I have implemented these two algorithm 
using object oriented programming in python. All the details of different class, function and inline comment 
has been added. Still if you face any problem please feel free to contact me. 
I will be trying to implement more algorithm from scratch. please upvote if you like.
Happy learning...
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
   
     
    def __init__(self, X, y, W = None, b = None, alpha = 0.1, n_iteration = 100):     
        
        """
        Linear Regression machine learning class
        
        param X : Independent features of the data.
        param y : Target variable of the data. 
        param w:  Slope  of the regression line for each features
        param b: intercept of the regression line
        param alpha: Learning rate
        param n_iteration: Number of iterations to run the optimization step
        
        This is regression class. This algorithm will help users to predict a
        linear regression problem
        
        
        """
        np.random.seed(0) 
        self.X = X
        self.y = y
        b = np.random.randint(1) * 1000
        W = np.random.randint(1,X.shape[1]) *20000
        self.W = W
        self.b = b
        self.alpha = alpha
        self.n_iteration = n_iteration
        
    
        
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : Independent features of the data.
        y : Target variable of the data.

        Returns
        -------
        plot the number of iteration vs cost function
        Return the co-efficient to predict the test set.

        """
        cost_function = []
        for i in range(self.n_iteration):
            
            # predicting the value using the formula W*X + b
            yhat = np.dot(self.W, self.X.T) + self.b  
            #calculating the cost
            cost_function.append((1/2*self.X.shape[0]) * np.sum((yhat - self.y.T) **2) ) 
            ## calculating the error part of gradient descent 
            error_ = np.sum(np.multiply((yhat - self.y.T), self.X.T),axis = 1)
            # updating the intercept parameter
            self.b = self.b - ((self.alpha/self.X.shape[0]) * np.sum(yhat - self.y.T)) 
            # updating the slope parameters 
            self.W = self.W- ((self.alpha/self.X.shape[0]) * error_.T)  
            
            
        plt.plot(cost_function)
        plt.xlabel("Number of Iteration")
        plt.ylabel("Cost_Functon")
        plt.title("Cost_function per Iteration")
        return self.W,self.b,error_
    
    def predict(self, X):
        
        """
        Parameters
        ----------
        X : Independent features of the target data.
        
        Returns
        -------
        Return the predicted values.

        """
        #predicting for the test data
        predict = np.dot(self.W,X.T) + self.b
        
        return predict
        
 
    
 
    
 
        
class LogisticRegression:
   
     
    def __init__(self, X, y, W = None, b = None, alpha = 0.1, n_iteration = 100):     
        
        """
        Logistic Regression machine learning class
        
        param X : Independent features of the data.
        param y : Target variable of the data. 
        param w:  Slope  of the regression line for each features
        param b: intercept of the regression line
        param alpha: Learning rate
        param n_iteration: Number of iterations to run the optimization step
        
        This is regression class. This algorithm will help users to predict a
        linear regression problem
        
        
        """
        np.random.seed(0)
        self.X = X
        self.y = y
        b = np.random.randint(1) 
        W = np.random.randint(1,X.shape[1])
        self.W = W
        self.b = b
        self.alpha = alpha
        self.n_iteration = n_iteration
        
        
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : Independent features of the data.
        y : Target variable of the data.

        Returns
        -------
        plot number of iteration vs error grpah
        Return the co-efficient to predict the test set.

        """
        cost_function = []
        
        for i in range(self.n_iteration):
            
            # predicting the value using the sigmoid  of the formula W*X + b
            yhat = (np.dot(self.W, self.X.T) + self.b) 
            yhat = 1/(1 + np.exp(-yhat))
            #calculating the cost function
            cost_function.append((np.sum(np.multiply(y.T,np.log(yhat)) + np.multiply((1-y).T,np.log(1 - yhat))))/(self.X.shape[0]))
            ## calculating the error part of graient descent 
            error_ = np.sum(np.multiply((yhat - self.y.T), self.X.T),axis = 1) 
             # updating the intercep parameter
            self.b = self.b - ((self.alpha/self.X.shape[0]) * np.sum(yhat - self.y.T)) 
            # updating the slope parameters 
            self.W = self.W- ((self.alpha/self.X.shape[0]) * error_.T)  
            
            
        plt.plot(cost_function)
        plt.xlabel("Number of Iteration")
        plt.ylabel("Cost_Functon")
        plt.title("Cost_function per Iteration")
        return self.W,self.b,error_
    
    def predict(self, X):
        
        """
        Parameters
        ----------
        X : Independent features of the target data.
        
        Returns
        -------
        Return the predicted values.

        """
        #predicting for the test data
        predict = np.dot(self.W,X.T) + self.b
        predict = 1/(1 + np.exp(-predict))
        
        #if the probabilit of prediction is greater than 0.5 we will make it 1.
        for j,i in enumerate(predict):
           if (i<.5):
               predict[j]=0
           else:
               predict[j]=1
        
        return predict