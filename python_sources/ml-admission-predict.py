import math
from matplotlib import pyplot
import numpy as np
import pandas as pd

##########################Import data##########################################

file = input("input file location: \t")
data = pd.read_csv(file)

print(data.columns)

features_house = ['X1','GRE Score','TOEFL Score','University Rating','SOP', 'LOR ','CGPA']

X = data[features_house]
print(X.describe())

y = data['Chance of Admit ']
print(y.describe())

theta = np.zeros(7 , dtype = float)
alpha = 0.001
iterations = 148
m = len(y)

def computecost(X,y,theta):
    m = len(y)
    j1 = ((np.matmul(X,theta) - y)**2)
    j2 = 0
    for i in range (0,m):
        j2 = j2 + j1[i]
    j3 = j2/(2*m)
    return(j3)

J = computecost(X,y,theta)

def gradientdescent(X,y,theta,alpha,iterations):
    m = len(y)
    cost = [0.0 for i in range(iterations)]
    for i in range(iterations):
        error = (np.matmul(X,theta)-y)
        error_derivative = np.sum(np.matmul(error,X), axis=0)
        theta = theta - (alpha / m) * error_derivative
        cost[i] = computecost(X, y, theta)
    return theta, cost

theta, cost = gradientdescent(X, y, theta,alpha, 149)
print('Theta: ', theta)
print('Cost: ', cost[-1])

