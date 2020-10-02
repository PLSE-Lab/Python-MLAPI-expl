#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kagglesdsdata/datasets/559075/1016976/IMG_8969.jpg?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584660734&Signature=ss9vyarP4QyHPZW6flEkEb4zMgkvOwUGgIAwaEcJEUi85iiyolTL0xPwNpfjgK300yb9hOqwdSN8R%2FpmBDwhRxvL2yABHvs8uE%2FJwYxKjzymxsHvX8vLVteC1UBMJd%2FHCWx%2BKzFCLxb86GpXxiaL%2FtYXDOXX3JC3n21byYwc%2Fjv4QFk15Vk%2BM96vYy694hixm36XlkNobLjwLMdIH3qelUnebHqwtyyuAlnd5QVWZa4GvGGjfkWBPWIsIapf%2Ffuo%2FWjuYp7DJvcm9RkLALA0xEAsqS%2FeHqKl82FqxCB%2Ff18CviPuWVGMZnd5C9BLKUwHVfZGN161TL7yNJe%2BEtXDpg%3D%3D)
# 
# 
# 
# $
# \begin{bmatrix}
#   F \\
#   M_x 
#  \end{bmatrix} = 
#  \begin{bmatrix}
#  1 & 1 & 1 & 1\\ 
#   -2 & -1 & 1 & 2 
#  \end{bmatrix}
#  \begin{bmatrix}
#   f_1 \\ f_2 \\ f_3 \\ f_4
#  \end{bmatrix}\\
# min \sum{F}\\
# min \sum{M_x}\\ 
# f_{min} \leq f_i \leq f_{max}\\
# F=10 \\
# M_x=2$
# 
# $Find \ f_i$

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

"""
Q1, Q2, Q3:
formatting the convex optimization problem:
Af = b
f_min <= f_i <= f_max
minimize z = ||f||_p
We could always use regressive approach to obtain the global optimum because of the nature of convex
BUT here we use the CVXPY library
"""
# Construct the problem.
n = 4
x=0
aux=np.zeros(n)
for i in range(-n // 2,0,1):
    aux[x] = i
    x+=1
for i in range(1,n // 2 + 1,1):
    aux[x] = i
    x+=1    
A = np.array([np.ones(n), aux])
b = np.array([10, 2])
# number of rotors



##############################################################################################
##############################################################################################
from scipy.optimize import minimize

#objetctive function
def objective(fi):
    return np.linalg.norm(fi,1)

#contraint
def constraint1(fi):
    return A.dot(fi.transpose())-b

con2 = {'type': 'eq', 'fun': constraint1}

# initial guesses
f0 = np.ones(n)

bon = (0,60000)
for i in range(0,n-1,1):
    bnds = np.vstack((bnds, bon)) 

solution = minimize(objective,f0, constraints=con2)
x_norm1 = solution.x

plt.bar(A[1], x_norm1,width=0.2)
plt.grid(True)


# In[ ]:



from scipy.optimize import least_squares

A1 = np.array([[1,1,1,1], [-2,-1,1,2]])
b1 = np.array([10, 2])

#objetctive function
def objective(fi):
    return (fi[0]+fi[3])+fi[1]+fi[2]

#contraint
def constraint1(fi):
    return A1.dot(fi.transpose())-b1

con21 = {'type': 'eq', 'fun': constraint1}

# initial guesses
f01 = np.ones(4)

bo = (0,60000)
bnds1 = (bo,bo,bo,bo)

solution = minimize(objective,f01,bounds=bnds1, constraints=con21)
x_least_squares = solution.x

plt.bar(A1[1], x_least_squares,width=0.2)
plt.grid(True)


# In[ ]:


#objetctive function
def objective(fi):
    return np.linalg.norm(fi,2)

#contraint
def constraint1(fi):
    return A.dot(fi.transpose())-b

con2 = {'type': 'eq', 'fun': constraint1}

# initial guesses
f0 = np.ones(n)

bon = (0,60000)
for i in range(0,n-1):
    bnds = np.vstack((bnds, bon)) 
    
solution = minimize(objective,f0, constraints=con2)
x_norm2 = solution.x

plt.bar(A[1], x_norm2,width=0.2)
plt.grid(True)


# In[ ]:


#objetctive function
def objective(fi):
    return np.linalg.norm(fi,np.inf)

solution = minimize(objective,f0, constraints=con2)
x_norm_infinit= solution.x

plt.bar(A[1], x_norm_infinit,width=0.2)
plt.grid(True)

