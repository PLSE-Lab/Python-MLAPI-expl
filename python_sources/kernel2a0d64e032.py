#!/usr/bin/env python
# coding: utf-8

# In[69]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
# Reading Data
insurance_data = pd.read_csv("../input/insurance.csv")
#view the entire data 
#print(insurance_data)
# view the first 5 rows 
#print(insurance_data.head())
# view the first 5 rows 
#print(insurance_data.tail())
# Collecting information about X and Y
#X = insurance_data["X"].values 
#Y = insurance_data["Y"].values
#lets display the number of rows
#Num_of_rows = len(X)
#print("Number of rows and columns = " , insurance_data.shape)
# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
# Using the formula to calculate slope and intercept
numerator = 0
denominator = 0
for i in range(Num_of_rows):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2
slope = numerator / denominator
intercept = mean_y - (slope * mean_x)
print("slope = " , round(slope,2) ,"intercept = " , round(intercept,2))

# Plotting Values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = intercept + slope * x
# Ploting Line
'''plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rgression Line & Scatter Plot Combined')
plt.legend()
plt.show()'''
# Calculating Root Mean Squares Error
rmse = 0
for i in range(Num_of_rows):
    y_pred = intercept + slope * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/Num_of_rows)
print("RMSE = ", rmse)
# Calculating R2 Score
ss_tot = 0
ss_res = 0
for i in range(Num_of_rows
    y_pred = intercept + slope * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score = " , r2)

