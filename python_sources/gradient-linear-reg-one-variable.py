# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:22:11 2019

@author: perei
"""



##### importing libraries....

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

##### importing dataset.....

data = pd.read_csv("../input/Salary_Data.csv")
df = pd.DataFrame(data)
x = df['YearsExperience']
y = df['Salary']
n = len(df)

##### split dataset for train and test

x_train = df['YearsExperience'][:20]
y_train = df['Salary'][:20]
x_test = df['YearsExperience'][21:]
y_test = df['Salary'][21:]
n_train = len(x_train)

#### parameters tuning by gradient descent

t0 = 0.0        #### intial parameter c
t1 = 0.0        #### intial parameter m
L = 0.01       #### learning rate
simult = 1000   #### simulations

simul = []
h = 0
for h in range(simult):
    simul.append(h)
    h+=1

##### histroy of cost function and parameters past values
    
cost_fn_histroy = []
tt_0 = [] 
tt_1 = []

#### gradient descent process

for z in range(simult):
    tt_0.append(t0)
    tt_1.append(t1)
    y_pred = t0 + t1*x_train
    y_error = y_pred - y_train
    y_sqrd_er = np.power(y_error, 2)
    cost_fn_histroy.append(np.sum(y_sqrd_er)/n)    #### cost function      
    y_grad_t1 = x_train*(y_error)
    t0 = t0 - L*((2/n_train)*(sum(y_error)))       ##### updating t0           
    t1 = t1 - L*((2/n_train)*(sum(y_grad_t1)))     #### updating t1    

#### creating new dataframe for the simulation histroy for plotting
       
df_1 = pd.DataFrame(cost_fn_histroy)
df_1 = df_1.rename(columns = {0:'x'})

id = []                              #### creating id to retrive data
for i in range(len(df_1)):
    id.append(i)
df_1['id'] = id

df_1['t0'] = tt_0
df_1['t1'] = tt_1

df_1 = df_1[['id', 'x', 't0', 't1']]

##### retriving minimized cost function parameter


def get_par():    
    for a,b in zip(df_1['x'], df_1['id']):
        if ( a == df_1['x'].min()):
            pos = b
            t0_val = df_1['t0'][pos]
            t1_val = df_1['t1'][pos]
            return(t0_val, t1_val)
            
final_par = get_par()

c = final_par[0]
m= final_par[1]

y_test_pred = m*x_test + c           #### predicting y_test value
y_error_h = y_test_pred - y_test     
 
##### evaluation metrics

r2_scores = metrics.r2_score(y_true = y_test, y_pred = y_test_pred)    ### r^2 squared score 
mae =  metrics.r2_score(y_true = y_test, y_pred = y_test_pred)         ### Mean Absolute error



#### summary_report

print("The optimized parameter for the 34% of test values are m = {} and c = {}".format(m, c))
print("r^2 value for the model is = {}".format(r2_scores)) 
print("mean absolute error for the model is = {}".format(mae))
print("cost function value reduced from {} to {}".format(max(cost_fn_histroy),min(cost_fn_histroy) ))
##### plots

plt.plot(simul, cost_fn_histroy)  #### cost_fn vs simulations      
plt.title("Simulations vs Cost function values")
plt.show()
   
plt.scatter(x_test, y_test)      #### checking how well regression line is fitted to y_test
plt.plot(x_test, y_test_pred)
plt.show() 
    
plt.plot(x_test, y_test)         #### checking y_test true value with y_test_predicted regression
plt.plot(x_test, y_test_pred)
plt.title("Actual test values vs predicted test values")
plt.show()  

