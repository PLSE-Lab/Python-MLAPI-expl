#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## FuelConsumption.csv:
# We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)
# 
# - **MODELYEAR** e.g. 2014
# - **MAKE** e.g. Acura
# - **MODEL** e.g. ILX
# - **VEHICLE CLASS** e.g. SUV
# - **ENGINE SIZE** e.g. 4.7
# - **CYLINDERS** e.g 6
# - **TRANSMISSION** e.g. A6
# - **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# - **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# - **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# - **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0
# 

# In[ ]:


df = pd.read_csv("/kaggle/input/fuelconsumption/FuelConsumptionCo2.csv")

# take a look at the dataset
df.head()


# In[ ]:


# summarize the data
df.describe()


# In[ ]:


cdf = df[['ENGINESIZE', 'CYLINDERS','TRANSMISSION', 'FUELTYPE', 'CO2EMISSIONS']]
cdf.head(9)


# In[ ]:


sns.relplot(x="ENGINESIZE", y="CO2EMISSIONS", data=cdf);


# In[ ]:


sns.relplot(x="ENGINESIZE", y="CO2EMISSIONS", kind="line", ci="sd", data=cdf);


# In[ ]:


sns.jointplot(x=cdf['ENGINESIZE'], y=cdf['CO2EMISSIONS'], kind="hex");


# In[ ]:


sns.relplot(x="CYLINDERS", y="CO2EMISSIONS", data=cdf);


# In[ ]:


sns.relplot(x="CYLINDERS", y="CO2EMISSIONS", kind="line", ci="sd", data=cdf);


# In[ ]:


sns.jointplot(x=cdf['CYLINDERS'], y=cdf['CO2EMISSIONS'], kind="hex");


# In[ ]:


fig, ax =plt.subplots(1,2, figsize = (19, 7))
sns.countplot(x="TRANSMISSION", data=cdf, ax=ax[0])
sns.catplot(y="TRANSMISSION", x="CO2EMISSIONS", data=cdf, ax=ax[1]);
fig.show()


# In[ ]:


fig, ax =plt.subplots(1,2, figsize = (19, 7))
sns.countplot(x="FUELTYPE", data=cdf, ax=ax[0])
sns.catplot(y="FUELTYPE", x="CO2EMISSIONS", data=cdf, ax=ax[1]);
fig.show()


# In[ ]:


data_train_x = ['FUELTYPE','TRANSMISSION', 'CYLINDERS', 'ENGINESIZE']

data_train = pd.get_dummies(cdf[data_train_x])


# In[ ]:


cols_in_train = data_train.columns.tolist()
len(cols_in_train)


# In[ ]:


data_train.sample(5)


# We are not normalizing because the values do matter.
# https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning
# 

# In[ ]:


#setting the matrixes
X = data_train.iloc[:].values


# In[ ]:


type(X)


# In[ ]:


noOfTrainEx = X.shape[0] # no of training examples
print("noOfTrainEx: ",noOfTrainEx)
noOfWeights = X.shape[1]+1 # no of features+1 => weights
print("noOfWeights: ", noOfWeights)


# In[ ]:


ones = np.ones([noOfTrainEx, 1]) # create a array containing only ones 
X = np.concatenate([ones, X],1) # cocatenate the ones to X matrix
theta = np.ones((1, noOfWeights)) #np.array([[1.0, 1.0]])


# In[ ]:


y = cdf['CO2EMISSIONS'].values.reshape(-1,1) # create the y matrix


# In[ ]:


print(X.shape)
print(theta.shape)
print(y.shape)


# In[ ]:


#set hyper parameters
alpha = 0.01
iters = 1000


# In[ ]:


## Creating cost function
def computeCost(X, y, theta):
    h = X @ theta.T
    error = h-y
    loss = np.power(error, 2) 
    J = np.sum(loss)/(2*noOfTrainEx)
    return J


# In[ ]:


computeCost(X, y, theta) #Computing cost now produces very high cost


# In[ ]:


## Gradient Descent funtion
def gradientDescent(X, y, theta, alpha, iters):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost[i] = computeCost(X, y, theta)
        if i % 100 == 0: # just look at cost every ten loops for debugging
            print(i, 'iteration, cost:', cost[i])
    return (theta, cost)


# In[ ]:


g, cost = gradientDescent(X, y, theta, alpha, iters)  


# In[ ]:


print(g)


# In[ ]:


print(cost)


# In[ ]:


#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')


# In[ ]:


axes = sns.scatterplot(x = "ENGINESIZE", y = "CO2EMISSIONS", data = cdf, ci = False)
x_vals = np.array(axes.get_xlim()) 
y_vals = g[0][0] + g[0][1]* x_vals #the line equation
plt.plot(x_vals, y_vals, '--')


# In[ ]:


axes = sns.scatterplot(x = "CYLINDERS", y = "CO2EMISSIONS", data = cdf, ci = False)
x_vals = np.array(axes.get_xlim()) 
y_vals = g[0][0] + g[0][1]* x_vals #the line equation
plt.plot(x_vals, y_vals, '--')


# ## Using sklearn

# In[ ]:


data_train.columns.to_list()


# In[ ]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(data_train[data_train.columns.to_list()])
y = np.asanyarray(cdf[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Intercept: ',regr.intercept_)
print ('Coefficients: ', regr.coef_)


# ### Evaluation

# In[ ]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(data_train[data_train.columns.to_list()])
test_y = np.asanyarray(cdf[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

