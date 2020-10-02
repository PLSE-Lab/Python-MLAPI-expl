#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **This is a kernel I am doing to practice Linear Regression in one variable. Goal is to find a relationship between Water Salinity and Water Temperature based on the calcofi dataset(bottle.csv)**

# In[ ]:


data = pd.read_csv("../input/calcofi/bottle.csv")
data.head


# Looking at the dataset, our sole purpose is to find relationship between water temperature and salinity so looking at water salinity values we have quite a lot of rows with NaN values and since most of the values are around 33 we fill those NaN values with the mean of Salinity.

# In[ ]:


data["Salnty"].head
data["Salnty"].isna().sum()
mean_salinity = np.mean(data["Salnty"])
print(mean_salinity)
Salinity_Independent_X = data["Salnty"].fillna(mean_salinity)


# We also check for any NaN values in the Temperature column and see what we can do to fill those NaN values up.

# In[ ]:


data["T_degC"].head
max_temp = data["T_degC"].max()
min_temp = data["T_degC"].min()
mean_temp = np.mean(data["Salnty"])
temperature_dependent_y = data["T_degC"].fillna(random.uniform(min_temp, max_temp))
#we are also going to fill up those NaN values with random values 
#ranging between min and max which is 1.44 and 31.14 since there are a lot of NaN values and corresponding salinity values are all around 33.


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.scatter(Salinity_Independent_X, temperature_dependent_y, c="#ef5423", label= "Scatter Plot")
plt.xlabel("Salinity")
plt.ylabel("Temperature")
plt.legend()
plt.show()


# I am going to just use the ordinary least squares method to find the theta0 and theta1 values. These are the values that provide with a best fit for the given data.

# In[ ]:


numerator = 0
denominator = 0
m = data["Salnty"].size

x = Salinity_Independent_X
y = temperature_dependent_y

x_mean = np.mean(x)
y_mean = np.mean(y)

for i in range(m):
    numerator += (x[i] - x_mean)* (y[i] - y_mean)
    denominator += (x[i] - x_mean)**2

theta1 = numerator / denominator
theta0 = y_mean - (theta1 * x_mean)

print(theta1, theta0)


# From using this method of ordinary least squares, we find that theta1 value is -4.613 which determines that there is a negative correlation between the 2 variables.
# 
# Now we will display a few diagnostic plots that I learned while doing Rachel's Regression Challenge Day 2.
# Want more info on the uses and significance of the diagnostic plots. Read this https://www.kaggle.com/rtatman/regression-challenge-day-2

# In[ ]:


import statsmodels.api as sm
import statsmodels.graphics
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot

#Using the Gaussian family we see that we are fitting a Simple Linear Regression Model.
model = sm.GLM(temperature_dependent_y, Salinity_Independent_X, family=sm.families.Gaussian()).fit()

sns.regplot(Salinity_Independent_X, model.resid_deviance, fit_reg=False)
plt.title('Residual plot')
plt.xlabel('Salinity_Independent_x')
plt.ylabel('Residuals');

#From the below plot we can see that the residual plot is like a shapeless cloud so this diagnostic plot is quite correct as we want a normally distributed
#cloud of values from this plot.


# In[ ]:


# statsmodels Q-Q plot on model residuals
QQ = ProbPlot(model.resid_deviance)
fig = QQ.qqplot(alpha=0.5, markersize=5);
#From this plot we see that we have a diagonal line from bottom to top right with a high skew at the bottom which is what we want. This is how 
#we want the plot to be more or less.


# In[ ]:


# get data relating to high leverage points using statsmodels

# fit OLS regression model 
model_g = sm.OLS(temperature_dependent_y, Salinity_Independent_X).fit()

# leverage, from statsmodels
model_leverage = model_g.get_influence().hat_matrix_diag
# cook's distance, from statsmodels
model_cooks = model_g.get_influence().cooks_distance[0]

# plot cook's distance vs high leverage points
sns.regplot(model_leverage, model_cooks, fit_reg=False)
plt.xlim(xmin=-0.005, xmax=0.02)
plt.xlabel('Leverage')
plt.ylabel("Cook's distance")
plt.title("Cook's vs Leverage");


# We see that all the data points are highly clustered around the 0 side which is how we want it to be.

# We now use the Sklearn Linear Regression and use train test split to train the model and then see how it works on the test set.

# In[ ]:


from sklearn.linear_model import LinearRegression 
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')

X = Salinity_Independent_X.values.reshape(-1,1)
y = temperature_dependent_y.values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

#The regressor.coef is our theta1 value where as regressor.intercept value is our theta0 value from our ordinary least squares method.


# In[ ]:


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)


# In[ ]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


Many of the predicted values are close to the actual value which means that our model is quite good in predicting 
the temperature based on salinity values. Now we will plot a regression line. 


# In[ ]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# Now we calculate some metrics from the fitted model

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# So we come to an end to this kernel. My first Kernel after reading and practicing by myself with some common datasets. I went through Rachel Taetman's Datasets for Regression Analysis and thought I would practice on this and write my first Kernel of Linear Regression.
# 
# I would love everyone to give me feedback on what all I could have done with this dataset and how I could improve on while practicing data science. Thank you.
