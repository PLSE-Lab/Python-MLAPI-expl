#!/usr/bin/env python
# coding: utf-8

# # Applied Statistics : Assignment 1
# #### Omkar Nitin Pawar
# #### A20448802

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import random


# # Question 4
# 
# Generate the data according to given conditions
# 
# 

# In[ ]:


x = []
for i in range(0,40):
    x.append(random.uniform(-1,1)) #X in range -1 to 1
    
s = np.random.normal(0,0.1,40) # Normally distributed E


# In[ ]:


df = pd.DataFrame(data = x, columns = ["Xi"])
df["E"] = s


# Calculate the value of y as y = 2x + E

# In[ ]:


df["Yi"] = 2*df["Xi"]+df["E"]


# In[ ]:


df.head()


# # Regression Through Origin 
# Estimate of B when it is regression through origin

# In[ ]:


bRTO = sum(df["Xi"]*df["Yi"]) / sum(df["Xi"]**2)
bRTO


# Predict the value of Y using regression through origin

# In[ ]:


df["YiOrigin"] = (bRTO*df["Xi"])


# In[ ]:


df.head()


# # Ordinary Linear Regression

# Now, we calculate the values of normal regression line parameters (b0 and b1)

# In[ ]:


x_mean = df["Xi"].mean()
y_mean = df["Yi"].mean()


# In[ ]:


sum((df["Xi"] - x_mean)**2)


# In[ ]:


b1 = sum((df["Xi"] - x_mean)*(df["Yi"] - y_mean)) / sum((df["Xi"] - x_mean)**2) # Value of regression coefficient(Slope)
b1


# In[ ]:


b0 = y_mean - (b1 * x_mean) # Value of intercept
b0


# Predict the value of Y by using ordinary linear regression

# In[ ]:


df["YiHat"] = b0 + (b1*df["Xi"])


# In[ ]:


df.head()


# # Value of ei
# Here, we find the value of error(ei) for both, regression through origin and ordinary linear regression

# In[ ]:


df["Ei Origin"] = df["Yi"] - df["YiOrigin"]
print("Value of ei for regression through origin =",sum(df["Ei Origin"]))


# In[ ]:


df["Ei Hat"] = df["Yi"] - df["YiHat"]
print("Value of ei for ordinary linear regression =",sum(df["Ei Hat"]))


# #### Comparing both the above values, it can be seen that e value is very small ordinary linear regression than regression through origin. Hence, it can be inferred that ordinary regression line is the best fit than the line passing through origin.

# In[ ]:


df.head()


# ### R - squared values
# Calculate the value of r squared for regresion through origin
# 

# In[ ]:


n = len(df["Xi"])
import math
r_squared_origin = ((n*sum(df["Xi"]*df["YiOrigin"])) - (sum(df["Xi"]) * sum(df["YiOrigin"]))) / math.sqrt((n*sum(df["Xi"]**2))*((n*sum(df["YiOrigin"]**2) - ((sum(df["YiOrigin"]))**2))))


# In[ ]:


r_squared_origin


# Calculate the value of r squared for ordinary regression line

# In[ ]:


r_squared_hat = ((n*sum(df["Xi"]*df["YiHat"])) - (sum(df["Xi"]) * sum(df["YiHat"]))) / math.sqrt((n*sum(df["Xi"]**2))*((n*sum(df["YiHat"]**2) - ((sum(df["YiHat"]))**2))))
r_squared_hat


# # Question 5

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
data = pd.read_fwf("/kaggle/input/skincancer2.txt")


# In[ ]:


data.head()


# In[ ]:


# y = Mort.    x = lat
lat_mean = sum(data.Lat) / len(data.Lat)
mort_mean = sum(data.Mort) / len(data.Mort)
lat_mean , mort_mean


# ## Estimated value for B1 (Slope of the line)

# In[ ]:


b1_skn = sum((data.Lat - lat_mean)*(data.Mort - mort_mean)) / sum((data.Lat - lat_mean)**2)
b0_skn = mort_mean - (b1_skn * lat_mean)
print ("Slope =", b1_skn,"Intercept =",b0_skn)


# ## Sum of Squared

# In[ ]:


data["predictions"] = (data.Lat * b1_skn) + (b0_skn)
data.head()


# In[ ]:


mort_mean


# ## SSE SSR and SST

# In[ ]:


SSE = sum((data.Mort - data.predictions)**2)
SST = sum((data.Mort - mort_mean)**2)
SSR = sum((data.predictions - mort_mean)**2)

print ("Sum of Squared Errors = ",SSE," \nSum of Squared due to Regression =",SSR , "\nSum of Squared Total = ",SST) 


# ## Hypotheses

# In[ ]:


import math
MSE = math.sqrt(SSE/len(data.Lat))
yyy = MSE/math.sqrt(sum((data.Lat-lat_mean)**2))


# In[ ]:


#TO calculate T value
from scipy import stats
ci = 95
n = 48
t = stats.t.ppf(1- ((100-ci)/2/100), n-2)
x = t*yyy


# In[ ]:


print("The 95% confidence interval turns out to be\n",round(b1_skn - x,3),"< B1 <",round(b1_skn + x,3))


# ## Using functions to verify the parameters calculated above

# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()

train_x = data.iloc[:,1:2]
train_y = data.iloc[:,2:3]
reg.fit(train_x,train_y)

print ("Slope = ",reg.coef_,"\nIntercept = ",reg.intercept_)


# ## Summary

# In[ ]:


from statsmodels.formula.api import ols
model = ols("train_y ~ train_x", data).fit()
model.summary()


# In the summary, we can note the values of b0(intercept) and b1(regression coefficeint) and compare them to the values that we calculated previously. These values are similar to each other.
# 
# Also in the summary, it can be observed that the confidence interval lies between -7.220 to -4.843, which is also quite close to the values that we calculated previously.
#   

# ## Plot

# In[ ]:


import seaborn as sns
sns.regplot(data.Lat,data.Mort)


# In[ ]:




