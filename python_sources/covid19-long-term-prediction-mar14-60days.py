#!/usr/bin/env python
# coding: utf-8

# # Introduction
#  * Predict confirmed cases for United States using linear regression. First find the number of new cases daily and perform linear regression to predict the number of new cases and the total number of cases in the next 60 days. 
#  * This is done out of curiosity and this is highly inaccurate. 
#  * With this, I wanted to spark discussions for long term predictions for coronavirus. 
# 
# # Comparing it with Fbprophet
#  * Please compare the model used in this study, linear regression, with fbprophet. Check out my other notebook:
# https://www.kaggle.com/myunghosim/covid19-60-day-predictions-for-us-using-fbprophet
# 
# ## questions
#  * When will it reach the peak and stop spreading? 
#  * How long will it keep on spreading for how long? 
# 
# ## references
#  * linear regression
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
#  * COVID19 dataset
#   - Johns Hopkins
#   https://github.com/CSSEGISandData/COVID-19
#   - Kaggle notebook by Andy yh
#  https://www.kaggle.com/andyyh/coronavirus-analysis-and-predictions

# In[ ]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[ ]:


import os
print(os.listdir("../input/"))


# In[ ]:


df = pd.read_csv("../input/covid-19-mar14/covid_19_data.csv")
df


# In[ ]:


usa = df.loc[df['Country/Region'] == 'US']
usa.max()


# In[ ]:


usa


# In[ ]:


usa.min()


# In[ ]:


confirmed_us = usa.groupby('ObservationDate')['Confirmed'].sum().reset_index()


# In[ ]:


confirmed_us = confirmed_us.set_index('ObservationDate')
confirmed_us.index


# In[ ]:


confirmed_us


# In[ ]:


size = len(confirmed_us)
us_change = [confirmed_us['Confirmed'][i]-confirmed_us['Confirmed'][i-1] for i in range(1,size)]


# In[ ]:


confirmed_us.plot(figsize=(15, 6))
plt.show()


# In[ ]:


days=[x+1 for x in range(size-1)]
plt.plot(days, us_change, color='red', linewidth=2)
plt.show()


# # linear regression

# In[ ]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


X = np.asarray([x for x in range(len(us_change))]).reshape(-1,1)
y = np.asarray(us_change).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[ ]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[ ]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


size = len(confirmed_us)
X_pred = np.asarray([x for x in range(size,size+60)]).reshape(-1,1)


# In[ ]:


y_pred = regressor.predict(X_pred)


# In[ ]:


#plot predictions for change of confirmed cases
plt.scatter(X_train, y_train,  color='gray')
plt.plot(X_pred, y_pred, color='red', linewidth=2)
plt.xlabel('Nth Day of Coronavirus in US')
plt.ylabel('Number of NEW Confirmed Cases(Daily)')
plt.title('Predicted NEW confirmed coronavirus cases in US - rate of change')
plt.show()


# In[ ]:


y_list=[]
cumul_sum=0
for y in list(y_pred.flatten()):
    cumul_sum+=y
    y_list.append(cumul_sum)


# In[ ]:


total_y = list(y_train.flatten())
total_y.extend(y_list)


# In[ ]:


total_x = [x for x in range(len(total_y))]


# In[ ]:


#plot predictions for cumulative confirmed cases
x_orig = [x for x in range(len(confirmed_us))]
plt.scatter(x_orig, confirmed_us,  color='gray')
plt.plot(total_x, total_y, color='red', linewidth=2)
plt.xlabel('Nth Day of Coronavirus in US')
plt.ylabel('Number of Confirmed Cases(Daily)')
plt.title('Predicted confirmed coronavirus cases in US')
plt.show()

