#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# In[11]:


# In[ ]:



#importing visualization tool matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# 
# # ## Global Land Temperatures by Country Data
# # ### Getting sense of Data
# 

# In[ ]:


global_temp = pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv")


# In[ ]:


print(global_temp.info())


# #### 1. Dropping NAN data

# In[ ]:


# dropping NaN values
global_monthly_temp = global_temp.dropna()
global_monthly_temp.head(5)
global_monthly_temp.info()


# #### 2. Extracting Year information

# In[ ]:



#converting string date to date format
global_monthly_temp['dt'] = pd.to_datetime(global_monthly_temp['dt'], format="%Y-%m-%d")

#[x.date() for x in global_monthly_temp['dt']]
global_monthly_temp.loc[:, 'Year'] = [x.year for x in global_monthly_temp['dt']]

global_monthly_temp.head(5)
# 


# #### 3. Finding world global yearly average

# ### STEP 1: 
# - dropping data that have less than or equal to 6 entries which may not be a valid representation of that countries average. 

# In[ ]:



filtered_avg = global_monthly_temp.groupby(['Country','Year']).filter(lambda x: len(x) > 6)

filtered_avg.info()

filtered_global_yearly_average = filtered_avg.groupby('Year').mean().reset_index()
filtered_global_yearly_average.head(15)
# 


# ### STEP 2: 
# - Plotting the yearly global average temperature in Celcius which shows drastic increase at the first glance

# In[ ]:



fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(filtered_global_yearly_average['Year'], filtered_global_yearly_average['AverageTemperature'])
ax.set_xlabel("Year")
ax.set_ylabel("Global Average Temperature (C)")


# ### STEP 3: 
# - Checking how many countries data were recorded each year

# In[ ]:



#Finding for each year, the number of countries data. 
nCountries = filtered_avg.groupby('Year')['Country'].apply(lambda x: len(x.unique())).reset_index()
nCountries.describe()
# 


# - including no of countries data considered each year to the plot.

# In[ ]:



# 
# 
# 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(filtered_global_yearly_average['Year'], filtered_global_yearly_average['AverageTemperature'] )
ax.set_xlabel("Year")
ax.set_ylabel("Global Average Temperature (C)")

#adding secondary x-axis
ax2 = ax.twinx()
ax2.plot(nCountries['Year'], nCountries['Country'], 'r')
ax2.set_ylabel("# Countries")

# 


# ### STEP 4: 
# - Data Selection: Considering only the portion of data having largest consistent number of countries.

# In[ ]:



# 

nCountries[nCountries.Country > 220]
# The cut off for the greatest number of countries is 1891. 
# Lets filter data only from 1891 till date to see effect of global warming 


# -  Analysis of global temperature since 1891

# In[ ]:




global_temp_1891onwards = filtered_global_yearly_average[filtered_global_yearly_average.Year > 1891]
global_temp_1891onwards.head(10)


# In[ ]:


# plotting global temperature
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(global_temp_1891onwards['Year'], global_temp_1891onwards['AverageTemperature'] )
ax.set_xlabel("Year")
ax.set_ylabel("Global Average Temperature (C)")
# Global temperature has steadily increased by less than 2 degree celcius


# ### STEP 5: Extracting Train and Test Data
# 
#  - Feature: Time
#  - Response/Target: Average Global Temperature
#  - **Train Data:** Temperature observations between 1891-1989
#  - **Test Data:** Temperature observations since 1990 onwards

# In[ ]:


# Train Data
X_train = global_temp_1891onwards[global_temp_1891onwards['Year'] < 1990].Year
y_train = global_temp_1891onwards[global_temp_1891onwards['Year'] < 1990].AverageTemperature

# Since X variable has only one feature, this needs to reshaped
X_train = X_train.reshape(-1, 1)

# y is series data type. converting into np array
y_train = np.array(y_train)

# Test Data
X_test = global_temp_1891onwards[global_temp_1891onwards['Year'] >= 1990].Year
y_test = global_temp_1891onwards[global_temp_1891onwards['Year'] >= 1990].AverageTemperature

X_test = X_test.reshape(-1, 1)
y_test = np.array(y_test)


# In[ ]:


# Checking parameter shape and types before model building
print(y_train.shape)
print(X_train.shape)

print(type(X_train))
print(type(y_train))


# ### STEP 6: Building temperature predicting model using Linear Regression
#  - Training model predicting and estimating accuracy of the model 

# In[ ]:



# Load Class
from sklearn.linear_model import LinearRegression

# Instantiate Class
lin_reg = LinearRegression()

# Train the data
lin_reg.fit(X_train, y_train)

# predict the test
y_prediction = lin_reg.predict(X_test)

#eyeballing prediction results
print("Predicted results")
print(y_prediction[1:10])
print("Original results")
print(y_test[1:10])


# ### STEP 7: Visualizing results

# In[ ]:


# Plotting the prediction against original target
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure()
plt.plot(X_test, y_prediction)
plt.scatter(X_test, y_test)

