#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### **IMPORTING DATA**

# In[ ]:


data=pd.read_csv("../input/daily-temperature-of-major-cities/city_temperature.csv")
data.head()


# ## Now that we had a quick look over the data, let's try some TIMEEEE SERRIEESSSS MODEEELLLSS
# <img src="https://media.tenor.com/images/8636ae856342f311049ec5573e263889/tenor.gif">

# **NOTE**: This dataset contains various cities and nations ,I will be using only the temperatures of Delhi.  

# In[ ]:


#Taking out only Delhi data
delhi=data[data["City"]=="Delhi"]
delhi.reset_index(inplace=True)
delhi.drop('index',axis=1,inplace=True)
delhi.describe()           


# ### Plotting the temperatures

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))
plt.plot(delhi["AvgTemperature"])
plt.ylabel("Temperature",fontsize=20)

OHHHH BOY O BOY we have some wrong values too , as the data have some -99 degree temps. WOOOOSSSSHHH I dont think that is possible as a student of science. We have to deal with this data , so I will make these values imputed.
# In[ ]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
delhi["AvgTemperature"].replace(-99,np.nan,inplace=True)#Replacing wrong entries with nan 
delhi["AvgTemperature"]=pd.DataFrame(imputer.fit_transform(delhi.loc[:,"AvgTemperature":]))


# Let's see how many years of data we have in our data.

# In[ ]:


print(min(delhi["AvgTemperature"]))
years=delhi["Year"].unique()
years


# In[ ]:


#Defining training and testing data
training_set=delhi[delhi["Year"]<=2015]
test_set=delhi[delhi["Year"]>2015]


# In[ ]:


#Mean of the temperatures
delhi.iloc[:,-1].mean()


# ### Let's see how our data looks after dealing with all the wrong values. 

# In[ ]:


plt.figure(figsize=(15,7))
plt.plot(delhi.iloc[:,-1])
plt.xlabel("Time Series",fontsize=20)
plt.ylabel("Temperature",fontsize=20)
#making a list of values to be plotted on y axis
y_values=[x for x in range(50,101,10)]
y_values.extend([delhi.iloc[:,-1].min(),delhi.iloc[:,-1].max(),delhi.iloc[:,-1].mean()])
plt.yticks(y_values)
plt.axhline(y=delhi.iloc[:,-1].mean(), color='r', linestyle='--',label="Mean")
plt.legend(loc=1)
plt.axhline(y=delhi.iloc[:,-1].max(), color='g', linestyle=':')
plt.axhline(y=delhi.iloc[:,-1].min(), color='g', linestyle=':')


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(delhi["AvgTemperature"],lags=365)
#plt.show()


# ### Clearly the graph looks geometrical and there is no abrupt cutoff in the values

# In[ ]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(delhi["AvgTemperature"],lags=50)


# ### There is an early cutoff after the first lag in the graph

# Let's start with the AutoReg model
# **Lags taken into consideration in model = 1,2,3,.....365**

# In[ ]:


from statsmodels.tsa.ar_model import AutoReg
model_AR=AutoReg(training_set["AvgTemperature"],lags=365)
model_fit_AR=model_AR.fit()
predictions_AR = model_fit_AR.predict(training_set.shape[0],training_set.shape[0]+test_set.shape[0]-1)


# In[ ]:



import seaborn as sns
plt.figure(figsize=(15,5))
plt.ylabel("Temperature",fontsize=20)
plt.plot(test_set["AvgTemperature"],label="Original Data")
plt.plot(predictions_AR,label="Predicted Data")
plt.legend()


# ## *LOLLLLL This went down too well, worked like a charm but unfortunately other models won't work like this*
# <img src="https://media1.tenor.com/images/610c4fe56dad195b0ffe5e76d2a02761/tenor.gif?itemid=4461765">

# Let's calculate the mean squared error of the predicted data

# In[ ]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(predictions_AR,test_set["AvgTemperature"])
mse


# Now with only lag=365 taking into consideration 

# In[ ]:


from statsmodels.tsa.ar_model import AutoReg
model_AR2=AutoReg(training_set["AvgTemperature"],lags=[365])
model_fit_AR2=model_AR2.fit()
predictions_AR2= model_fit_AR2.predict(training_set.shape[0],training_set.shape[0]+test_set.shape[0]-1)


# In[ ]:


plt.figure(figsize=(15,5))
plt.ylabel("Temperature",fontsize=20)
plt.plot(test_set["AvgTemperature"],label="Original Data")
plt.plot(predictions_AR2,label="Predicted Data")
plt.legend()


# **This graph looks accurate than the previous one but let's check if it actually works better or not**

# In[ ]:


mse=mean_squared_error(predictions_AR2,test_set["AvgTemperature"])
mse


# # **MOVING AVERAGE**

# In[ ]:


from statsmodels.tsa.arima_model import ARMA
model_MA=ARMA(training_set["AvgTemperature"],order=(0,10))
model_fit_MA=model_MA.fit()
predictions_MA=model_fit_MA.predict(test_set.index[0],test_set.index[-1])


# In[ ]:


plt.figure(figsize=(15,5))
plt.ylabel("Temperature",fontsize=20)
plt.plot(test_set["AvgTemperature"],label="Original Data")
plt.plot(predictions_MA,label="Predictions")
plt.legend()


# In[ ]:


mse=mean_squared_error(predictions_MA,test_set["AvgTemperature"])
mse


# # **Autoregressive Moving Average (ARMA)**

# In[ ]:


model_ARMA=ARMA(training_set["AvgTemperature"],order=(5,10))
model_fit_ARMA=model_ARMA.fit()
predictions_ARMA=model_fit_ARMA.predict(test_set.index[0],test_set.index[-1])


# In[ ]:


plt.figure(figsize=(15,5))
plt.ylabel("Temperature",fontsize=20)
plt.plot(test_set["AvgTemperature"],label="Original Data")
plt.plot(predictions_ARMA,label="Predictions")
plt.legend()


# In[ ]:


mse=mean_squared_error(predictions_ARMA,test_set["AvgTemperature"])
mse


# 

# # ***NOTE*:**
# 
# It is pretty lame to try to predict something that is going to happen over the next 5 years , but when we have data at our disposal why should'nt we ?
# That's what data science is for.
# The model used are just for showcase purposes i.e. how they perform when used on an *almost stationary* data.

# # I would be trying to create another notebook for ARIMA and ARIMAX models with a different data set, this is enough for now with the current dataset. Have a nice day.
# <img src="https://media1.tenor.com/images/b93d03f39d1b379cf648e5568a4537e7/tenor.gif?itemid=11696907">
