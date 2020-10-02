#!/usr/bin/env python
# coding: utf-8

# # Weather Analysis of World War 2

# Allied bombing raids during World War II turned the English sky white with contrails, providing a case study for modern scientists studying how the weather is affected by these long, feathery lines of condensation that form behind aircraft. 
# 
# Researchers focused on larger bombing raids between 1943 to 1945 after the United States Army Air Force (USAAF) joined the air campaign against Adolf Hitler's forces. Civil aviation was rare in the 1940s, so these combat missions represented a huge increase in flights and in potentially weather-altering contrails.

# Through this dataset, we are trying to analyze the weather variation during World War 2

# ### Importing Libraries

# In[ ]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
import os
import pandas_profiling as pp
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


os.getcwd()


# In[ ]:


data = pd.read_csv("../input/weatherww2/Summary of Weather.csv", encoding='latin', low_memory=False) 
data.tail(2)


# ### Loading the dataset

# ### Finding the Rows and Columns in the dataset

# In[ ]:


data.shape


# In[ ]:


pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


# In[ ]:


data.describe()


# ### Null Values

# In[ ]:


data.isnull().sum()


# #### Remove columns which were containing mostly NULL and also removed MIN, MAX, MEA as there are in fahrenhite  

# In[ ]:


data=data[['STA','Date','Precip','MaxTemp','MinTemp','MeanTemp','Snowfall',
           'PoorWeather','YR','MO','DA','SNF','TSHDSBRSGF']]


# In[ ]:


data.info()


# #### Exploratory Analysis using Panada _ Profiling

# In[ ]:


pp.ProfileReport(data)


# #### Distribution of Max Temperature with respect to Min Temperature

# We Can see from the graph that there seems to be strong positice relationship between min and max temperture

# In[ ]:


plt.scatter(data.MinTemp, data.MaxTemp,  color='gray')


# #### Correlation Matrix 

# We can see from the correlation matrix that there is a high correlation among min, max, mean temperature

# In[ ]:


f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax, cmap="BuPu")
plt.show()


# #### Distribition of Max Temperature

# Max Temperature distribution can be considered as Normal

# In[ ]:


plt.figure(figsize=(15,10))
sns.distplot(data['MaxTemp'], color='orange')
plt.xlabel('Max Temperature')
plt.xlim(0,60)
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.distplot(data['MinTemp'], color='Red')
plt.xlabel('Min Temperature')
plt.xlim(0,40)
plt.show()


# #### Reshaping the data before the regression model

# In[ ]:


X = data['MinTemp'].values.reshape(-1,1)
y = data['MaxTemp'].values.reshape(-1,1)


# #### Spliting the dataset into test and train

# We have considered 80% of the data as train and 20% as test

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Brief view of the data after spliting 

# In[ ]:


X_train


# #### Training the test dataset on the regression model

# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# #### Finding intercept and Coefficient

# In[ ]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# #### Predicting the value of test data based on trained model

# In[ ]:


y_pred = regressor.predict(X_test)
y_pred


# #### Comparing the resultset

# In[ ]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.head(10)


# #### Visualizing the variation in the actual V/S Predicted

# In[ ]:


df1 = df.head(20)
df1.plot(kind='bar',figsize=(8,12))
plt.grid(which='both', linestyle='-', linewidth='0.5')
plt.show()


# #### Predicted values distribution and spread

# In[ ]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# #### Efficiency of the model

# Our model have a very low RMSE 

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# #### R2

# In[ ]:


r2_score(y_test, y_pred,multioutput='variance_weighted') 


# This model explains 76% of the variability in the Max temperature and we can improve the afficency of the model by including other varuables into the cosideration for the model

# Thanks for reviewing this notebook.
# I will keep on updating the analysis
