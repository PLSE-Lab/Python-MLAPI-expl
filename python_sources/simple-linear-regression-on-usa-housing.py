#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import the needed packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#import the dataset
df  = pd.read_csv('../input/USA_Housing.csv')


# In[4]:


#Look at the dataset
df.head(5)


# In[5]:


#Looking the data types of dataset
#df.info()


# In[6]:


#Looking at mathematical stats for the datasets
df.describe()


# In[7]:


df.columns


# In[8]:


#Creating the pair-plot for the dataset 
sns.pairplot(df)


# In[9]:


#Creating distribution plot for the price
#To see how the price is distributed along the dataset
sns.distplot(df['Price'])


# In[31]:


#making a heatmap for the correlation of dataset
fig = plt.figure(figsize = (10,7))
sns.heatmap(df.corr(), annot = True,cmap = "coolwarm")


# In[11]:


#Predicting Features
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[12]:


#response feature
y = df['Price']


# In[14]:


from sklearn.cross_validation import train_test_split


# In[15]:


#Dividing our dataset in train and test data's
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size = 0.4, random_state=101)


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


#lets fit a model
lm = LinearRegression()


# In[18]:


lm.fit(X_train,y_train)


# In[19]:


#Intercept for our predictions
print(lm.intercept_)


# In[20]:


#Coefficient for our predictions
lm.coef_


# In[21]:


#Joining the coefficient with its features
cdf = pd.DataFrame(lm.coef_,X.columns, columns = ['coeff'])


# In[22]:


cdf


# In[23]:


#predicting the models for test dataset
predictions = lm.predict(X_test)


# In[24]:


#Plotting the predictions
plt.scatter(y_test, predictions)


# In[25]:


#Residuals
sns.distplot((y_test-predictions))


# In[26]:


from sklearn import metrics


# In[27]:


#Some mathametics errors
#lesser the error more accurate is our predictions.
#Mean absolute error
metrics.mean_absolute_error(y_test, predictions)


# In[28]:


#Mean squared error
metrics.mean_squared_error(y_test, predictions)


# In[29]:


#Root mean squared error
np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[ ]:




