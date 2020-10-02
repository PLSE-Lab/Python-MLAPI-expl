#!/usr/bin/env python
# coding: utf-8

# __Customer Churn using Telco Dataset__

# ### **Importing the packages**

# In[21]:


##Importing the packages
#Data processing packages
import numpy as np 
import pandas as pd 

#Visualization packages
import matplotlib.pyplot as plt 
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')


# ### **Importing the data**

# In[22]:


#Add the dataset Telco Customer Churn - Focused customer retention programs#
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# ### **Basic Analysis**

# In[23]:


#Find the size of the data Rows x Columns
data.shape


# **COMMENTS:** 

# In[24]:


#Display first 5 rows of Customer Data
data.head()


# **COMMENTS:** 
# 1. 
# 2. 
# 3. 
# 4. 

# In[26]:


#Find the the information about the fields, field datatypes and Null values
data.info()


# **COMMENTS:**  

# ### **Visualizing the impact of Categorical Features on the Target**

# In[27]:


#These fields does not add value, hence removed
data = data.drop(['customerID'], axis = 1)


# In[28]:


#Confirm that customerID column is dropped
data.head()


# In[29]:


#Find Churn size (Values)
data['Churn'].value_counts()


# ![](http://) **COMMENTS:**  

# ### **Convert Categorical values to Numeric Values**

# In[30]:


#A lambda function is a small anonymous function.
#A lambda function can take any number of arguments, but can only have one expression.
data['Churn']=data['Churn'].apply(lambda x : 1 if x=='Yes' else 0)


# In[31]:


#Finding the Count of Customer Churn. The output shows that 1869 customers churned(left) last month
data.Churn.value_counts()


# In[32]:


#Compare gender with Churn using crosstab.
pd.crosstab(data.Churn, data.gender)


# ### **Compare the fields**

# In[33]:


#Compare gender with Churn using crosstab. Add Total(margins)
pd.crosstab(data.Churn, data.gender, margins=True)


# In[34]:


#Compare gender with Churn using crosstab. Add Total(margins). Make it colorful.
pd.crosstab(data.Churn, data.gender, margins=True).style.background_gradient(cmap='autumn_r')


# In[35]:


#Compare gender with Churn using crosstab. Add Total(margins). Make it colorful. Normalize the data
pd.crosstab(data.Churn, data.gender, margins=True, normalize='index').style.background_gradient(cmap='autumn_r')


# In[38]:


#Compare gender with Churn using crosstab. Add Total(margins). Make it colorful. Normalize the data. Round it to two digits after decimal
pd.crosstab(data.Churn, data.gender, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:**  

# In[37]:


#Compare Partner with Churn using crosstab
pd.crosstab(data.Partner, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**   The Telco that had Partners have less customers leaving
# ![](http://) **RECOMMENDED ACTION:**  Target the customers who have Partners as they have higher retention rate

# In[39]:


#Compare Dependents with Churn using crosstab
pd.crosstab(data.Dependents, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# 

# ![](http://) **OBSERVATION:**  The Telco that had Dependents have less customers leaving
# ![](http://) **RECOMMENDED ACTION:**  Target the customers who have dependents. Family plans may help

# In[40]:


#Compare PhoneService with Churn using crosstab
pd.crosstab(data.PhoneService, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  PhoneService has less impact on customers leaving
# ![](http://) **RECOMMENDED ACTION:**  No action is required

# In[41]:


#Compare MultipleLines with Churn using crosstab
pd.crosstab(data.MultipleLines, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  No. of lines have less impact on customers leaving
# ![](http://) **RECOMMENDED ACTION:**  No action required

# In[42]:


#Compare InternetService with Churn using crosstab
pd.crosstab(data.InternetService, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**   Customers with FiberOptic internet connection have higher probability of leaving
# ![](http://) **RECOMMENDED ACTION:**  Recommended DSL connection to the customers and also investigate if there are problems in Fiber Optic connection

# In[43]:


#Compare OnlineSecurity with Churn using crosstab
pd.crosstab(data.OnlineSecurity, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  Customers who are not provided Online Security have higher chance of leaving
# ![](http://) **RECOMMENDED ACTION:**  Provide OnlineSecurity to the customers as the default offering or value added service with minimum fee

# In[44]:


#Compare OnlineBackup with Churn using crosstab
pd.crosstab(data.OnlineBackup, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:**  

# In[45]:


#Compare DeviceProtection with Churn using crosstab
pd.crosstab(data.DeviceProtection, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:**  

# In[46]:


#Compare TechSupport with Churn using crosstab
pd.crosstab(data.TechSupport, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:**  

# In[47]:


#Compare StreamingTV with Churn using crosstab
pd.crosstab(data.StreamingTV, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:** 

# In[48]:


#Compare StreamingMovies with Churn using crosstab
pd.crosstab(data.StreamingMovies, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:**  

# In[49]:


#Compare Contract with Churn using crosstab
pd.crosstab(data.Contract, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:**  

# In[50]:


#Compare PaperlessBilling with Churn using crosstab
pd.crosstab(data.PaperlessBilling, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:**  

# In[51]:


#Compare PaymentMethod with Churn using crosstab
pd.crosstab(data.PaymentMethod, data.Churn, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# ![](http://) **OBSERVATION:**  
# ![](http://) **RECOMMENDED ACTION:**  

# ### **Visualizing the impact of Numerical Features on the Target**

# In[53]:


#Plot the pairplot of all the Numerical parameters(data) against Churn
sns.pairplot(data=data,hue='Churn')


# ![](http://) **OBSERVATIONS:** 
# 1. 
# 2. 
# 3.  
#  ![](http://) **RECOMMENDED ACTIONS:**  
# 1. 
# 2. 
# 3. 

# In[55]:


#Comparing the numeric fields SeniorCitizen, tenure and MonthlyCharges against Customer Churn using boxplots
plt.figure(figsize=(24,12))
plt.subplot(131)  ; sns.boxplot(x='Churn',y='SeniorCitizen',data=data)
plt.subplot(132)  ; sns.boxplot(x='Churn',y='tenure',data=data)
plt.subplot(133)  ; sns.boxplot(x='Churn',y='MonthlyCharges',data=data)


# In[ ]:




