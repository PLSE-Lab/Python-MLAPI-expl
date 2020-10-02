#!/usr/bin/env python
# coding: utf-8

# In this kernel I'll try to find out the analytical insight about the churn prediction at the bank customers 
# 
# ## 1.Importiing libraries

# In[ ]:


import pandas as pd
import numpy as np


# visualization
import seaborn as sns           
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




#baselie model libraries 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense,Flatten



# warning ignore
import warnings 
warnings.filterwarnings('ignore')


# ## 2.Loading dataset

# In[ ]:


data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')


# In[ ]:


data.head()


# Useful information to understand the table 
# 
# - **Tenure**: The customer's number of years in the bank 
# - **Balance**: The customer's account balance (amount of money present in a financial repository)
# - **Exited**: Churned or not ? 0 = No, 1= Yes

# ### Missing values

# In[ ]:


data.isna().sum()


# ### Duplication

# In[ ]:


sum(data.duplicated())


# ### Incorrect datatype

# In[ ]:


data.info()


# ## 3. EDA 
# 
# 
# ### 3.1 Countries: Geographcial reason of exited customer?

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
data['Geography'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number of customer by countries')
ax[0].set_ylabel('count')
sns.countplot(data=data,x='Geography',hue='Exited',ax=ax[1])
ax[1].set_title('Countries:Exited vs Non Exited')
ax[1].set_ylabel('count');


# **Observation**
# 
# `Germany` presents the hightest ratio of leaving (Lowest percentage of Non-exited customer and Highest percentage of exited customer)

# ### 3.2 Gender: relationship of exited customer and sex?

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
data['Gender'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number of customer by gender')
ax[0].set_ylabel('count')
sns.countplot(data=data,x='Gender',hue='Exited',ax=ax[1])
ax[1].set_title('Gender:Exited vs Non Exited')
ax[1].set_ylabel('count');


# Observation 
# 
# Ratio of leaving by `Female` >> Ratio of leaving by `Male` 

# ### 3.3 Age: which age group prefer to leave ?  

# In[ ]:


Non_Exited = data[data['Exited']==0]
Exited = data[data['Exited']==1]


# In[ ]:


plt.subplots(figsize=(18,8))
sns.distplot(Non_Exited['Age'])
sns.distplot(Exited['Age'])
plt.title('Age:Exited vs Non Exited')
plt.legend([0,1],title='Exited')
plt.ylabel('percentage');


# Observation 
# 
# - Non-Exited: right skewed graph
# - Exited: normal distributed graph
# 
# Age groupe over 40 is likely to leave bank. 

# ### 3.4 If the customer has more than 1 product will they stay rather than find other providers ? 

# In[ ]:


pd.crosstab(data.NumOfProducts,data.Exited,margins=True).style.background_gradient(cmap='OrRd')


# In[ ]:


f,ax = plt.subplots(1,2,figsize=(18,8))
data['NumOfProducts'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number of customer by Number of Product')
ax[0].set_ylabel('count')
sns.countplot(data=data,x='NumOfProducts',hue='Exited',ax=ax[1])
ax[1].set_title('Number of Product:Exited vs Non Exited')
ax[1].set_ylabel('count');


# Observation 
# 
# The ratio of exited cases from 3 Products definitely higher than under 2 products

# ## 3.5 If customer have higher creditscore are they willing to stay or leave ? 

# In[ ]:


plt.figure(figsize=(18,8))
plt.hist(x='CreditScore',bins=100,data=Non_Exited,edgecolor='black',color='red')
plt.hist(x='CreditScore',bins=100,data=Exited,edgecolor='black',color='blue')
plt.title('Credit score: Exited vs Non-Exited')
plt.legend([0,1],title='Exited');


# Observation  
# 
# Roughly can say that customers who have high credit score, tend to stay rather than to leave

# ## 3.6 Won't customer exit when they have a lot of money in the bank account ? 

# In[ ]:


plt.figure(figsize=(18,8))
p1=sns.kdeplot(Non_Exited['Balance'], shade=True, color="r")
p1=sns.kdeplot(Exited['Balance'], shade=True, color="b");
plt.title('Account Balance: Exited vs Non-Exited')
plt.legend([0,1],title='Exited');


# Observation 
# 
# Focus on 2 local maximum points of both graph.
# 
# - At the first local maximum (Balance = 0) ratio of `staying` is higher than `exited`.
# 
# - But at the second local maximum (Balacne = about120000) ratio of `exited` is higher than `staying` 
# 
# ->>> If the customers have a lot of money on their account they tend to become a churn customers

# ## 3.7 Can high Tenure & active menber guarantee that the customer will not leave ?  

# In[ ]:


sns.factorplot('IsActiveMember','Exited',col='Tenure',col_wrap=4,data=data);


# Observation 
# 
# We can clearly find 2 remerkable points 
# 
# - Tenure can influence on the exited but it's incomplete. For example clients who used this bank service for 7 years (Tenure) shows higher ratio to leave than 10 years client.
#   So it's hard to say that the The higher duration of service user have higher posibiities to stay.
#   
# - But the difference between acitve member and non active member is 100% clear that non active members are willing to leave double or triple times higher than active members

# ## 3.8 Which impact does Creditcard have on churn customer ? 

# In[ ]:


f,ax = plt.subplots(1,3,figsize=(18,8))
data['HasCrCard'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number of customer by credit card')
ax[0].set_xlabel('Credit card')

sns.countplot(data=data,x='HasCrCard',hue='Exited',ax=ax[1])
ax[1].set_title('Number of Product: Exited vs Non Exited')
ax[1].set_ylabel('count');

sns.boxplot(data=data,y='CreditScore',x='HasCrCard',hue='Exited',ax=ax[2])
ax[2].set_title('Credit card & score: Exited vs Non Exited');


# ## 3.9 Which impact does EstimatedSalary have on churn customer ? 

# In[ ]:


plt.figure(figsize=(18,8))
plt.hist(x='EstimatedSalary',bins=100,data=Non_Exited,edgecolor='black',color='red')
plt.hist(x='EstimatedSalary',bins=100,data=Exited,edgecolor='black',color='blue')
plt.title('Estimated salary: Exited vs Non-Exited')
plt.legend([0,1],title='Exited');


# Observation 
# 
# Hard to find any analytical insight that the estimated salary has a impact on churn customer.

# Observation 
# 

# ## 3.10 Correlation  analysis 

# In[ ]:


plt.title("features correlation matrics".title(),
          fontsize=20,weight="bold")

sns.heatmap(data.corr(),annot=True,cmap='RdYlBu',linewidths=0.2, vmin=-1, vmax=1,linecolor = 'black') 
fig=plt.gcf()
fig.set_size_inches(10,8);

