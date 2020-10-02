#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Importing the dataset
data = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


data.shape


# In[ ]:


#lets look at the Column Naames in our Data
data.columns


# In[ ]:


#lets see if we have any Nulls to Process
data.isnull().sum()
#We have 1474-Rating, 1-Type, 1-Content Rating, 8-Current Ver, 3-Andriod Ver


# In[ ]:


#Lets Look at the Rating column
data.Rating.head()


# In[ ]:


#We wil drop the row containing the App Category 1.9 as it seems to be irrelavant
data.head()
data.index[data['Rating'] == 19].tolist()


# In[ ]:


data.iloc[10471:10473,]


# In[ ]:


data.drop(data.index[10472],inplace=True)
data.reset_index()


# In[ ]:


data.shape


# In[ ]:


data.tail()


# In[ ]:


data.isnull().sum()
#Rating-1474, Type-1, Current Ver-8, Android Ver-2
#We will predict the missing rating values, but before that fill the other missing values


# In[ ]:


data.Type.value_counts()
#We have two categories Free10039, Paid800
# we will fill the remaining one vaue with Free Type


# In[ ]:


data.Type.fillna('Free',inplace=True)
data.Type.isnull().sum()


# In[ ]:


data['Current Ver'].value_counts()
#This Column shows the app version I assume, so it is better to fill the missing value with 
#'Varies with Device', wich is also the max occuring amongst apps


# In[ ]:


data['Current Ver'].fillna('Varies with device',inplace=True)
data['Current Ver'].isnull().sum()


# In[ ]:


data['Android Ver'].value_counts()
#We have only two misssing values, so it is better to fill with 4.1 and up


# In[ ]:


data['Android Ver'].fillna('4.1 and up',inplace=True)
data['Android Ver'].isnull().sum()


# In[ ]:


data.isnull().sum()
#We have null only in 'Rating' column


# In[ ]:


#Remove the + and , in Installs Column
data['Installs'] = data['Installs'].apply(lambda x : x.replace('+','') if '+' in str(x) else x)
data['Installs'] = data['Installs'].apply(lambda x : x.replace(',','') if ',' in str(x) else x)
data['Installs'] = data['Installs'].apply(lambda x : int(x))


# In[ ]:


temp = data.Size


# In[ ]:


data.Size.isnull().sum()


# In[ ]:


data.Size = data.Size.str.replace('.','')


# In[ ]:


data.Size = data.Size.replace('1000+',1000)


# In[ ]:


#Handle Size Column
data.Size = data.Size.str.replace('k','000')
data.Size = data.Size.str.replace('M','000000')


# In[ ]:


data.Size.value_counts()
#WE have a value 'Varies with device', which we can replace with Nan


# In[ ]:


##data.Size=data.Size.replace('Varies with device',np.nan)


# In[ ]:


data.Size.isnull().sum()


# In[ ]:


data.isnull().sum()
#We have null in Rating


# In[ ]:


labels = data.Category.value_counts(sort = True).index
cnt = data.Category.value_counts(sort = True)

plt.pie(cnt,labels=labels,shadow=True,autopct='%1.1f%%')
plt.show()
#This shows that famil and Gaming apps are most used, while other app categoreis being less than 10% each


# In[ ]:


sns.countplot(x='Category',data=data)
plt.xticks(rotation='vertical')
#Similar to the above plot, but we will visualize the count here


# In[ ]:


sns.countplot(x='Content Rating',data=data)
plt.xticks(rotation='vertical')
#Shows the count across the App wrt COntent Rating


# In[ ]:


sns.countplot(x='Genres',data=data)

plt.xticks(rotation='vertical')
from matplotlib import rcParams

# figure size in inches
#rcParams['figure.figsize'] = 20,20
#Shows the count across the App wrt Genres Rating
#AS there are large number of Genresin, we dont have a clear visualization


# In[ ]:


sns.countplot(x='Genres',data=data,hue='Type')
plt.xticks(rotation='vertical')
#Plot is not clear s we have lot of Gnres, but sure th installations of free apps tops in all Genres


# In[ ]:


sns.countplot(x='Content Rating',data=data,hue='Installs')
plt.xticks(rotation='vertical')


# In[ ]:


labels = data.Type.value_counts(sort = True).index
cnt = data.Type.value_counts(sort = True)

plt.pie(cnt,labels=labels,shadow=True,autopct='%1.1f%%')
plt.show()
#This shows that famil and Gaming apps are most used, while other app categoreis being less than 10% each


# In[ ]:




