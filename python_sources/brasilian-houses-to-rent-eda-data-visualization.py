#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hello Everyone. Today i'm gonna share my first kernel on Kaggle. I love explain things with questions. Hence, questions will keypoints in this tutorial.
# 
# ### Questions:
# * [Question 1: How is the distribution of cities?](#1)
# * [Question 2: How many house owners accept animals?](#2)
# * [Question 3: How many houses are furnished?](#3)
# * [Question 4: Where is the accumulation point of total price?](#4)
# * [Question 5: How is the distribution of floors?](#5)
# * [Question 6: Which city has the most expensive rent prices?](#6)
# * [Question 7: Which floor is the most expensive?](#7)
# * [Question 8: Does the number of bathroom affects the rent amount?](#8)
# * [Question 9: How is the correlation between area,number of bathroom and rent amount?](#9)
# * [Question 10: How is the correlation between total amount and hoa?](#10)
# * [Question 11: How is the correlation between total amount and fire insurance?](#11)
# * [Question 12: How is the correlation between number of rooms, number of bathrooms and rent amount?](#12)
# * [Question 13: Which feature is correlated the most with rent amount: Area? Number of rooms? or Parking Spaces?](#13)
# 
#  
#  
# 
# 
# 
# 

# In[ ]:


import numpy as np # Numerical Python
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import iplot
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore') 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#I will analyze v2 because only this dataset includes the city feature
dataset=pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
dataset.head()


# In[ ]:


#General Info about dataset
dataset.info()


# In[ ]:


#Checking general statistical values
dataset.describe().T


# <a id="1"></a><br>
#  ### Question 1: How is the distribution of cities?

# In[ ]:


city_quantity=dataset.city.value_counts()

plt.figure(figsize=(15,10))
plt.pie(x=city_quantity, labels=city_quantity.index, autopct='%1.1f%%')
plt.title('Ratio of Cities',color = 'red',fontsize = 35)
plt.show()


# <a id="2"></a><br>
# ### Question 2: How many house owners accept animals?

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(dataset['animal'])
plt.show()


# <a id="3"></a><br>
# ### Question 3: How many houses are furnished?

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x="furniture", data=dataset)
plt.show()


# <a id="4"></a><br>
# ### Question 4: Where is the accumulation point of total price?

# In[ ]:


f,ax = plt.subplots(figsize = (15,15))
sns.distplot(dataset["total (R$)"],bins=1000,kde=False)
plt.show()


# ![](https://scontent.fadb3-1.fna.fbcdn.net/v/t1.15752-9/94996357_3795479163857846_3176845381656903680_n.png?_nc_cat=102&_nc_sid=b96e70&_nc_ohc=mcYGlazCunAAX9exNG_&_nc_ht=scontent.fadb3-1.fna&oh=8efe7274a71a27a0c165e52991b7e841&oe=5ECE7DC7)

# ![](https://scontent.fadb3-1.fna.fbcdn.net/v/t1.15752-9/95214926_581108095849232_5438832239292973056_n.png?_nc_cat=104&_nc_sid=b96e70&_nc_ohc=8a4Ph1p8rukAX-TsbMj&_nc_ht=scontent.fadb3-1.fna&oh=9008c37f9e28caac8f78d07a6a021c79&oe=5ECC6568)

# The point is between 2000 and 3000

# <a id="5"></a><br>
# ### Question 5: How is the distribution of floors?

# In[ ]:


#Let's check counts
print(dataset.floor.value_counts())


# In[ ]:


# '-' is not an acceptable value so i will drop that columns.
#Data cleaning
floor_data=dataset.drop(dataset[dataset.floor=='-'].index)
floor_data.floor=floor_data.floor.astype(int)

#Visualization Part
floor_data=floor_data.sort_values('floor')
floor_data.floor=floor_data.floor.astype(object)
plt.figure(figsize=(15,10))
sns.countplot(floor_data.floor)
plt.show()


# <a id="6"></a><br>
# ### Question 6: Which city has the most expensive rent prices?

# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x ='city',y='rent amount (R$)', data = dataset, showfliers = False)
plt.show()


# Looks like the answer is Sao Paulo

# <a id="7"></a><br>
# ### Question 7: Which floor is the most expensive?

# In[ ]:


#I will change room feature as a categoric feature in order to create a boxplot
room_data=dataset.iloc[:,[2,9]]
room_data['rooms']=room_data['rooms'].astype(object)
plt.figure(figsize=(15,10))
sns.boxplot(x ='rooms',y='rent amount (R$)', data = dataset)
plt.show()


# According to this graph floors including 5-8 are so close. The answer could be 5th or 7th floor.

# <a id="8"></a><br>
# ### Question 8: Does the number of bathroom affects the rent amount?

# In[ ]:


#I will change bathroom feature as a categoric feature to create a boxplot
categoric_bathroom=dataset.copy()
categoric_bathroom.bathroom=categoric_bathroom.bathroom.astype(object)
plt.figure(figsize=(15,10))
sns.boxplot(x ='bathroom',y='rent amount (R$)', data = categoric_bathroom)
plt.show()


# The answer is yes

# **Let's examine general relationships between numeric features**

# In[ ]:


numeric_data=dataset.select_dtypes(include=['int64']).copy()
plt.figure(figsize=(15,10))
sns.pairplot(numeric_data)
plt.show()


# **Correlation Heatmap could be more descriptive to see correlations between numeric features**

# In[ ]:


#Correlation Heatmap
corelation_matrix=numeric_data.corr()
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corelation_matrix, annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax,cmap='jet')
plt.show()


# <a id="9"></a><br>
# ### Question 9: How is the correlation between area,number of bathroom and rent amount?

# In[ ]:


sns.set(font_scale=1.5)
#I'm gonna use np.log() because there are many outliers that prevents a clear visualization
plt.figure(figsize=(15,10))
sns.scatterplot( x =np.log(dataset['area']+1) , y = np.log(dataset['rent amount (R$)']), 
                hue = dataset['bathroom'],size=dataset['parking spaces'],
                palette="afmhot",alpha=0.85)
plt.axis([2,8,6,10])
plt.show()


# <a id="10"></a><br>
# ### Question 10: How is the correlation between total amount and hoa?

# In[ ]:


plt.figure(figsize=(15,10))
sns.lmplot(x='total (R$)',y='hoa (R$)',data=dataset)
plt.show()


# <a id="11"></a><br>
# ### Question 11: How is the correlation between total amount and fire insurance?

# In[ ]:


plt.figure(figsize=(15,10))
sns.scatterplot(x='total (R$)',y='fire insurance (R$)',data=dataset)
plt.axis([0,35000,0,500])
plt.show()


# <a id="12"></a><br>
# ### Question 12: How is the correlation between number of rooms, number of bathrooms and rent amount?

# In[ ]:


import plotly.express as px
fig = px.scatter_3d(numeric_data, x='rooms',
                    y='bathroom', 
                    z=np.log(numeric_data['rent amount (R$)']), #I used np.log() again because there are many outliers. So z represents rent amount
                   color='rooms', 
       color_continuous_scale='icefire'
       )
iplot(fig)


# <a id="13"></a><br>
# ### Question 13: Which feature is correlated the most with rent amount: Area? Number of rooms? or Parking Spaces?

# In[ ]:


#I will use a simple normalization technique in order to see all features in the same domain.
normalized_data=numeric_data.copy()
for column in normalized_data.columns:
    normalized_data[column]=normalized_data[column]/normalized_data[column].max()
    
normalized_data.head()


# In[ ]:


#Visualization part
fig,ax1 = plt.subplots(figsize =(15,9))
sns.pointplot(x=normalized_data['area'],y=normalized_data['rent amount (R$)'],data=normalized_data,color='lime',alpha=0.8)
sns.pointplot(x=normalized_data['bathroom'],y=normalized_data['rent amount (R$)'],data=normalized_data,color='red',alpha=0.8)
sns.pointplot(x=normalized_data['parking spaces'],y=normalized_data['rent amount (R$)'],data=normalized_data,color='darkslategray',alpha=0.6)
plt.xticks(rotation=90)
plt.text(5.5,0.50,'area-rent amount (R$)',color='red',fontsize = 18,style = 'italic')
plt.text(5.4,0.46,'rooms-price rent amount (R$)',color='lime',fontsize = 18,style = 'italic')
plt.text(5.3,0.42,'parking spaces-rent amount (R$)',color='darkslategray',fontsize = 18,style = 'italic')
plt.xlabel('X - Axis',fontsize = 15,color='blue')
plt.ylabel('Y - Axis',fontsize = 15,color='blue')
plt.title('Area-Rent amount (R$) vs Room-Rent amount (R$) vs Parking space-Rent amount (R$)',fontsize = 20,color='blue')
plt.grid()


# The answer is area

# # Conclusion
# I hope graphs and results are descriptive enough for EDA. Please comment below if you have any questions or additional phrases. Thank's for your time and attention.
