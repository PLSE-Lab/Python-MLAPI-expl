#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")


# In[ ]:


data.head()


# In[ ]:


country = pd.read_excel("../input/Country-Code.xlsx")
country.head()


# ## Merding both the dataframes on the common attribute that is **Country Code** . 

# In[ ]:


data1 = pd.merge(data, country, on='Country Code')
data1.head(3)


# # Existence of zomato over the globe except India

# In[ ]:


fig,ax = plt.subplots(1,1,figsize = (15,4))
ax = sns.countplot(data1[data1.Country != 'India']['Country'])
plt.show()


# ### Amongst these countries Zomato is most popular in United States and least popular in Canada

# # Zomato in India

# In[ ]:


res_India = data1[data1.Country == 'India']
res_India.head(3)


# ## Top 5 cities in India where maximum number of restaurants are registered on zomato

# In[ ]:


top5 = res_India.City.value_counts().head()
top5


# In[ ]:


f , ax = plt.subplots(1,1,figsize = (14,4))
ax = sns.barplot(top5.index,top5,palette ='Set1')
plt.show()


# # Majority of the orders are from 
# -  New Delhi
# -  Gurgaon
# -  Noida 
# - .Faridabad
# 
# **So we will focus on these cities**

# In[ ]:


NCR = ['New Delhi','Gurgaon','Noida','Faridabad']
res_NCR = res_India[(res_India.City == NCR[0])|(res_India.City == NCR[1])|(res_India.City == NCR[2])|
                    (res_India.City == NCR[3])]
res_NCR.head(3)


# In[ ]:


f,ax = plt.subplots(1,1,figsize = (14,4))
sns.countplot(res_NCR.City,palette ='cubehelix')
plt.show()


# ## Only 1046 restaurants in NCR provides Table booking facility

# In[ ]:


print(res_NCR['Has Table booking'].value_counts())
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax = sns.countplot(res_NCR['Has Table booking'],palette= 'Set1')
plt.show()


# # Around 41% of the restaurants in the NCR provides Online Delivery service 

# ###  This percentage is very low as in the time of growing technology people want to save their time by ordering the food online and spending the time in other works instead of going to restaurants especially in Delhi NCR. Restaurants can maximize their business by providing online delivery services to the customers.
# 

# In[ ]:


print(res_NCR['Has Online delivery'].value_counts())
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax = sns.countplot(res_NCR['Has Online delivery'],hue = res_NCR['City'],palette ='Set1')
plt.show()


# # Around 44% of the total restaurants in NCR provides average quality of food, 25% of them are not rated ,19% of them provides good food , 5% provides Very good food and just 3% of them provides poor food quality.

# In[ ]:


f, ax = plt.subplots(1,1, figsize = (14, 5))
ax = sns.countplot(res_NCR['Rating text'],palette ='Set1')
plt.show()


# # Price range of the restaurants varies from 1(Cheapest) - 4(Most Expensive)

# In[ ]:


f, ax = plt.subplots(1,1, figsize = (14, 5))
ax = sns.countplot(res_NCR['Price range'],hue = res_NCR['City'])
plt.show()


# # Number of restaurants in NCR with aggregate rating ranging from 1.9 to 4.9

# In[ ]:


agg_rat = res_NCR[res_NCR['Aggregate rating'] > 0]
f, ax = plt.subplots(1,1, figsize = (14, 4))
ax = sns.countplot(agg_rat['Aggregate rating'])
plt.show()


# ### Most of the restaurants have aggregate  rating  ranging from 2.8 to 3.8

# ## Top 5 Places in New Delhi with restaurants having aggregate rating over 4

# In[ ]:


res_NCR[(res_NCR.City == 'New Delhi') & (res_NCR['Aggregate rating'] >=4 )]['Locality'].value_counts().head()


# # Places in gurgaon best for Dinner dates

# In[ ]:


res_NCR[(res_NCR['City']=='Gurgaon') & (res_NCR['Aggregate rating']> 4) & (res_NCR['Votes'] > 1000) 
& (res_NCR['Rating text'] =='Excellent')]


# # Variety Cuisines in Delhi NCR

# In[ ]:


res_NCR.reset_index(inplace=True)
res_NCR.head(3)


# In[ ]:


cuisines = {"North Indian":0,'Chinese':0,'Fast Food':0,'Mughlai':0,'Bakery':0,'Continental':0,'Italian':0,
           "South Indian":0,'Cafe':0,'Desserts':0,'Street Food':0,'Mithai':0,'Pizza':0,'American':0,'Ice Cream':0}

for i in range(len(res_NCR.Cuisines)):
    for j in res_NCR.loc[i,'Cuisines'].split(','):
        if  j in cuisines.keys():
            cuisines[j] +=1
print(cuisines)


# In[ ]:


f, ax = plt.subplots(1,1, figsize = (15, 4))
ax = sns.barplot(x = list(cuisines.keys()),y=list(cuisines.values()),palette='cubehelix')
plt.show()


# ## We can clearly see that most of the restaurants provides North Indian Cuisines which is most popular all over India and other delicious cuisines including Mughlai, South Indian, Chinese , Italian and a lot more. 

# # Price range increases with the aggregate rating of the customers

# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(10,6))
ax  = sns.boxplot(x='Price range',y = 'Aggregate rating',data=res_NCR,palette='cubehelix')
plt.show()


# # End
