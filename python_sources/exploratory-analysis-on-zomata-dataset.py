#!/usr/bin/env python
# coding: utf-8

# Hello! This is my first Analysis ever. Any feedback would be appreciated.Let me know what you think :)
# 
# 
# 
# 
# **About the Dataset :** 
# For this analysis I'll be using the Zomato Bangalore Restaurants Dataset which consists of information like the name, address, location, rate, votes, approx cost(for two), cuisines and reviews about the different restaurants in Bangalore. I'll be analysing this data to find meaningful insights.
# 
# 
# 
# **About the Analysis :**
# Bangalore is an IT hub so a lot of people come there either because of their job or are in search of a job. There are a high chances of employees ordering food from restaurants or dinning out so its a great opportunity for the restaurant businesses to bloom. By analysing this data some questions such as what type of services are provided by the restaurants? Which type of cuisines are preferred? Which restaurant types are more in number? Are the services provided and the ratings correlated? etc.

# ### **1. Importing all the libraries.**

# In[ ]:




import pandas as pd
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import iplot,plot,init_notebook_mode
import plotly.figure_factory as ff
import plotly.express as px
init_notebook_mode(connected = True)
import warnings
warnings.filterwarnings('ignore')


sns.set()


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file=os.path.join(dirname, filename)


# ### **2. Loading the dataset**

# In[ ]:


data=pd.read_csv(file)


# ### **3. Exploring the dataset.**

# In[ ]:


data.shape


# There are 51717 rows and 17 columns.

# In[ ]:


data.head()


# ### **4. Data Cleaning**

# #### 1. Removing duplicates on the basis of name and address of the restaurants.

# In[ ]:


data.drop_duplicates(subset=['name','address'],keep='first',inplace=True)
data.shape


# Result : 39218 rows were dropped which had duplicate values
# 

# #### 2. Removing unwanted columns.

# In[ ]:


data.drop(['url','address','phone','location'],axis=1,inplace=True)
data


# #### 3. Renaming Column Names 

# In[ ]:


data.rename(columns={'approx_cost(for two people)':'cost_fortwo','listed_in(type)':'dinning_type','listed_in(city)':'city'},inplace=True)
data


# #### 4. Checking for missing values 

# In[ ]:


# using a heatmap
fig,ax=plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(8)
ax=sns.heatmap(data.isnull(),cbar=False)


# In[ ]:


# expressing missing values as percentages to supplement the heatmap
for col in data.columns:
    pct_missing=np.mean(data[col].isnull())
    print('{} - {}%'.format(col,round(pct_missing*100)))


# In[ ]:


# dropping columns with missing values more than 36%
# dropping dish_liked column
data.drop('dish_liked',axis=1,inplace=True)


# #### 5. Cleaning Columns 

# In[ ]:


# exploring the menu_item column
data.loc[data['menu_item']=='[]']


# In[ ]:


# dropping the menu_item column
data.drop('menu_item',inplace=True,axis=1)


# In[ ]:


#handling the rate column
data['rate'].isnull().sum()


# In[ ]:


# checking for the types of values stored in rate
data['rate'].unique()


# In[ ]:


data.drop(data.index[data.rate=='NEW'],inplace=True)
data.drop(data.index[data.rate=='-'],inplace=True)


# In[ ]:


data['rate']=data['rate'].apply(lambda x: str(x).split('/')[0].strip())
data.drop(data.loc[(data['rate']=='nan') & (data['votes']!=0)].index,inplace=True)
data.loc[data['rate']=='nan']


# #### All the nan value rows in rate have corresponding value 0 in votes column. Hence replcing nan with 0 in the rate column

# In[ ]:


data['rate']=data['rate'].replace('nan','0')
data['rate']=pd.to_numeric(data['rate'])
data['rate'].dtype


# In[ ]:


# cleaning cost_fortwo column
data['cost_fortwo'].dtype


# In[ ]:


data['cost_fortwo']=data['cost_fortwo'].apply(lambda x: str(x).replace(',',''))
data.drop(data.index[data['cost_fortwo']=='nan'],inplace=True)
data['cost_fortwo']=data['cost_fortwo'].astype('int')
data['cost_fortwo'].dtype


# ### 5. Necessary Functions for visualization

# In[ ]:


def doughnutchart(labels,values,title):
    colors=["#F7B7A3","#EA5F89"]
    data=go.Pie(labels=labels,values=values,hole=0.6,pull=0.04,marker=dict(colors=colors))
    layout=go.Layout(title=title)
    fig=go.Figure(data=data,layout=layout)
    iplot(fig)
    
    
def barchart(index,values,name):
    data=go.Bar(x=index,y=values,orientation='v',text=values,textposition='auto',name=name,marker=dict(color='#003F5C'))
    layout=go.Layout(title=name,xaxis=dict(tickangle=-25),barmode='group')
    fig=go.Figure(data=data,layout=layout)
    iplot(fig)


# ### 6. Exploratory Data Analysis 

# #### 1. Citywise number of restaurants 

# In[ ]:


location=data['city'].value_counts()[:10]
barchart(location.index,location.values,'Citywise Number of Restaurants')


# Result : Top 10 cities with the number of restaurants. BTM, Brigade Road have more than 1000 restaurants.
# The reasons for the number or restaurants in a particluar city might be : population, number of offices, less land cost or rent comparatively than other city's.
# 

# #### 2. Top 10 Restaurant Types

# In[ ]:


temp=pd.DataFrame(data['rest_type'].dropna().str.split(',').tolist()).stack()

rest_type_value=list(temp.values)

# removing extra whitespaces
rest_type=[]
for ele in rest_type_value:
    rest_type.append(ele.strip())

# counting the values
rest_type_count=[]
rest_type_unique=list(set(rest_type))
for ele in rest_type_unique:
    rest_type_count.append(rest_type.count(ele))

# selecting the top 10 types
rest_type_val=[]
rest_type_ind=[]
pt=1
while pt<=10:
    num=max(rest_type_count)
    pos=rest_type_count.index(num)
    rest_type_val.append(rest_type_unique.pop(pos))
    rest_type_ind.append(rest_type_count.pop(pos))
    pt+=1

barchart(rest_type_val,rest_type_ind,'Top 10 Restaurant Types')


# In[ ]:


colors=['#57167E','#9B3192','#EA5F89','#F7B7A3','#FFF1C9','#E6F69D','#AADEA7','#64C2A6','#2D87BB','#3700FF']
fig=go.Figure(data=[go.Pie(labels=rest_type_val,values=rest_type_ind,marker=dict(colors=colors))],layout=go.Layout(title='Top 10 Types of Restaurants'))
fig.show()


# Result : Quick Bites is the most popular restaurant type followed by casual dinning and then delivery

# #### 3. Top 10 cuisines restaurants serve

# In[ ]:


temp=pd.DataFrame(data['cuisines'].dropna().str.split(',').tolist()).stack()
cui_type_value=list(temp.values)

# removing extra whitespaces
cui_type=[]
for ele in cui_type_value:
    cui_type.append(ele.strip())

# counting the values
cui_type_count=[]
cui_type_unique=list(set(cui_type))
for ele in cui_type_unique:
    cui_type_count.append(cui_type.count(ele))

# selecting the top 10 types
cui_type_val=[]
cui_type_ind=[]
pt=1
while pt<=10:
    num=max(cui_type_count)
    pos=cui_type_count.index(num)
    cui_type_val.append(cui_type_unique.pop(pos))
    cui_type_ind.append(cui_type_count.pop(pos))
    pt+=1


# In[ ]:


barchart(cui_type_val,cui_type_ind,'Top 10 cuisines restaurants serve')


# In[ ]:


fig=go.Figure(data=[go.Pie(labels=cui_type_val,values=cui_type_ind,marker=dict(colors=colors))],layout=go.Layout(title='Famous Types of Cuisines'))
fig.show()


# #### Result : More number of restaurants serve North Indian cuisine than South Indian it may be because a lot of people are settled in Bangalore due to work so they might prefer some other cuisine than South Indian and some change for the locals

# #### 4. Comparison of Restaurants providing online table booking facility

# In[ ]:


online_book=data['book_table'].value_counts()
doughnutchart(online_book.index,online_book.values,"Online Table Book Facility")


# #### Result : There are a lot of quick bites type of restaurants and people in India do not usually book tables in restaurants unless its a very famous place or for those restaurants where its compulsory to book tables.  

# #### 5. Comparison of Restaurants providing Online Ordering facility

# In[ ]:


order=data['online_order'].value_counts()
doughnutchart(order.index,order.values,"Online Ordering Facility")


# #### 6. Restaurants with their number of outlets

# In[ ]:


name=data['name'].value_counts()[:10]
barchart(name.index,name.values,"Restaurants with maximun number of outlets")


# #### Result : These restaurants have more outlets compared to others.

# #### 7. Relatioship of Rate and Votes

# In[ ]:


fig=go.Figure(data=go.Scatter(x=data['rate'],y=data['votes'],mode='markers',marker=dict(color='#64C2A6')))
fig.update_layout(title="Relation between Rate and Votes",xaxis_title='Rate',yaxis_title="Votes",font=dict(family='monospace',size=14))
iplot(fig)


# #### Result : We can see that the restaurants with higher number of votes have a higher rate.

# #### 8. Distribution of Cost

# In[ ]:


fig,ax=plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(8)
ax=sns.distplot(data['cost_fortwo'],kde=False,color='orange')
ax.set_title("Distribution of Aprrox Cost for two people")
plt.show()


# #### Result : A lot of restaurants have their average cost less than 1000.

# #### 9. 10 expensive restaurants

# In[ ]:


expensive=data.nlargest(10,'cost_fortwo')
expensive.reset_index(inplace=True)
print(expensive[['name','cost_fortwo']])


# #### 10. Dinnig Type

# In[ ]:


dinning=data['dinning_type'].value_counts()[:10]
barchart(dinning.index,dinning.values,'Dinning Types')


# #### Result : Delivery is the most famous dinning type

# #### 11. Top 10 cities according to votes

# In[ ]:


mostvoted=pd.DataFrame(data.groupby('city')['votes'].sum()).sort_values(by='votes',ascending=False).reset_index()
barchart(mostvoted['city'][:10],mostvoted['votes'],'Top 10 cities with highest number of votes')


# In[ ]:




