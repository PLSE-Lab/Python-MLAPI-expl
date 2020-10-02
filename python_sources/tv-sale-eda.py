#!/usr/bin/env python
# coding: utf-8

# <h2>Introduction</h2>
#         Given data contains Jan-2014 to Aug-2016 daily TV sales quantity. There are total 124 Models. This data is collected from one of the leading brand of Bangladesh. Annually there are two big festivals (EID) which follows Islamic lunar calendar. provided data are in csv format. it contains only three columns.
# 
#     Date: Date of Sale
# 
#     Model: TV Model (not Original)
# 
#     Count: Sale Quantity

# <h2>Load Libraries</h2>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Retrieving the  data</h2>

# <h3>a. Read the Data</h3>

# In[ ]:


data = pd.read_csv("../input/Date and model wise sale.csv")


# <h3>b. Glimpse of Data</h3>
# let's see how datas are distributed over the dataset

# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# Lets convert the datatype of 'date' column from object to datetime. We can use the date informations for further analysis of dataset.

# In[ ]:


data['Date']=pd.to_datetime(data['Date'])


# Let's check whether there are any missing data in dataset

# In[ ]:


total = data.isnull().sum()
percent = (data.isnull().sum()/data.isnull().count()*100)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(3)


# <h2>Feature Engineering</h2>

# <h2>Date</h2>

# In[ ]:


# Adding date features
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year


# <h3>Sales representation with respect to Date </h3>
# Let's see the yearly,monthly and daily basis sale for models.

# In[ ]:


#Sell by year
sell_by_year = data.groupby('year')['Count'].sum()
sell_by_year


# In[ ]:


plt.figure(figsize=(15,5))
plt.title(' Total sell throughout 2014-2016')
sns.countplot(x='year', data=data, color='lightblue');


# The collected datas are from 2014 to 2016. There is an increase in sales for each year except 2016 but this is probably because it is not yet over.

# In[ ]:


print('Starting date on the dataset')
print(min(data['Date']))
print('Ending date on the dataset')
print(max(data['Date']))


# We can see, the year 2016 is not fully completed, so our assumption is right. 
# Let's find out monthly basis sale throughout 2014-2016

# In[ ]:


sell_by_month = data.groupby('month')['Count'].sum()
sell_by_month


# In[ ]:


#Total sell on each month throughout 2014-2016
plt.figure(figsize=(10,5))
plt.title(' Total Sale on each month throughout 2014-2016')
plt.ylabel('Sale amount')
data.groupby('month').Count.sum().plot(kind='bar',color='lightblue')


# We can see that the sale is on its peak on 7th month(July).Let's find the reason by analyzing the sale on each year.

# In[ ]:


data_2014 = data[data["Date"].dt.year == 2014]
data_2015 = data[data["Date"].dt.year == 2015]
data_2016 = data[data["Date"].dt.year == 2016]


# In[ ]:


#Sell on each month of 2014
plt.figure(figsize=(10,5))
plt.title(' Sale on each month of 2014 ')
plt.ylabel('Sale amount')
data_2014.groupby('month').Count.sum().plot(kind='bar', color='lightblue')


# Here, we can that the sales of televisions are witnessing remarkable growth on July and September as these two month are followed by Eid-ul-Fitr(July 29, 2014) and Eid-ul-Azha(October 6,2014). So, sale of televisions have reached its peak ahead of Eid-ul-Fitr and Eid-ul-Azha,

# In[ ]:


#Sell on each month of 2015
plt.figure(figsize=(10,5))
plt.title(' Sale on each month of 2015 ')
plt.ylabel('Sale amount')
data_2015.groupby('month').Count.sum().plot( kind='bar',color='lightblue')


# Here, we can see that the sales of televisions are witnessing remarkable growth on July as this month is followed by Eid-ul-Fitr(July 17, 2015). But there is a decline on September(this month is followed by Eid-ul-Adha(September 25,2015), it may be the effect of natural hazards occured on that month. 

# In[ ]:


#Sell on each month of 2016
plt.figure(figsize=(10,5))
plt.title(' Sale on each month of 2016 ')
plt.ylabel('Sale amount')
data_2016.groupby('month').Count.sum().plot(kind='bar', color='lightblue')


# Here, we can see that the sales of televisions are witnessing remarkable growth on July as this month is followed by Eid-ul-Fitr(July 7, 2016). 

# <h4>Find out sell by each date throughout 2014-2016</h4>

# In[ ]:


#Sell on each day of a month in 2014
plt.figure(figsize=(10,5))
plt.title(' Sale on each month of 2014 ')
plt.ylabel('Sale amount')
data_2014.groupby('day').Count.sum().plot(kind='bar', color='lightblue')


# In[ ]:


#Sell on each day of a month in 2015

plt.figure(figsize=(10,5))
plt.title(' Sale on each month of 2015 ')
plt.ylabel('Sale amount')
data_2015.groupby('day').Count.sum().plot(kind='bar', color='lightblue')


# In[ ]:


#Sell on each day of a month in 2016

plt.figure(figsize=(10,5))
plt.title(' Sale on each month of 2016 ')
plt.ylabel('Sale amount')
data_2016.groupby('day').Count.sum().plot(kind='bar', color='lightblue')


# While counting the sale on each day for a specific year->it's basically sum up all the sale on a specific day for each month(i.e sale on day 1 represents all the sale on day 1 throughout the year). Again there is decline on day 31. This is because all months do not have day 31.

# In[ ]:


data['Model'].unique()


# In[ ]:


#Total numbers of models
total_no_of_models = len(data.Model.unique())
total_no_of_models


# In[ ]:


#model and sale relationship 
models_by_sale = data.groupby('Model')['Count'].sum()
models_by_sale.sort_values(axis=0, ascending=False, inplace=True)


# In[ ]:


#figure model by count
f, ax = plt.subplots(figsize=(12, 48))
ax=sns.barplot(models_by_sale, models_by_sale.index,orient='h', color='lightblue')
ax.set(title='Models by Sales',xlabel='Count ', ylabel='Model name')
plt.show()


# From this figure we can see that M24 is the most popular tv model(If sale is high for a specific product than it is the most popoular product). Let's visualize the sale of top 5 models.

# In[ ]:


#Assigning variable for the top 5 models
top_models = models_by_sale.head(5)

#Visualizing the top 5 models sale

for model in top_models.index:
    top_model = data[data.Model == model]
    plt.figure(figsize=(12,7))
    plt.title(model+' Sale vs. Time (top)')
    plt.ylabel('Sale amount')
    plt.xlabel('Time')
    plt.plot(pd.to_datetime(top_model['Date']),top_model['Count'],color='lightblue') 
    plt.show()


# From the visualization of the sale of top 5 tv models we can see that a model reaches its peak after launching and the growth declines after the launching of another new model

# Now visualize which tv model sells on what amount in a specific year

# In[ ]:


#Sell of each model in 2014
plt.figure(figsize=(30,10))
plt.title(' Sale of each model on 2014 ')
plt.ylabel('Sale amount')
data_2014.groupby('Model').Count.sum().plot(kind='bar', color='lightblue')


# The sale of TV model M60 is higher than others model in 2014

# In[ ]:


#sell of each models in 2015

plt.figure(figsize=(30,10))
plt.title(' Sale of each model on 2015 ')
plt.ylabel('Sale amount')
data_2015.groupby('Model').Count.sum().plot(kind='bar', color='lightblue')


# The sale of TV model M24 and M25 are higher than others model in 2015

# In[ ]:


#sell of each models in 2016

plt.figure(figsize=(25,10))
plt.title(' Sale of each model on 2016 ')
plt.ylabel('Sale amount')
data_2016.groupby('Model').Count.sum().plot(kind='bar', color='lightblue')


# The sale of TV model M22 is higher than others model in 2016

# In[ ]:




