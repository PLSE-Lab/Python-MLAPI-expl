#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

import plotly.graph_objects as go
import plotly.express as px


# In[ ]:


df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")


# In[ ]:


df.shape


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


print('Attribute '+ 'Values')
for i in df.columns:
    print( i,len(df.loc[:,i].unique()) )


# ### There are 116 Port names but 117 port codes. This could be an enry error or there could be two ports with same port names. Let's see this in detail.

# In[ ]:


temp = df[['Port Name','Port Code']].drop_duplicates()
temp[temp['Port Name'].duplicated(keep=False)]


# In[ ]:


df.iloc[[29,217]]


# #### Eastport has two different port codes because there are two differnt ports by the name 'Eastport' in different states.

# ### Also, there are almost duble the locations than the port codes. This can mean that a port useually has 2 locations asociated with it. Let's see this.

# In[ ]:


indexes = df['Location'].drop_duplicates().index
temp = df.iloc[indexes].groupby(by='Port Code')['Location'].count()
temp.value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, explode=[0,0.20],startangle=15)
del temp


# #### We have a date column that is in string format we can get better results if we change it to datetime format. Also than we can extract Year and Month from date and see the distribution according to them.

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].apply(lambda x : x.year)

month_mapper = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun'
               ,7:'Jul', 8:'Aug', 9:'Sep' ,10:'Oct', 11:'Nov', 12:'Dec'}
df['Month'] = df['Date'].apply(lambda x : x.month).map(month_mapper)

del month_mapper


# In[ ]:


df.head()


# #### Let's see number of crossings by Mesures.

# In[ ]:


temp = pd.DataFrame(df.groupby(by='Measure')['Value'].sum().sort_values(ascending=False)).reset_index()
fig = px.bar(temp, x='Measure', y='Value', height=400)
fig.show()
del temp


# #### Most crossings are done by personal vehicle pessengers and personal vehicles. Let's see if it is likely distributed in both the borders of not.

# In[ ]:


temp = df.groupby(by=['Border','Measure'])['Value'].sum().reset_index()
temp.fillna(0,inplace=True)
temp.sort_values(by='Value', inplace=True)
fig = px.bar(temp, x='Measure', y='Value', color='Border', barmode='group')
fig.show()
del temp


# #### measures are likely distributed in both borders. But, One has higher number of crossings than the other. Let's see that.

# #### Below graph represents total number of crossings from borders.

# In[ ]:


temp = df.groupby(by='Border')['Value'].sum()
fig = go.Figure(data=[go.Pie(labels = temp.index, values=temp.values)])
fig.update_traces(textfont_size=15,  marker=dict(line=dict(color='#000000', width=2)))
fig.show()
del temp


# In[ ]:


plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='Year', y='Value', hue='Measure',legend='full')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Measure Values Through Years')


# #### Above chart shows number of crossings throughout years.Crossings have been decreasing since year 2000. There is a slight increment in pedestrians crossing over past few yers.

# In[ ]:


plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='Month', y='Value',legend='full', hue='Measure')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Value by month')


# ### Above graph shows number of crossings by month. July and Aug have highest crossings where Feb has the least number of crossings.

# In[ ]:


temp = pd.DataFrame(df.groupby(by='Port Name')['Value'].sum().sort_values(ascending=False)).reset_index()
px.bar(temp, x='Port Name', y='Value')
del temp


# #### Above graph represents ports and their number of crossings

# ### We can group measures by their size.

# #### Below we have two plots, both showing bar chart showing sum of values of different size of measures by different states.

# In[ ]:


measure_size = {'Trucks' : 'Mid_Size', 'Rail Containers Full' : 'Mid_Size', 'Trains' : 'Big_Size',
       'Personal Vehicle Passengers':'Small_Size', 'Bus Passengers':'Small_Size',
       'Truck Containers Empty':'Mid_Size', 'Rail Containers Empty':'Mid_Size',
       'Personal Vehicles' : 'Small_Size', 'Buses' : 'Mid_Size', 'Truck Containers Full' : 'Mid_Size',
       'Pedestrians':'Small_Size', 'Train Passengers':'Small_Size'}

df['Size'] = df['Measure'].map(measure_size)


# In[ ]:


temp = df.groupby(by=['Size','State'])['Value'].sum()
temp.fillna(0,inplace=True)
temp = temp.reset_index()
px.bar(temp, x='State', y='Value', facet_col='Size')


# In[ ]:


temp = df.groupby(by=['Size','State'])['Value'].sum().unstack()
temp.fillna(0,inplace=True)

plt.figure(figsize=(15,4))

plt.subplot(131)
temp.iloc[0].sort_values().plot(kind='bar')
plt.xticks(rotation=90)
plt.title('Big_Size')

plt.subplot(132)
temp.iloc[1].sort_values().plot(kind='bar')
plt.xticks(rotation=90)
plt.title('Mid_Size')

plt.subplot(133)
temp.iloc[2].sort_values().plot(kind='bar')
plt.xticks(rotation=90)
plt.title('Small_Size')

del temp


# ## Insights :
# - Minnesota has most number of big_size crossings but has averege on the other two categories.
# - Arizona has good number of small size crossings but average on the other two categories.
# - Ohio, Alaska and Montana has least amount of crossings in all the categories.
# - Michigan has 2nd heighest BIg and Mid size crossings but comparitively less small size crossings.
# - Texas has most mid_size and small size crossings and also, 3rd largest big_size crossings.
# - New York also has good number of crossings in all the three states.

# ### Let's see if crossings of different sizes are seasonal or not.

# In[ ]:


plt.figure(figsize=(15,6))
g = sns.FacetGrid(data=df, col='Size', sharey=False, height=5, aspect=1)
g.map(sns.lineplot, 'Month', 'Value')


# ## Insights :
# - Mid_Size crossings are least in dec, jan and jul and most in oct, mar and aug.
# - Big_Size crossings are least in feb and most in oct, mar and aug.
# - Small_Size crossings are least in jan and feb and most in aug and july.
# - Crossing rate per month is negetively correlated with size of crossing.

# In[ ]:





# In[ ]:




