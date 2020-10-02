#!/usr/bin/env python
# coding: utf-8

# [1. Ovearall Literacy Rate in India.](#1)
# 
# 
# [2. Total Literacy Rate Across Nation.](#2)
# - Lowest and Highest literacy rate in 2001
# - Lowest and Highest literacy rate in 2011
# - Percentage change in 'Total Literacy Rate' during 2001-2011.
# 
# 
# [3. Rural Literacy Rate Across Nation](#3)
# - Lowest and Highest literacy rate in 2001
# - Lowest and Highest literacy rate in 2011
# - Percentage change in 'Rural Literacy Rate' during 2001-2011.
# 
# 
# [4. Urban Literacy Rate Across Nation](#4)
# - Lowest and Highest literacy rate in 2001
# - Lowest and Highest literacy rate in 2011
# - Percentage change in 'Urban Literacy Rate' during 2001-2011.
# 
# [5. States vs Union Territories](#5)
# 
# 
# [6. East, West, North, South](#6)
# 
# 
# [7. Literacy Rate in each State/ Union Territory](#7)

# #### Importing Libraries:

# In[ ]:


import numpy as np #Linear Algebra
import pandas as pd #To Wrok WIth Data
## I am going to use plotly for visualizations. It creates really cool and interective plots.

import matplotlib.pyplot # Just in case.
import plotly.express as px #Easy way to plot charts
import plotly.graph_objects as go #Does the same thing. Gives more options.
import plotly as ply # The whole package
from plotly.subplots import make_subplots #As the name suggests, for subplots.
import seaborn as sns #Just in case. It can be useful sometimes.


# In[ ]:


df = pd.read_csv("../input/govt-of-india-literacy-rate/GOI.csv") #Loading the dataset.


# In[ ]:


df.head() # Let's take a look at the data.


# In[ ]:


df.isnull().sum() # See if we have any null values.


# #### We have data for two years 2011 and 2001 which have a difference of a decade between them. We can generate new attribute to see the percentage change in litercy rate over the decade.

# In[ ]:


df['Total - Per. Change'] = (df.loc[:,'Literacy Rate (Persons) - Total - 2011'] - 
                df.loc[:,'Literacy Rate (Persons) - Total - 2001'])/df.loc[:,'Literacy Rate (Persons) - Total - 2001']
df['Rural - Per. Change'] = (df.loc[:,'Literacy Rate (Persons) - Rural - 2011'] - 
                df.loc[:,'Literacy Rate (Persons) - Rural - 2001'])/df.loc[:,'Literacy Rate (Persons) - Total - 2001']
df['Urban - Per. Change'] = (df.loc[:,'Literacy Rate (Persons) - Urban - 2011'] - 
                df.loc[:,'Literacy Rate (Persons) - Urban - 2001'])/df.loc[:,'Literacy Rate (Persons) - Total - 2001']


# In[ ]:


## Column names are too long, I don't need that much info in a column name. So, i am altering the names.
new_col=[]
for i in df.columns:
    new_col.append(i.split('(Persons) - ')[-1])
df.columns=new_col


# In[ ]:


df.head() # Let's look at our dataframe after all the changes.


# #### We have data of the whole country and the states and union territories. I am going to view the overall Literacy rates of the country and then we'll remove this from our dataset. So that it is easy for us to view and compare literacy rates amongst States/ Union Territories.

# <a class = 'anchor' id=1></a>
# ## Overall Literacy Rates in India.

# In[ ]:


India = df[df['Category'] == 'Country'].T
India = India.iloc[2:8,:]
India.reset_index(inplace=True)
India.columns = ['Measure', 'Value']
India.loc[:,'Measure'] = India['Measure'].apply(lambda x : str(x).split(' -')[0])
India_2001 = India.iloc[[0,2,4],:]
India_2011 = India.iloc[[1,3,5],:]
fig = go.Figure(data=[
    go.Bar(name='2001', x=India_2001['Measure'], y=India_2001['Value'], marker_color='rgb(55, 83, 109)'),
    go.Bar(name='2011', x=India_2011['Measure'], y=India_2011['Value'], marker_color='rgb(26, 118, 255)')
])
fig.update_layout(barmode='group', title='Overall Literacy Rate in India :')
fig.show()


# ### Insights:
# - Total literacy rate in india has incresed by 8.2 units in previous dacade. That is an increse of 12.65% in the previous measure.
# - Literacy rate in rural india has incresed by 9.1 units in the previous dacade. That is an increse of 14.04% in the precious measure.
# - iteracy rate in urban india has incresed by 4.5 units in the previous decade. That is an increse of 14.04% in the precious measure.

# In[ ]:


df = df.iloc[1:,:] #Removing data for India as a whole country.
df.rename(columns={'Country/ States/ Union Territories Name' :'States/ Union Territories'}, inplace = True) 


# #### We have three attributes for literacy rates: total, rural and urban. We'll take a look on each of them to see how they're distributed across the nation.

# <a class = 'anchor' id = 2></a>
# 
# ## Total Literacy Rate Across Nation:

# In[ ]:


df.sort_values(by='Total - 2001', inplace=True)

fig = go.Figure(data = [
    go.Scatter(name='2001', x=df['States/ Union Territories'], y=df['Total - 2001'], mode='markers'),
    go.Scatter(name='2011', x=df['States/ Union Territories'], y=df['Total - 2011'], mode='markers')
])

fig.update_layout(barmode='group', title = 'Total Literacy Rate Across Nation :')
fig.show()


# In[ ]:


lowest_2001 = df.sort_values(by=['Total - 2001']).head()
highest_2001 = df.sort_values(by=['Total - 2001']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2001', x=lowest_2001['States/ Union Territories'], y=lowest_2001['Total - 2001'], mode='markers'),
    go.Line(name = 'Highest_2001', x=highest_2001['States/ Union Territories'], y=highest_2001['Total - 2001'], mode='markers')
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Total literacy" rate in 2001 :')
fig.show()


# In[ ]:


lowest_2011 = df.sort_values(by=['Total - 2011']).head()
highest_2011 = df.sort_values(by=['Total - 2011']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2011', x=lowest_2011['States/ Union Territories'], y=lowest_2011['Total - 2011'], mode='markers'),
    go.Line(name = 'Highest_2011', x=highest_2011['States/ Union Territories'], y=highest_2011['Total - 2011'], mode='markers')
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Total Literacy" literacy rate in 2011 :')
fig.show()


# In[ ]:


px.bar(df.sort_values(by='Total - Per. Change'),
       x='States/ Union Territories', y='Total - Per. Change',
       color='Total - Per. Change', title='Totel Per. Change')


# ### INSIGHTS : 
# - Bihar, Jharkhand, Arunachal Pradesh, Jammu & Kashmir and Uttar Pradesh were the least literate states/Union Territories in 2001.
# - Kerala, Mizoram, Lakshadweep, Goa and Chandigarh are the most literate states/Union territories in 2001.
# 
# 
# - Rajasthan and Andhra Pradesh Couldn't keep up with other states and fell in 5 least literate states with Bihar, Arunachal Pradesh and Jharkhand. Whereas Jammu & kashmir and Uttar pradesh managed to improve in 2011.
# - Tripura managed to increse it's literacy rate to 5 most literate states along with Kerala, Lakshadweep, mizoram and Goa in 2011.
# 
# 
# - Mizoram, Kerala, Chandigarh, NCT of Delhi and Ponducherry have least percentage increse in literacy rate.
# - Percentage Increse in Total Literacy is highest in D & N Haveli, Bihar, Jharkhand, Jammu & Kashmir and Arunachal Pradesh.
# 
# 
# - In Year 2001 total 13 States/Union Territories had lesser literacy rate than overall indian literacy rate.
# - In Year 2011 total 11 States/Union Territories had lesser literacy rate than overall indian literacy rate. Meghalaya and D & N Haveli managed to increse their literacy rate.
# -  Bihar, Jharkhand, Arunachal Pradesh, Jammu & Kashmir, Uttar Pradesh, Rajasthan, Andhra Pradesh, Odisha, Assam, Madhya Pradesh and Chhattisgarh still have lesser Total literacy rate than overall literacy rate of the Country.

# <a class = 'anchor' id=3></a>
# 
# ## Rural Literacy Rate Across Nation:

# In[ ]:


df.sort_values(by='Rural - 2001', inplace=True)

fig = go.Figure(data = [
    go.Line(name='2001', x=df['States/ Union Territories'], y=df['Rural - 2001'], mode='markers'),
    go.Line(name='2011', x=df['States/ Union Territories'], y=df['Rural - 2011'], mode='markers')
])

fig.update_layout(barmode='group', title = 'Literacy rate in rural areas acorss the country :')
fig.show()


# In[ ]:


lowest_2001 = df.sort_values(by=['Rural - 2001']).head()
highest_2001 = df.sort_values(by=['Rural - 2001']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2001', x=lowest_2001['States/ Union Territories'], y=lowest_2001['Rural - 2001'], mode='markers'),
    go.Line(name = 'Highest_2001', x=highest_2001['States/ Union Territories'], y=highest_2001['Rural - 2001'], mode='markers')
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Rural literacy" rate in 2001 :')
fig.show()


# In[ ]:


lowest_2011 = df.sort_values(by=['Rural - 2011']).head()
highest_2011 = df.sort_values(by=['Rural - 2011']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2011', x=lowest_2011['States/ Union Territories'], y=lowest_2011['Rural - 2011'], mode='markers'),
    go.Line(name = 'Highest_2011', x=highest_2011['States/ Union Territories'], y=highest_2011['Rural - 2011'], mode='markers')
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Rural literacy" rate in 2011 :')
fig.show()


# In[ ]:


px.bar(df.sort_values(by='Rural - Per. Change'),
       x='States/ Union Territories', y='Rural - Per. Change',
       color='Rural - Per. Change', title='Rural Per. Change')


# ### Insights :
# - We have the same distribution of rural literacy rate among States/Union Territories as we saw in total literacy rate.
# - Bihar, Jharkhand, Jammu & Kashmir, D & N Haveli and Utter Pradesh have worked hard in their rural areas and thus they have highest percentage increrse in rural literacy rate.
# - Mizoram, Kerala, NCT of Delhi, Chandigarh and A & N Islands have least percentage increse in rural literacy rate.
# - The states that have worked the most in their rural areas are the ones which had least rural literacy rate in 2001.

# <a class = 'anchor' id=4></a>
# 
# ## Urban Literacy Rate Across Nation:

# In[ ]:


df.sort_values(by='Urban - 2001', inplace=True)

fig = go.Figure(data = [
    go.Line(name='2001', x=df['States/ Union Territories'], y=df['Urban - 2001'], mode='markers'),
    go.Line(name='2011', x=df['States/ Union Territories'], y=df['Urban - 2011'], mode='markers')
])

fig.update_layout(barmode='group')
fig.show()


# In[ ]:


lowest_2001 = df.sort_values(by=['Urban - 2001']).head()
highest_2001 = df.sort_values(by=['Urban - 2001']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2001', x=lowest_2011['States/ Union Territories'], y=lowest_2001['Urban - 2001'], mode='markers'),
    go.Line(name = 'Highest_2001', x=highest_2011['States/ Union Territories'], y=highest_2001['Urban - 2001'], mode='markers')
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Urban literacy" rate in 2001 :')
fig.show()


# In[ ]:


lowest_2011 = df.sort_values(by=['Urban - 2011']).head()
highest_2011 = df.sort_values(by=['Urban - 2011']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2011', x=lowest_2011['States/ Union Territories'], y=lowest_2011['Urban - 2011'], mode='markers'),
    go.Line(name = 'Highest_2011', x=highest_2011['States/ Union Territories'], y=highest_2011['Urban - 2011'], mode='markers')
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Urban literacy" rate in 2011 :')
fig.show()


# In[ ]:


px.bar(df.sort_values(by='Urban - Per. Change'),
       x='States/ Union Territories', y='Urban - Per. Change',
       color='Urban - Per. Change', title='Urban Per. Change')


# ### Insights :
# - Again, We have the same distribution of rural literacy rate among States/Union Territories as we saw in total literacy rate.
# - States/Union Territories that had higher urban literacy rate in 2001 have lesser percentage increse and those which had lesser urban literacy have worked hard on their literacy rate.

# <a class='anchor' id=5></a>
# 
# ## States vs Union Territories

# In[ ]:


temp_1 = df.groupby(by=['Category'])['Total - 2001'].mean().reset_index().T
temp_2 = df.groupby(by=['Category'])['Total - 2011'].mean().reset_index().T

temp_3 = df.groupby(by=['Category'])['Rural - 2001'].mean().reset_index().T
temp_4 = df.groupby(by=['Category'])['Rural - 2011'].mean().reset_index().T

temp_5 = df.groupby(by=['Category'])['Urban - 2001'].mean().reset_index().T
temp_6 = df.groupby(by=['Category'])['Urban - 2011'].mean().reset_index().T

frames = [temp_1, temp_2, temp_3, temp_4, temp_5, temp_6]
temp = pd.concat(frames)
loc = [0,1,3,5,7,9,11]
temp = temp.iloc[loc,:]
temp = temp.iloc[1:,:]
temp.reset_index(inplace=True)
temp.columns=['Category','State','Union Territory']


fig = go.Figure(data = [
    go.Bar(name='States', y=temp['Category'], x=temp['State'], orientation='h', marker_color='rgb(26, 118, 255)'),
    go.Bar(name='Union Territories', y=temp['Category'], x=temp['Union Territory'], orientation='h', marker_color='rgb(55, 83, 109)')
])
fig.update_layout(barmode='group')
fig.show()


# ### Average Literacy rate in union territories have always been greater than that of states in every category.

# <a class='anchor' id=6></a>
# 
# ## East, West, North, South

# ### Let's see how the literacy rates in four zones(East, West, North and South). For this, we have to first look at the distribution of the attributes so that we can decide how to aggregate the values.

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Box(y=df['Total - 2001'], name='Total-2001', boxpoints='suspectedoutliers'))
fig.add_trace(go.Box(y=df['Total - 2011'], name='Total-2011',boxpoints='suspectedoutliers'))
fig.add_trace(go.Box(y=df['Rural - 2001'], name='Rural-2001', boxpoints='suspectedoutliers'))
fig.add_trace(go.Box(y=df['Rural - 2011'], name='Rural-2011', boxpoints='suspectedoutliers'))
fig.add_trace(go.Box(y=df['Urban - 2001'], name='Urban-2001', boxpoints='suspectedoutliers'))
fig.add_trace(go.Box(y=df['Urban - 2011'], name='Urban-2011', boxpoints='suspectedoutliers'))
fig.show()


# #### We might have possible outliers in values of 'Urban-2001'. So, we'll go for 'median' to aggrigate the values.

# In[ ]:


East = ['Arunachal Pradesh','Assam','Jharkhand','West Bengal','Odisha',
        'Mizoram','Meghalaya','Manipur','Sikkim','Tripura','Nagaland']
West = ['Maharashtra','Gujarat','Goa']    
North = ['Uttar Pradesh','Bihar','Jammu & Kashmir','Rajasthan', 'Punjab','Haryana','Madhya Pradesh',
        'Chhattisgarh','Uttarakhand','NCT of Delhi','Tamil Nadu','Chandigarh','Himachal Pradesh',]
South = ['Andhra Pradesh','Karnataka','Kerala']

def zone_applier(x):
    if x in East :
        return 'East'
    elif x in West :
        return 'West'
    elif x in North :
        return 'North'
    else :
        return 'South'

State = df[df['Category']=='State']
State['Zone'] =State['States/ Union Territories'].apply(zone_applier)
State = State.groupby(by='Zone').agg('median')
State = State.iloc[:,:6]
State.reset_index(inplace=True)

State = State.T.reset_index()
State.columns = State.iloc[0,:]
State = State.iloc[1:,:]

fig = go.Figure(data=[
    go.Bar(name='East', x=State['Zone'], y=State['East']),
    go.Bar(name='West', x=State['Zone'], y=State['West']),
    go.Bar(name='North', x=State['Zone'], y=State['North']),
    go.Bar(name='South', x=State['Zone'], y=State['South'])
])
fig.update_layout(barmode='group', title='Avg. Literacy Rate by Zone:')
fig.show()

fig = make_subplots(rows=2,cols=2)
fig.add_trace(go.Bar(name='East', x=State['Zone'], y=State['East']), row=1,col=1)
fig.add_trace(go.Bar(name='West', x=State['Zone'], y=State['West']), row=1, col=2)
fig.add_trace(go.Bar(name='North', x=State['Zone'], y=State['North']), row=2, col=1)
fig.add_trace(go.Bar(name='South', x=State['Zone'], y=State['South']), row=2, col=2)
fig.show()


# ### Insights:
# - West Zone has highest average literacy rate in all sectors(rural, urban and total).
# - North zone has least average literacy rate in all sectors.
# - In urban sector both, east and west zones have almost equal literacy rate which is also the highest.

# <a class = 'anchor' id=7></a>
# 
# 
# ### Literacy Rate in each State/ Union Territory

# In[ ]:


df1 = pd.melt(df, id_vars='States/ Union Territories', value_vars=['Total - 2001', 'Total - 2011',
       'Rural - 2001', 'Rural - 2011', 'Urban - 2001', 'Urban - 2011',
       'Total - Per. Change', 'Rural - Per. Change', 'Urban - Per. Change'])
fig = px.bar(df1, 'variable', 'value', animation_frame='States/ Union Territories',
             color_discrete_sequence=['brown'])
fig.update_layout(title='Literacy Rate of each State/ Union Territory.')
fig.show()


# In[ ]:




