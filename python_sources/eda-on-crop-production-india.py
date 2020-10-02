#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np ## Linear Algebra
import pandas as pd ## To work with data
import plotly.express as px ## Visualization
import plotly.graph_objects as go ## Visualization
import matplotlib.pyplot as plt ## Visualization
import plotly as py ## Visuaization
from plotly import tools ## Visualization
import os


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = os.path.join(dirname, filename)

df = pd.read_csv(path)


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(inplace=True) # looking at the data we can drop the null values as they are less in number.


# In[ ]:


df.head()


# ## EDA

# #### Overall Crop Production By state

# In[ ]:


temp = df.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production')
px.bar(temp, 'State_Name', 'Production')


# #### From above graph we can see that :
# - Kerala is the highest crops producing state overall. It had produced more than 500% crop than it's runner up state Andhra Pradesh.
# - Top 3 crop producing states are from south India, which put together leave no space to compare rest states.

# ### Productivity of different states

# In[ ]:


temp = df.groupby('State_Name')['Area', 'Production'].sum().reset_index()
temp['Production_Per_Unit_Area'] = temp['Production']/temp['Area']
temp = temp.sort_values(by='Production_Per_Unit_Area')
px.bar(temp, 'State_Name', 'Production_Per_Unit_Area', color='Production_Per_Unit_Area', )


# #### Above graph tells us that :
# - Kerala is the most productive state when we compare in terms of production by area.
# - We see Andaman and nikobar islands, Goa, Panduchery and many other states which are low in overall production, have high productivity when we compare with the crop areas.

#  ### Overall production through years

# In[ ]:


temp = df.groupby(by='Crop_Year')['Production'].sum().reset_index()
px.line(temp, 'Crop_Year', 'Production')


# ### Average Crop Area through years

# In[ ]:


temp = df.groupby(by='Crop_Year')['Area'].mean().reset_index()
px.scatter(temp, 'Crop_Year', 'Area', color='Area', size='Area')


# #### In Above Graph we can see that :
# - Average Crop Area has decresed over the years.
# - We had the lowest Average Crop area in Years 2002 and 2003. (We have very comparitively very less data of year 2015 so, we'll not consider that)

# ### Productivity in different states.

# In[ ]:


temp = df.groupby('State_Name')['Area', 'Production'].sum().reset_index()
temp['Production_Per_Unit_Area'] = temp['Production']/temp['Area']
temp = temp.sort_values(by='Production_Per_Unit_Area')
px.bar(temp, 'State_Name', 'Production_Per_Unit_Area', color='Production_Per_Unit_Area')


# ### Most and Least crop producing districts

# In[ ]:


fig = py.subplots.make_subplots(rows=1,cols=2,
                    subplot_titles=('Highest crop producing districts', 'Least overall crop producing districts'))

temp = df.groupby(by='District_Name')['Production'].sum().reset_index().sort_values(by='Production')
temp1 = temp.tail()
trace1 = go.Bar(x= temp1['District_Name'], y=temp1['Production'])

temp1=temp.head()
trace2 = go.Bar(x= temp1['District_Name'], y=temp1['Production'])

fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)
fig.show()
del temp,temp1


# #### MOST PRODUCED CROPS

# In[ ]:


temp = df.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production')
px.bar(temp.tail(), 'Crop', 'Production')


# #### NONE PRODUCED CROPS

# In[ ]:


temp[temp['Production']==0]


# #### Coconot is the most produced crop and kerala is the most crop producing state. Let's explore these two a little more

# ### COCONUT

# In[ ]:


coconut = df[df['Crop']=='Coconut ']

fig = py.subplots.make_subplots(rows=1,cols=2,
                               subplot_titles=('Coconut production in different states', 'Coconut crop area in states'))

temp = coconut.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production')
trace0 = go.Bar(x=temp['State_Name'], y=temp['Production'])

temp = coconut.groupby(by='State_Name',)['Area'].mean().reset_index().sort_values(by='Area')
trace1 = go.Bar(x=temp['State_Name'], y=temp['Area'])

fig.append_trace(trace0, 1,1)
fig.append_trace(trace1, 1,2)
fig.show()


# In[ ]:


temp = coconut.groupby(by='Crop_Year')['Production'].sum().reset_index()
px.line(temp, 'Crop_Year', 'Production', title='Coconut production over the years')


# ### KERALA

# In[ ]:


kerala = df[df['State_Name']=='Kerala']
temp = kerala.groupby(by='Crop')['Production'].mean().reset_index().sort_values(by='Production')
px.bar(temp, 'Crop', 'Production', title = 'Avg. Crop Production')


# #### Aperarently coconut tooks over and then we can't see anything else.

# In[ ]:


kerala = kerala[~(kerala['Crop']=='Coconut ')]
temp = kerala.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production')
px.bar(temp, 'Crop', 'Production', title='AVG. Crop Production excluding coconut')


# ### From now, we'll do analysis exludiing kerala and coconut

# In[ ]:


df1 = df[~((df['State_Name']=='Kerala') | (df['Crop']=='Coconut '))]


# In[ ]:


temp=df1.groupby('Crop')['Production'].sum().reset_index().sort_values(by='Production')
px.bar(temp, 'Crop', 'Production', title='Overall production of crops')


# In[ ]:




