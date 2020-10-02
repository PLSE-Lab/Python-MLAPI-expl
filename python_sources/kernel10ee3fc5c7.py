#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:





# In[ ]:


kiva_loan=pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')


# In[ ]:


kiva_loan.head()


# In[ ]:


kiva_loan.describe()


# The median loan fuunded amount is $450

# ### Which countries have the largest loan amount?

# In[ ]:


countries= kiva_loan.groupby(['country'])['loan_amount'].sum().sort_values(ascending=False).reset_index()


# In[ ]:


countries.to_excel('countries.xlsx')


# In[ ]:


countries.head(20)


# The country with the largest loan amount is Philippines with $55342225.0

# ### Which countries have the smallest loan amount?

# In[ ]:


countries= kiva_loan.groupby(['country'])['loan_amount'].sum().sort_values(ascending=True).reset_index()


# In[ ]:


countries.to_excel('countries.xlsx')


# In[ ]:


countries.head(20)


# The country with the smallest loan amount is Guam with $4300

# ### How many countries are in this dataset?

# In[ ]:


kiva_loan['country'].unique


# In[ ]:


kiva_loan['country'].nunique()


# In[ ]:


kiva_loan['country'].value_counts().reset_index()


# 
# ## Which sector are most of the borrowers in from India?
# 

# In[ ]:


India['sector'].value_counts().reset_index()


# In[ ]:


India = kiva_loan[kiva_loan['country'] == 'India']


# In[ ]:


India.groupby(['sector'])['loan_amount'].sum().reset_index()


# 
# ### Select all the data points from India, group by region and compute the total loan amount for each region.
# 

# In[ ]:


regions=India.groupby('region')['loan_amount'].sum().sort_values(ascending=False).reset_index()


# In[ ]:


regions.head(10)


# Jeypore,Odish is the largest borrower from Kiva in India

# ### Create a pivot table with region and sector as the columns and count as the aggregate function
# 

# In[ ]:


India.pivot_table(columns=['region','sector'],aggfunc='sum')


# In[ ]:


themes = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')


# In[ ]:


merged_data = kiva_loan.merge(themes, on='id')


# In[ ]:


merged_data.head(20)


# In[ ]:


merged_data['Loan Theme Type'].unique()


# In[ ]:


themes_data = merged_data.groupby('Loan Theme Type')['loan_amount'].sum().sort_values(ascending=False).reset_index()


# In[ ]:


themes_data.head(20)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


md=merged_data.head(500)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


x=md['sector']
y=md['loan_amount']
bar_width=.7
plt.xticks(rotation=65)

plt.title('loan Amount by sector')
plt.xlabel('Sector')
plt.ylabel('loan Amount')

plt.bar(x, y, bar_width, color ='blue')


# In[ ]:


x=md['term_in_months']
y=md['loan_amount']

plt.title('loan Amount by lenders')
plt.xlabel('term in months')
plt.ylabel('loan amount')

plt.scatter(x, y)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


# In[ ]:


ux=md['loan_amount']

plt.hist(ux, bins =10)
plt.show()


# In[ ]:


import seaborn as sns


# In[ ]:


merged_region =merged_data.groupby ('region')['loan_amount'].sum().sort_values(ascending=False).reset_index()


# In[ ]:


India


# In[ ]:


merged_data['region'].unique()


# In[ ]:


local_region=merged_data[merged_data['country']=='India']
local_region.head(20)


# In[ ]:


local =local_region.groupby('region')['loan_amount'].sum().sort_values(ascending=False).reset_index()


# In[ ]:


local.head(20)


# In[ ]:


plt.figure(figsize=(20,6))
plt.xticks(rotation=45)
plt.title('Loan Amount by Region:Top 20')
sns.barplot(x='region', y='loan_amount', data=local.head(20))


# Jeypore,Odish has the highest loan amount from kiava

# In[ ]:


Jeypore=local_region[local_region['region']=='Jeypore']
Jeypore.head()


# In[ ]:


correlation=local_region.corr()


# In[ ]:


correlation=correlation.drop(['partner_id'],axis=1)


# In[ ]:


correlation


# In[ ]:


plt.figure(figsize=(20,5))
plt.yticks(rotation=90)
sns.heatmap(correlation,cmap='rainbow')


# In[ ]:


plt.figure(figsize=(25,8))
sns.scatterplot(x='lender_count',y='loan_amount', data =India, hue='loan_amount')


# In[ ]:


def clean_gender(gender):
    gender=str(gender)
    if gender.startswith('f'):
        gender='female'
    return gender


# In[ ]:


local_region['borrower_genders']=local_region['borrower_genders'].apply(clean_gender)


# In[ ]:


activity=local_region.groupby (['borrower_genders', 'sector'])['loan_amount'].sum().sort_values(ascending=False).reset_index()


# In[ ]:


plt.figure(figsize=(20,5))
plt.xticks(rotation=35)
sns.barplot(x='sector',y='loan_amount', hue='borrower_genders',data=activity.head(25))


# In[ ]:


local.head(20)


# In[ ]:


import plotly.graph_objects as go
labels=activity['sector']
values=activity ['loan_amount']
fig=go.Figure(data=[
    go.Pie(labels=labels, values=values)
])

fig.show()


# In[ ]:


import plotly.graph_objects as go
labels = activity['sector']
values =activity['loan_amount']
fig=go.Figure(data=[go.Pie(labels=labels, values=values, hole=.7)])

fig.show()import plotly.express as px
fig = px.bar(activity, x='sector', y='loan_amount')
fig.show()


# In[ ]:


import plotly.express as px
fig = px.bar(activity, x='sector', y='loan_amount')
fig.show()


# In[ ]:


import plotly.express as px
fig = px.bar(activity, x='sector', y='loan_amount',
             hover_data=['loan_amount', 'borrower_genders'], color='loan_amount',
             labels={'loan_amount':'Loan Amount'}, height=600)
fig.show()


# In[ ]:


import plotly.express as px
fig = px.scatter(local_region, x="loan_amount", y="lender_count", color="sector",
                 size='lender_count', hover_data=['funded_amount'])
fig.show()


# In[ ]:


def create_year(date):
    year =pd.DatetimeIndex(date).year
    return year


# In[ ]:


local_region['year'] =create_year(local_region['date'])


# In[ ]:


def create_day(date):
    month =pd.DatetimeIndex(date).day
    return month


# In[ ]:


local_region['day_of_month'] =create_day(local_region['date'])


# In[ ]:


local_region.head(10)


# In[ ]:


import plotly.express as px
px.scatter(local_region, x='loan_amount', y='lender_count', animation_frame='day_of_month', animation_group='activity', 
           size="loan_amount", color="sector", hover_name="loan_amount",
           log_x=True, size_max=55, range_x=[25,50000], range_y=[0,200])


# In[ ]:


loan_themes_by_region = pd.read_csv('loan_themes_by_region.csv')


# In[ ]:


loan_themes_by_India = loan_themes_by_region[loan_themes_by_region['country']=='India']


# In[ ]:


loan_themes_by_India.head()


# In[ ]:


px.set_mapbox_access_token('pk.eyJ1IjoibXdpdGlkZXJyaWNrIiwiYSI6ImNrMHJyZ2w0bTA4dGwzbW85OHdwb2RoNjcifQ.YkzAW4L2Z6TgZKW65_uyEQ')
fig = px.scatter_mapbox(loan_themes_by_India, lat="lat", lon="lon", color="region", size="amount",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15)
fig.show()


# In[ ]:


import plotly.graph_objects as go

mapbox_access_token ='pk.eyJ1IjoibXdpdGlkZXJyaWNrIiwiYSI6ImNrMHJyZ2w0bTA4dGwzbW85OHdwb2RoNjcifQ.YkzAW4L2Z6TgZKW65_uyEQ'
size = loan_themes_by_India['amount']/30000
fig = go.Figure(go.Scattermapbox(
        lat=loan_themes_by_India['lat'],
        lon=loan_themes_by_India['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=size,
            color='red',
        ),
        text= loan_themes_by_India['region'],
    
    ))
fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=0.1769,
            lon=37.9083
        ),
        pitch=0,
        zoom=5
    )
)

fig.show()


# In[ ]:


import os
for dirname, -, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:




