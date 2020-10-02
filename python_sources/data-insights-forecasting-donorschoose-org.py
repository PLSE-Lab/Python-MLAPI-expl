#!/usr/bin/env python
# coding: utf-8

# # **This kernel mainly focuses on how we will do the data pre-processing, use different visualization libraries and find out the deep insights from the dataset. We will also be looking into Time-Series Forecasting using fbprophet package**
# 
# ### So lets get started
# 
# ## Importing the Libraries
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # We'll be serially working with all the datasets, first lets move onto the Donors.csv file

# In[ ]:


df=pd.read_csv("../input/Donors.csv")
df.head()


# ### First of all lets find out if there are any null values, or something, once we get through with the pre-processing we'll go in for the Visualization part

# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# ### What we can do here is, we can fill up the missing city with 'Unknown' value. And the Zip? We're not going to get much info focusing on the Zip

# In[ ]:


df['Donor City'].fillna('Unknown',inplace=True) # Please see we're going to drop unknown when we visualize maximum cities


# In[ ]:


df['Donor State'].nunique()


# ### Too many cities, right? We'll find  out the top 20 cities first then look into states and then maybe we can sort out the cities based on each state

# In[ ]:


cnt_srs = df.ix[df['Donor City']!='Unknown']['Donor City'].value_counts().head(20) # We are dropping cities with unknown values

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="green",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Top 20 Cities with Highest Donors'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Donors")  


cnt_srs = df['Donor State'].value_counts().head(20) # We are dropping cities with unknown values

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="green",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Top 20 States with Highest Donors'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Donors")  


# In[ ]:


df.head(1)


# ## Let us now find out how many donors are teachers and how many are not!

# In[ ]:


df['Donor Is Teacher'].unique()


# ### Lets replace all the "No's" with "Non-Teachers" and "Yes's" with "Teachers"

# In[ ]:


df['Donor Is Teacher'] = df['Donor Is Teacher'].map({'No': 'Non-Teachers', 'Yes': 'Teachers'})


# In[ ]:


temp_series = df['Donor Is Teacher'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title="Donor's Categorical Distribution (Teachers or Non-Teachers)",
    width=600,
    height=600,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Donor")


# In[ ]:


city_df=df.groupby('Donor City')


# # Now lets move on the Donations.csv file

# In[ ]:


df2=pd.read_csv('../input/Donations.csv')
df2.head()


# # **Let's future forecast the amount of donation that might happen**

# In[ ]:


df2.dtypes


# In[ ]:


df2['Donation Received Date']=pd.to_datetime(df2['Donation Received Date'])


# In[ ]:


df2['Receiving_Date'] =pd.DatetimeIndex(df2['Donation Received Date']).normalize()


# In[ ]:


df2.head(1)


# In[ ]:


df2.shape


# We will be taking the last 1000 rows for visualization and prediction

# In[ ]:


df2_dash=df2.tail(1000)


# In[ ]:


df_ts=df2_dash[['Donation Amount','Receiving_Date']]


# In[ ]:


df_ts.index=df_ts['Receiving_Date']


# In[ ]:


df_ts['Donation Amount'].plot(figsize=(15,6), color="green")
plt.xlabel('Year')
plt.ylabel('Donation Amount')
plt.title("Donation Amount Time-Series Visualization")
plt.show()


# # **Future Forecasting of  Donations**

# In[ ]:


from fbprophet import Prophet
sns.set(font_scale=1) 
df_date_index = df2_dash[['Receiving_Date','Donation Amount']]
df_date_index = df_date_index.set_index('Receiving_Date')
df_prophet = df_date_index.copy()
df_prophet.reset_index(drop=False,inplace=True)
df_prophet.columns = ['ds','y']

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=30,freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)


# In[ ]:


m.plot_components(forecast);


# # Here we have our next file Resources.csv

# In[ ]:


df3=pd.read_csv('../input/Resources.csv')
df3.head()


# In[ ]:


df3.isnull().sum()


# In[ ]:


df3.shape


# ## We'll drop the rows having null values, just to visualize which vendor had the highest sales

# In[ ]:


df3=df3.dropna() #The dataset is huge, filling in with Imputers was taking a lot of time


# In[ ]:


df3.shape


# In[ ]:


df3.isnull().any()


# In[ ]:


df3.head(1)


# In[ ]:


df4=df3.groupby(['Resource Vendor Name'],as_index=False)['Resource Quantity','Resource Unit Price'].agg('sum')


# In[ ]:


df4.head(1)


# In[ ]:


df4['revenue']=df4['Resource Quantity']*df4['Resource Unit Price']


# In[ ]:


df4.head(1)


# In[ ]:


df4=df4.sort_values('Resource Quantity',ascending=False)


# In[ ]:


df4.shape


# In[ ]:


type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

sns.set(rc={'figure.figsize':(11.7,8.27)}) 
ax= sns.barplot(x=df4['Resource Vendor Name'].head(10), y=df4['Resource Quantity'].head(10),palette = type_colors)
plt.xlabel('Name of the  Vendor',fontsize=20.5)
plt.ylabel('No. of Units',fontsize=20.5)
ax.tick_params(labelsize=15)
plt.title('Top 10 Vendors with Highest No. of Resources',fontsize=20.5)
for item in ax.get_xticklabels():
    item.set_rotation(90)


# # Importing the teachers dataset

# In[ ]:


df4=pd.read_csv('../input/Teachers.csv')
df4.head(1)


# In[ ]:


df4['Teacher Prefix'].unique()


# In[ ]:


df4.dtypes


# In[ ]:


df4['Teacher First Project Posted Date']=pd.to_datetime(df4['Teacher First Project Posted Date'])


# In[ ]:


df4.head(1)


# In[ ]:


df4.isnull().sum()


# In[ ]:


df4['year_of_project']=df4['Teacher First Project Posted Date'].dt.year


# In[ ]:


df4['year_of_project'].unique()


# In[ ]:


cnt_srs = df4['year_of_project'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="blue",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Year-wise Projects Posted'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Proj")  


# In[ ]:


df5=pd.read_csv('../input/Projects.csv')
df5.head()


# # Lets have a look at the project categories

# In[ ]:


temp_series = df5['Project Resource Category'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Project Category Distribution',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Projects")


# # Lets have a look at how many projects' current are funded

# In[ ]:


cnt_srs = df5['Project Current Status'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Funded & Non-Funded Projects'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="funding")


# In[ ]:


df5_fund_category=df5.groupby('Project Resource Category',as_index=False)['Project Cost'].agg('sum')


# In[ ]:


df5_fund_category.shape


# # Lets have a look which category has the highest cost

# In[ ]:


fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df5_fund_category["Project Resource Category"], y=df5_fund_category['Project Cost'], data=df5_fund_category)
f.set_xlabel("Category of Project",fontsize=15)
f.set_ylabel("Project Cost",fontsize=15)
f.set_title('Project Category-wise Cost')
for item in f.get_xticklabels():
    item.set_rotation(90)


# ### So technical projects have the highest cost

# # Lets have a look which category of projects get funding maximum time

# In[ ]:


df_funding=df5.ix[df5['Project Current Status']=="Fully Funded"]


# In[ ]:


df_funding.head()


# In[ ]:


cnt_srs = df_funding['Project Resource Category'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Funded Project Resources Category-wise'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="funding_proj")


# ### So technical and supply kind of project resources are funded maximum time

# In[ ]:


from PIL import Image
from wordcloud import WordCloud
import requests


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df5['Project Title'])


# In[ ]:


df6=pd.read_csv('../input/Schools.csv')
df6.head()


# In[ ]:


df6.shape


# # Lets have a look at the School Metro Type

# In[ ]:


temp_series = df6['School Metro Type'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='School Metro Type Distribution',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="School")


# In[ ]:


df6['School Name'].nunique()


# In[ ]:


df6=df6.sort_values('School Percentage Free Lunch',ascending=False)


# # Lets see how many schools give 100% Free Lunch

# In[ ]:


cnt_srs = df6['School Percentage Free Lunch'].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        reversescale = True,
        color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                    ),
    ),
)

layout = dict(
    title='No. of schools giving a %  of Free Lunch',
     yaxis=dict(
        title='% of Free Lunch')
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Lunch")


# # So we that we come to the analysis of this dataset. Please feel free to add so

# In[ ]:




