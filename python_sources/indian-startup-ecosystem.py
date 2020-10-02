#!/usr/bin/env python
# coding: utf-8

# # Overview

# This dataset is all about the startup ecosystem in India.It helps to know how companies are funded.Which industry is getting funds most frequently.Which city is getting funds for startups? Who are the important investors? <br>
# This dataset has funding information of the Indian startups from January 2015 to August 2017. It includes columns with the date funded, the city the startup is based out of, the names of the funders, and the amount invested (in USD).<br>
# Lets dive into it.

# # Loading all required Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
color=sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster

import squarify
from subprocess import check_output
print(check_output(['ls','../input']).decode('utf8'))

import warnings
warnings.filterwarnings('ignore')


# # Get the data

# In[3]:


df=pd.read_csv('../input/startup_funding.csv')


# # Glimpse Of Data

# ## Sample of Data

# In[ ]:


df.head()


# ## Columns and their types
# 

# In[ ]:


df.columns


# In[ ]:


df.dtypes


# Amount Should be Numerical here.We will change that to numerical in feature engineering step

# # Look For Missing Variales

# In[ ]:


total = df.isnull().sum().sort_values(ascending = False).reset_index()
total.columns=['Features','Total%']
total['Total%']=(total['Total%']*100)/df.shape[0]
total


# With more than 80% missing data I thing its wise to get rid of the 'Remarks' column.

# # Feature Enginnering

# Lets do some Feature enginnering with the dataset.<br><br>
# To get more out of this data we will do:<br><br>
#     1.Convert 'Date' type from object to datetime<br>
#     2.Extract 'Year','Month' feature from the Date column<br>
#     3.Create a new Column to get number of investors.<br>
#     4.Drop the 'Remarks','Date column'<br>
#     5.Convert The datype of AmountInUSD

# ## Convert Date datatype

# In[ ]:


df['Date'][df['Date']=='12/05.2015'] = '12/05/2015'
df['Date'][df['Date']=='13/04.2015'] = '13/04/2015'
df['Date'][df['Date']=='15/01.2015'] = '15/01/2015'
df['Date'][df['Date']=='22/01//2015'] = '22/01/2015'


# In[ ]:


df['Date']=pd.to_datetime(df['Date'],format='%d/%m/%Y')


# ## Extract Year and Month Columns

# #### Year Column

# In[ ]:


df['Year']=df['Date'].dt.year


# #### Month Column

# In[ ]:


df['Month']=df['Date'].dt.month


# ## Number Of Investors

# In[4]:


def num_invest(x):
    investors=x.split(',')
    return len(investors)

df.dropna(how='any',axis=0,subset=['InvestorsName'],inplace=True)
df['Num_Investors']=df['InvestorsName'].apply(lambda x:num_invest(x))
print(df.head())


# ## Drop The Remark and Date Column

# In[ ]:


df.drop(['Remarks','Date'],axis=1,inplace=True)


# ## Convert AmountInUSD from object type to float

# In[ ]:


df['AmountInUSD']=df['AmountInUSD'].apply(lambda x:float(str(x).replace(',','')))
df['AmountInUSD']=pd.to_numeric(df['AmountInUSD'])


# # Visualize The Data

# ## 1.Top StartUps with most Frequent Grants

# In[ ]:


startup=df['StartupName'].value_counts().sort_values(ascending=False).reset_index().head(10)
startup.columns=['Startup','Count']
data = [go.Bar(
            x=startup.Startup,
            y=startup.Count,
             opacity=0.6
    )]

py.iplot(data, filename='basic-bar')


# Swiggy seems to get most number of funds follwed by UrbanClap and Jugnoo

# ## 2.Top Startups in Terms Of Loan Amount

# In[ ]:


top=df.groupby(['StartupName']).sum().sort_values(by='AmountInUSD',ascending=False).reset_index().head(10)
data = [go.Bar(
            x=top.StartupName,
            y=top.AmountInUSD,
             opacity=0.6
    )]

py.iplot(data, filename='basic-bar')


# Paytm with 2Billion dollars as funded amount seems to be at the top follwed by Flipkart with more than 1.5 Billion dollars

# ## 3. Top Cities To Get Funds

# In[ ]:


city=df.CityLocation.value_counts().sort_values(ascending=False).reset_index().head(10)
city.columns=['City','Total']
fig,ax=plt.subplots(figsize=(15,10))
sns.barplot('City','Total',data=city,ax=ax)
plt.xlabel('City',fontsize=10)
plt.ylabel('StartUp Getting Funds',fontsize=10)
plt.title('Funding Across Cities',fontsize=15)
plt.show()


# Bengaluru,the sillion valley of India,lived upto to the expectations by leading the charts.

# ## 4.Visualize These Cities In Map

# In[ ]:


city_loc=pd.DataFrame({'City':['Bangalore','Mumbai','Chennai','New Delhi','Gurgaon','Pune','Noida','Hyderabad','Ahmedabad','Jaipur','Kolkata'],
                      'X':[12.9716,19.0760,13.0827,28.7041,28.4595,18.5204,28.5355,17.3850,23.0225,26.9124,22.5726],
                      'Y':[77.5946,72.8777,80.2707,77.1025,77.0266,73.8567,77.3910,78.4867,72.5714,75.7873,88.3639]})
city_loc


# In[ ]:



m = folium.Map(
    location=[20.5937,78.9629],
    tiles='Cartodb Positron',
    zoom_start=5
)

marker_cluster = MarkerCluster(
    name='City For StartUps',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(city_loc.shape[0]):
    location = city_loc.X.values[k], city_loc.Y.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = city_loc.City.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")

m


# With this Map I tried to locate the cities in India where most funding happened.

# In[ ]:


tech_bang=df[df['CityLocation']=='Bangalore']
tech_bang=tech_bang['IndustryVertical'].value_counts().sort_values(ascending=False).head(5).reset_index()
tech_bang.columns=['Industry','Count']
tech_bang
tech_mum=df[df['CityLocation']=='Mumbai']
tech_mum=tech_mum['IndustryVertical'].value_counts().sort_values(ascending=False).head(5).reset_index()
tech_mum.columns=['Industry','Count']
tech_new=df[df['CityLocation']=='New Delhi']
tech_new=tech_new['IndustryVertical'].value_counts().sort_values(ascending=False).head(5).reset_index()
tech_new.columns=['Industry','Count']


# In[ ]:


fig = {
  "data": [
    {
      "values": np.array((tech_bang['Count'] / tech_bang['Count'].sum())*100),
      "labels": tech_bang.Industry,
      "domain": {"x": [0, 0.48],'y':[0,0.48]},
      "name": "Bengalurelore",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },     
    {
      "values": np.array((tech_mum['Count'] / tech_mum['Count'].sum())*100),
      "labels": tech_mum.Industry,
      "text":"Mumbai",
      "textposition":"inside",
      "domain": {"x": [.52, 1],"y":[0,.48]},
      "name": "CO2 Emissions",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },
   {
      "values": np.array((tech_new['Count'] / tech_new['Count'].sum())*100),
      "labels": tech_new.Industry,
      "domain": {"x": [0.1, .88],'y':[0.52,1]},
      "name": "New Delhi",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    }],
  "layout": {
        "title":"Donut Chart For Industry in Top 3 Cities",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Bengaluru",
                "x": 0.15,
                "y": -0.08
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Mumbai",
                "x": 0.8,
                "y": -0.08
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "New Delhi",
                "x": 0.20,
                "y": 0.8
            }
        ]
    }
}
py.iplot(fig, filename='donut')


# This donut chart helps to visualize industry breakdown in top invested city.

# ## Top Industry To Get Frequent Funds

# In[ ]:


vertical=df['IndustryVertical'].value_counts().sort_values(ascending=False).reset_index().head(10)
vertical.columns=['vertical','Count']
tag = (np.array(vertical.vertical))
sizes = (np.array((vertical['Count'] / vertical['Count'].sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Top Industry To get Frequent Fund')
data = [trace]
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Activity Distribution")


# Consumer Internet and Technology seems to be top Industry to invest.

# In[ ]:


vertical=df['SubVertical'].value_counts().sort_values(ascending=False).reset_index().head(10)
vertical.columns=['vertical','Count']
tag = (np.array(vertical.vertical))
sizes = (np.array((vertical['Count'] / vertical['Count'].sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Top SubVerticals To get Frequent Fund')
data = [trace]
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Activity Distribution")


# Online Pharmacy and Food Delivery are two main subverticals.

# # Time Series Anlysis

# ## Funding For Top Industry Over the Months

# In[ ]:


df['YearMonth']=df['Year']*100+df['Month']


# In[ ]:


val=df['IndustryVertical'].value_counts().reset_index().head(10)
val.columns=['Industry','Count']
x=val.Industry
data=[]
for i in x:
    industry=df[df['IndustryVertical']==i]
    year_count=industry['Month'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Month','Count']
    trace = go.Scatter(
    x = year_count.Month,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')


# ## Type Of Funding Over Months

# In[ ]:


df['InvestmentType'][df['InvestmentType']=='SeedFunding']='Seed Funding'
df['InvestmentType'][df['InvestmentType']=='PrivateEquity']='Private Equity'
df['InvestmentType'][df['InvestmentType']=='Crowd funding']='Crowd Funding'

val=df['InvestmentType'].value_counts().reset_index().head(10)
val.columns=['Investment','Count']
x=val.Investment
data=[]
for i in x:
    industry=df[df['InvestmentType']==i]
    year_count=industry['Month'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Month','Count']
    trace = go.Scatter(
    x = year_count.Month,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')


# ## Funding Amount Over Months

# In[ ]:


total_invs=df.groupby('Month').sum().reset_index()
total_invs
trace = go.Scatter(
    x = total_invs.Month,
    y = total_invs.AmountInUSD
)

data = [trace]

py.iplot(data, filename='basic-line')


# ## Funding to Top cities Over Month

# In[ ]:


val=df['CityLocation'].value_counts().reset_index().head(10)
val.columns=['City','Count']
x=val.City
data=[]
for i in x:
    industry=df[df['CityLocation']==i]
    year_count=industry['Month'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Month','Count']
    trace = go.Scatter(
    x = year_count.Month,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')


# ## WordCloud to Show Investors Names

# In[ ]:


from wordcloud import WordCloud

names = df["InvestorsName"][~pd.isnull(df["InvestorsName"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Investor Names", fontsize=35)
plt.axis("off")
plt.show()


# ## Top Investors for Startups

# In[ ]:


df['InvestorsName'][df['InvestorsName'] == 'Undisclosed investors'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'undisclosed Investors'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'undisclosed investors'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'Undisclosed investor'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'Undisclosed Investor'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'Undisclosed'] = 'Undisclosed Investors'


# In[ ]:


investors = df['InvestorsName'].value_counts().head(10)
print(investors)
plt.figure(figsize=(15,8))
sns.barplot(investors.index, investors.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Investors Names', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Investors Names with number of funding", fontsize=16)
plt.show()


# ## Number of Investors per Project

# In[ ]:


investors = df['Num_Investors'].value_counts().head(10)
print(investors)
plt.figure(figsize=(15,8))
sns.barplot(investors.index, investors.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Number Of Investors', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Number Of Investors Per Funded Project", fontsize=16)
plt.show()

