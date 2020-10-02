#!/usr/bin/env python
# coding: utf-8

# # In this notebook we will analyze the dynamics of freedom, innovation and climate change
# <img src="https://images.unsplash.com/photo-1542067423794-1c9600c1efc2?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2233&q=80" width="500">

# In[ ]:


# modules we'll use
import pandas as pd #linear algebra
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)

from scipy.stats import norm
import plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.tools import FigureFactory as ff
from wordcloud import WordCloud,STOPWORDS
from PIL import Image
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")


# 
# # Summary Statistics for all the data

# In[ ]:


pathy = "../input/the-human-freedom-index/hfi_cc_2018.csv"
data = pd.read_csv(pathy)
data.describe()


# # Summary Statistics for 2016 data

# In[ ]:


data_2016 = data.loc[data['year'] == 2016]
data_2016.describe()


# In[ ]:


data1 = data['pf_score']
data1.describe()


# 
# Cleaning Null Data and Making columns more recognizable

# In[ ]:


data_2016 = data_2016.loc[:, (data_2016.isnull().sum(axis=0) <= 1242)]

# Rename the columns for a better undestanding
data_2016.rename(columns={"pf_score": "Personal Freedom Score",
                     "pf_ss": "Security",
                     "pf_expression": "Freedom_of_Expression",
                     "pf_religion": "Freedom of Religion",
                     "pf_rol_civil": "Civil Justice",
                     "ef_government": "Size of Government", 
                     "ef_legal": "Legal System and Property Rights",
                     "ef_money": "Sound Money",
                     "ef_trade": "Freedom to Trade Internationally"}, inplace=True)
data_2016.head()


# # Histograms displaying the distribution of personal, human, and economic freedom scores

# In[ ]:


sns.distplot(data_2016['Personal Freedom Score'], fit = norm, color = 'blue');


# Personal Freedom data appears bimodel and slightly skewed to the left. 

# In[ ]:


sns.distplot(data_2016['hf_score'], fit = norm, color = '#2980b9');


# Human Freedom data appears unimodel and skewed to the left.

# In[ ]:


sns.distplot(data_2016['ef_score'], fit = norm, color = '#3498db');


# Economic Freedom appears unimodel and skewed to the left

# # Correlation chart between multiple sections of personal and economic freedom

# In[ ]:


data2016_corr = data_2016[["Personal Freedom Score", "Civil Justice", "Security", "Freedom_of_Expression", "Freedom of Religion"]]
sns.heatmap(data2016_corr.corr(), square=True, cmap='Blues')
plt.show()
data2016_corr = data_2016[["ef_score", "Size of Government", "Legal System and Property Rights", "Sound Money", "Freedom to Trade Internationally"]]
sns.heatmap(data2016_corr.corr(), square=True, cmap='Blues')
plt.show()


# Based on our correlation chart, freedom of expression appears to be the best indicator of personal freedom, while freedom to trade internationally appears to be the best indicator of Economic Freedom.

# Lets create a correlation matrix to see the relationship between Freedom of Expression and Freedom to trade internationally

# In[ ]:


import plotly.figure_factory as ff
#prepare data
dataframe = data[data.year == 2016]
dataVar = dataframe.loc[:,["ef_trade", "pf_expression"]]
dataVar["index"] = np.arange(1,len(dataVar)+1)
#scatter matrix
fig = ff.create_scatterplotmatrix(dataVar, diag='box', index ='index', colormap = 'Portland', 
                                    colormap_type='cat',
                                   height = 900, width =900)
iplot(fig)


# # Personal Freedom vs. Human Freedom vs. Economic Freedom

# In[ ]:


x = data_2016['Personal Freedom Score'].values
y = data_2016['hf_score'].values
z = data_2016['ef_score'].values


trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color= 'blue',                # set color to an array/list of desired values
        colorscale='Jet',   # choose a colorscale
        opacity=0.5
    )
)

data = [trace1]
layout = go.Layout(
    showlegend=True,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# # Personal, Human, and Economic Freedom Rank For Top 10 Global Innovation Index Countries

# In[ ]:


df = pd.read_csv("../input/insead-global-innovation-index/INSEADGlobalinnovationIndex2018.csv")
df.head(10)


# These are the top 10 economies according to the Global Innovation Index 

# In[ ]:


pathy = "../input/the-human-freedom-index/hfi_cc_2018.csv"
data = pd.read_csv(pathy)
hdi = ['Switzerland', 'Netherlands', 'Sweden', 'United Kingdom', 'Ireland', 'Singapore', 'United States of America', 'Finland', 'Denmark','Germany']
df_innovation = df[df.Economy.isin(hdi)]
data_innovation = data[data.countries.isin(hdi)]
# import graph objects as "go"
import plotly.graph_objs as go

x = df_innovation.Economy

trace1 = {
  'x': x,
  'y': data_innovation.ef_rank,
  'name': 'Economic Freedom Score',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': data_innovation.hf_rank,
  'name': 'Human Freedom Score',
  'type': 'bar'
};
trace3 = {
  'x': x,
  'y': data_innovation['pf_rank'],
  'name': 'Personal Freedom Score',
  'type': 'bar'
};
data = [trace1, trace2, trace3];
layout = {
  'xaxis': {'title': ' Countries in 2016'},
  'barmode': 'relative',
  'title': 'Personal, Human, and Economic Freedom Rank For Top 10 Global Innovation Index Countries'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# The data above shows that 8 out of the top 10 GII countries are ranked top 10 in Human Freedom, 7 are ranked top 10 in Personal Freedom Score and 4 are ranked top 10 in Economic Freedom.

# # Climate Change and Innovation

# One potential antidote for climate change is innovation. Lets see the Global Innovation Index Rankings of the top 5 GHG emitting countries.

# In[ ]:


top = ['United States of America', 'China', 'India', 'Russian Federation', 'Japan']
df_power = df[df.Economy.isin(top)]
df_power.describe()


# The top 5 cleanest countries according to www.conserve-energy-future.com are: Iceland, Switzerland, Costa Rica, Sweden, and Norway. Lets take a look at their Global Innovation Index rankings.

# In[ ]:


clean = ["Iceland", "Switzerland", "Costa Rica", "Sweden", "Norway"]
df_clean = df[df.Economy.isin(clean)]
df_clean.describe()


# We can clearly see that the top 5 climate friendly countries are on average ranked higher in innovation than the top 5 GHG emitting countries. 

# # Human, Economic, and Personal Freedom Score For Top 10 Green House Gas Emitting Countries

# <img src="https://images.unsplash.com/photo-1494807081385-2193b6e24400?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=3150&q=80" width="500">

# In[ ]:


human = pd.read_csv('../input/the-human-freedom-index/hfi_cc_2018.csv')
power = ['United States', 'China', 'India', 'Russia', 'Japan', 'Germany', 'South Korea', 'Iran', 'Canada', 'Saudi Arabia']
data_power = data_2016[data_2016.countries.isin(power)]


# Climate change is a surmounting problem plaguing society today. The biggest factor contributing to climate change is the increase in green house gas emissions. According to "ucsusa.org", the top 10 green house gas emitting countries for 2015 are: United States, China, India, Russia, Japan, Germany, South Korea, Iran, Canada, and Saudi Arabia. Lets take a look at the freedom ranks for these countries.

# In[ ]:


# import graph objects as "go"
import plotly.graph_objs as go

x = data_power.countries

trace1 = {
  'x': x,
  'y': data_power.ef_rank,
  'name': 'Economic Freedom Score',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': data_power.hf_rank,
  'name': 'Human Freedom Score',
  'type': 'bar'
};
trace3 = {
  'x': x,
  'y': data_power['pf_rank'],
  'name': 'Personal Freedom Score',
  'type': 'bar'
};
data = [trace1, trace2, trace3];
layout = {
  'xaxis': {'title': ' Countries in 2016'},
  'barmode': 'relative',
  'title': 'Personal, Human, and Economic Freedom Rank For Top 10 Green House Gas Emitting Countries'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# 2016 Human Freedom Rank
# * **Canada**: 5              
# * China: 135                    
# * Germany: 13                
# * India: 110             
# * Iran: 153      
# * Japan: 31          
# * Russia: 119  
# * Saudi Arabia: 146    
# * United States: 17 
# 
# 
# 2016 Economic Freedom Rank
# * **Canada**: 10
# * China: 108
# * Germany: 20
# * India: 96
# * Iran: 130
# * Japan: 41
# * Russia: 87
# * Saudi Arabia: 103
# * **United States:** 6
# 
# 
# 2016 Personal Freedom Rank
# * Canada: 12
# * China: 141
# * **Germany**: 9
# * India: 112
# * Iran: 154
# * Japan: 29
# * Russia: 130
# * Saudi Arabia: 155
# * United States: 28
# 
# 

# # Global Perspective of Regulation and Innovation

# In[ ]:


from mpl_toolkits.basemap import Basemap
concap = pd.read_csv('../input/world-capitals-gps/concap.csv')
data18 = pd.read_csv('../input/the-human-freedom-index/hfi_cc_2018.csv')
data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],         data18,left_on='CountryName',right_on='countries')
def mapWorld():
    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,            llcrnrlon=-110,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90,91.,30.))
    m.drawmeridians(np.arange(-90,90.,60.))
    lat = data_full['CapitalLatitude'].values
    lon = data_full['CapitalLongitude'].values
    a_1 = data_full['ef_regulation_business'].values
    #a_2 = data_full['Economy (GDP per Capita)'].values
    #300*a_2
    m.scatter(lon, lat, latlon=True,c=a_1,s=500,linewidth=1,edgecolors='black',cmap='Blues', alpha=1)
    
    #m.fillcontinents(color='#FFFFFF',lake_color='#FFFFFF',alpha=0.3)
    cbar = m.colorbar()
    cbar.set_label('Business Regulation',fontsize=30)
    #plt.clim(20000, 100000)
    plt.title("Business Regulation (score)", fontsize=30)
    plt.show()
plt.figure(figsize=(30,30))
mapWorld()


# In[ ]:


from mpl_toolkits.basemap import Basemap
concap = pd.read_csv('../input/world-capitals-gps/concap.csv')
data18 = pd.read_csv("../input/insead-global-innovation-index/INSEADGlobalinnovationIndex2018.csv")
data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],         data18,left_on='CountryName',right_on='Economy')
def mapWorld():
    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,            llcrnrlon=-110,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90,91.,30.))
    m.drawmeridians(np.arange(-90,90.,60.))
    lat = data_full['CapitalLatitude'].values
    lon = data_full['CapitalLongitude'].values
    a_1 = data_full['Score'].values
    #a_2 = data_full['Economy (GDP per Capita)'].values
    #300*a_2
    m.scatter(lon, lat, latlon=True,c=a_1,s=500,linewidth=1,edgecolors='black',cmap='Reds', alpha=1)
    
    #m.fillcontinents(color='#FFFFFF',lake_color='#FFFFFF',alpha=0.3)
    cbar = m.colorbar()
    cbar.set_label('Global Innovation Score',fontsize=30)
    #plt.clim(20000, 100000)
    plt.title("Global Innovation Index (score)", fontsize=30)
    plt.show()
plt.figure(figsize=(30,30))
mapWorld()

