#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Import libraries
import pandas as pd # primary data structure library
import numpy as np # useful for many scientific computing in Python
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches # needed for waffle Charts
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import random
import math
import time
import datetime
from PIL import Image # converting images into arrays
import matplotlib.patches as mpatches # needed for waffle Charts
import folium # to create maps
from pylab import *
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
mpl.style.use('seaborn') #  for ggplot-like style
warnings.filterwarnings("ignore")


# # Downloading Data 

# This data is collected from repository for the 2019 Novel Coronavirus Visual Dashboard operated by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE).https://github.com/CSSEGISandData/COVID-19

# In[ ]:


### Select yesterday date as report provide figures for the day before
from datetime import date, timedelta
file_date = str((date.today() - timedelta(days = 1)).strftime('%m-%d-%Y'))
### Select data path
github_dir_path = '/kaggle/input/covid19/'
file_path = github_dir_path  + file_date + '.csv'
### Import data
df = pd.read_csv(file_path, error_bad_lines=False)


# # Exploring Data

# view the dataframe

# In[ ]:


df.head()


# dataframe dimensions

# In[ ]:


print ('dataframe dimensions:', df.shape)


# # Prepping Data 

# In[ ]:


### Check if the dataframe contains NaN values
df.isna().any()


# In[ ]:


# Countries affected
countries = df['Country_Region'].unique().tolist()
print("\nTotal countries affected by virus: ",len(countries))
print(countries)


# In[ ]:


### Replace NaN values by 0
df.fillna(0, inplace=True)
### Remove columns
df_countries = df.drop(['FIPS','Lat','Long_','Admin2','Province_State','Last_Update','Combined_Key'], axis=1)
### Rename the columns so that they make sense
df_countries.rename (columns = {'Country_Region':'Country'}, inplace = True)
### Re-order Columns
df_countries = df_countries[['Country','Confirmed','Active','Recovered','Deaths']]
### Group datas by Country
df_countries_grouped=df_countries.groupby('Country').sum()
### Set the country name as index
df_countries.set_index('Country', inplace=True)


# In[ ]:


# view the final dataframe
df_countries_grouped.head()


# In[ ]:


print ('data dimensions:', df_countries_grouped.shape)


# In[ ]:


confirmed_sum = df.Confirmed.sum()
active_sum=df.Active.sum()
recovered_sum = df.Recovered.sum()
death_sum = df.Deaths.sum()


# # Visualizing Data

# plotting configurations

# In[ ]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rc('figure', dpi=300)
plt.rc('savefig', dpi=300)
fig_size = (12,6)
big_fig_size = (18,8)


# Creating Dataframe with Totals

# In[ ]:


df_tot = pd.DataFrame(columns=['Confirmed', 'Active', 'Recovered','Deaths'])
# Append rows in Empty Dataframe by adding dictionaries
df_tot = df_tot.append({'Confirmed': confirmed_sum, 'Active': active_sum, 'Recovered': recovered_sum,'Deaths': death_sum}, ignore_index=True)


# # plot Total Cases

# In[ ]:


df_tot.plot(kind='barh', figsize=(14, 6),stacked=True, 
             alpha=0.5, 
            color=['blue', 'orange', 'green', 'red']) 
plt.legend(['Number of Confirmed Cases= '+ str(confirmed_sum), 'Number of Active Cases= '+ str(active_sum), 'Number of Recovery Cases='+ str(recovered_sum), 'Number of Death Cases='+ str(death_sum)], loc='upper right', fontsize=10)
plt.show()


# # Treemap charts<a id="6"></a>
# 
# visualize hierarchical data using nested rectangles. Click on one sector to zoom in/out, which also displays a pathbar in the upper-left corner of your treemap. To zoom out you can use the path bar as well.
# 

# In[ ]:


fig = px.treemap(df.sort_values(by='Confirmed', ascending=False ).reset_index(drop=True), 
                 path=["Country_Region"], values="Confirmed", 
                 title='Number of Confirmed Cases',color='Confirmed')
fig.show()


# In[ ]:


fig = px.treemap(df.sort_values(by='Active', ascending=False).reset_index(drop=True), 
                 path=["Country_Region"], values="Active", 
                 title='Number of Active Cases',color='Active')
fig.show()


# In[ ]:


fig = px.treemap(df.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 
                 path=["Country_Region"], values="Recovered", 
                 title='Number of Recovered Cases',color='Recovered')
fig.show()


# In[ ]:


fig = px.treemap(df.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 
                 path=["Country_Region"], values="Deaths", 
                 title='Number of Deaths Cases',color='Deaths')
fig.show()


# ## Countries with most Confirmed cases

# In[ ]:


df_countries_grouped = df_countries_grouped.sort_values('Confirmed', ascending=False)
df_countries_grouped.head(10).style.background_gradient(cmap='Blues')


# ## Countries with most Active cases

# In[ ]:


df_countries_grouped = df_countries_grouped.sort_values('Active', ascending=False)
df_countries_grouped.head(10).style.background_gradient(cmap='Blues')


# ## Countries with most Recovered cases

# In[ ]:


df_countries_grouped = df_countries_grouped.sort_values('Recovered', ascending=False)
df_countries_grouped.head(10).style.background_gradient(cmap='Blues')


# ## Countries with most Deaths cases

# In[ ]:


df_countries_grouped = df_countries_grouped.sort_values('Deaths', ascending=False)
df_countries_grouped.head(10).style.background_gradient(cmap='Blues')


# ## Countries with less Deaths cases

# In[ ]:


df_countries_grouped = df_countries_grouped.sort_values(['Deaths','Recovered','Confirmed'], ascending=False)
df_countries_grouped.tail(10).style.background_gradient(cmap='Blues')


# ## Countries with less Recovered cases

# In[ ]:


df_countries_grouped = df_countries_grouped.sort_values(['Recovered','Deaths'], ascending=True)
df_countries_grouped.head(10).style.background_gradient(cmap='Blues')


# # Unstacked Histogram

# In[ ]:


### Sort the top countries by Confirmed cases
df_countries_grouped.sort_values(['Confirmed'], ascending=False, axis=0, inplace=True)
# get the top entries
df_countries_top = df_countries_grouped.head(10)
### Show Unstacked Histogram
df_countries_top.plot(kind='bar', figsize=(18, 9), stacked=False, 
             alpha=0.5,
            color=['blue', 'orange', 'green', 'red'])
plt.title('Number of Cases by Country for the 10 top Countries')
plt.ylabel('Number of Cases')
plt.xlabel('Countries')
plt.show()


# # Box Plot

# In[ ]:


my_pal = {'Confirmed': "b",'Active': "y", 'Recovered': "g", 'Deaths':"r"}
ax = sns.boxplot(data=df_countries_grouped, orient="h")
ax = sns.swarmplot(data=df_countries_grouped, orient="h", palette=my_pal)


# # heatmap Correlation 

# In[ ]:


ax = plt.subplots(figsize=(18, 12))
sns.heatmap(df_countries.corr(), annot=True, linewidths=.5, cmap="Blues")


# # Pair plot

# In[ ]:


sns.pairplot(df_countries_grouped[['Confirmed','Deaths','Recovered','Active']], diag_kind = 'kde', 
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},size = 4);


# # heatmap top countries

# In[ ]:


sns.set_style("darkgrid")
ax = plt.subplots(figsize=(18, 12))
sns.heatmap(df_countries_top)


# # Area Plot<a id="6"></a>
# 
# visualize the top countries as a cumulative plot, also knows as a **Stacked Line Plot** 
# 

# In[ ]:


df_countries_top.plot(kind='area', figsize=(18, 9),stacked=False, 
            color=['blue', 'orange', 'green', 'red']) 
plt.show()


# In[ ]:


df_countries_top = df_countries_top.reset_index()


# # Multicollinearity top countries Pairplot 
# 
# linear relationships between independent variables
# 

# In[ ]:


sns.pairplot(df_countries_top, 
             vars = ['Confirmed', 'Active', 'Recovered', 'Deaths'], 
             hue = 'Country', diag_kind = 'kde', palette="husl",
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4);


# # Box Plot <a id="8"></a>
# 
# way of statistically representing the *distribution* of the data through five main dimensions: 
# 
# - **Minimun:** Smallest number in the dataset.
# - **First quartile:** Middle number between the `minimum` and the `median`.
# - **Second quartile (Median):** Middle number of the (sorted) dataset.
# - **Third quartile:** Middle number between `median` and `maximum`.
# - **Maximum:** Highest number in the dataset.

# In[ ]:


color = dict(boxes='Blue', whiskers='Orange',medians='Green', caps='Red')
df_countries_top.plot(kind ='box',notch= False,
          color=color, sym='r+', vert=False ,patch_artist=False,
          figsize=(18, 9))
plt.title('')
plt.show()


# # Histogram<a id="8"></a>
# 
# histogram representing the *frequency* distribution of numeric dataset

# In[ ]:


df_countries_top.plot(kind='hist', figsize=(18, 9),stacked=True, 
             alpha=0.5,
            color=['blue', 'orange', 'green', 'red']) 
plt.show()

