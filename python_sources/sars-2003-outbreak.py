#!/usr/bin/env python
# coding: utf-8

# ### SARS 2003 outbreak: When was a global epidemic stopped?
# **References and Acknowledgements:**
# * Dataset: [Sars Outbreak 2003 - Kaggle](https://www.kaggle.com/imdevskp/sars-outbreak-2003-complete-dataset)
# * [CDC SARS Response Timeline](https://www.cdc.gov/about/history/sars/timeline.htm)

# **Load packages**

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# **Load data**

# In[ ]:


data = pd.read_csv("../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")


# In[ ]:


data_cp = data.copy()
# To make sure that I DO NOT modify the orginal dataset, I copied this data into a variable 
data_cp.head()


# In[ ]:


# rename columns
data_cp['Date'] = pd.to_datetime(data_cp['Date'])
data_cp = data_cp.rename(columns={'Cumulative number of case(s)':'Confirmed cases',                         'Number of deaths': 'Deaths',                       'Number recovered': 'Recovered cases'})


# In[ ]:


data_cp.dtypes


# In[ ]:


print(f"The dataset has {data_cp.shape[0]} rows and {data_cp.shape[1]} colums")


# **Confirmed and recovered cases, and deaths over time**

# In[ ]:


by_date = data_cp.groupby('Date')['Confirmed cases', 'Deaths', 'Recovered cases'].sum().reset_index()


# In[ ]:


df_melt_bydate = by_date.melt(id_vars='Date', value_vars=['Confirmed cases', 'Deaths', 'Recovered cases'])


# In[ ]:


fig = px.line(df_melt_bydate, x='Date' , y='value' , color='variable',
             title = 'Worldwide confirmed and recovered cases, and deaths over time')
fig.add_annotation(x="2003-05-29", y=8295, xref="x", yref="y",
        text="The curve starts being flattened", showarrow=True,
        font=dict( family="Courier New, monospace", size=12, color="#ffffff"),
        align="center", arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363",
        ax=20, ay=-30, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#ff7f0e", opacity=0.8)
fig.update_layout(showlegend=True)

fig.show()

fig = px.line(df_melt_bydate, x='Date' , y='value' , color='variable',
             title = 'Worldwide confirmed and recovered cases, and deaths over time (Logarithmic Scale)',\
              log_y = True)
fig.show()


# - Although the first case of atypical pneumonia is reported in the Guangdong province in southern China, untill 17 Mar 2003 the number of confirmed cases was just recorded.
# - The slope of the confirmed cases line was very sharp from Mar 25 to 21 May 2003.
# - Looking at the same graph in Logarithmic scale reveals that the pandemic was in it's peaks on Mar 26 in terms of confirmed cases and deaths.
# - However, the curve started being flattened after only 2.5 months, which was really good news for us.

# In[ ]:


by_country = data_cp.groupby("Country")["Confirmed cases", "Deaths", "Recovered cases"].max().reset_index()
fig = px.choropleth(by_country, locations="Country", 
                    locationmode='country names', color="Confirmed cases", 
                    hover_name="Country", range_color=[1,5000], 
                    color_continuous_scale="peach", 
                    title='Countries with Confirmed cases')
fig.show()


# The above graph is an illustration of how the SARS was spread out across the globe.

# In[ ]:


by_country["% confirmed cases"] = round(100 * by_country["Confirmed cases"] / by_country["Confirmed cases"].sum(), 2)
by_country["Mortality Rate (%)"] = round(100 * by_country["Deaths"] / by_country["Confirmed cases"],2)
by_country["Recovery Rate (%)"] = round(100 * by_country["Recovered cases"] / by_country["Confirmed cases"],2)

top_20_contries = by_country.sort_values('Confirmed cases', ascending=False)[:20][::-1]
fig = px.bar(top_20_contries, 
             x='Confirmed cases', y='Country',
             title='Confirmed Cases Worldwide', text='Confirmed cases', height=600, orientation='h')
fig.show()


# In[ ]:


by_country.sort_values('% confirmed cases', ascending=False)            [['Country', '% confirmed cases','Mortality Rate (%)', 'Recovery Rate (%)']][:20]            .style.background_gradient(cmap='Greens')


# - It's worth noting that China and Hong Kong were most affected by SARS (took up over 80% confirmed cases over the world).
# - This is probably much diferent from COVID-19, now Europe is more affected than China and it's neighbors by COVID-19, and Iran being the most affected Asian country other than China.

# **Mortality and Recovery Rate**
# <br>
# *Top 10 countries with highest Recovery Rate*

# In[ ]:


by_country.sort_values('Recovery Rate (%)', ascending=False)            [['Country', 'Confirmed cases', '% confirmed cases','Mortality Rate (%)', 'Recovery Rate (%)']][:10]


# *Top 10 countries with highest Mortality Rate*

# In[ ]:


by_country.sort_values('Mortality Rate (%)', ascending=False)            [['Country', 'Confirmed cases', '% confirmed cases','Mortality Rate (%)', 'Recovery Rate (%)']][:10]


# In[ ]:


animated_data = data_cp.groupby(['Date', 'Country'])['Confirmed cases', 'Deaths'].max()
animated_data = animated_data.reset_index()
animated_data['Date'] = pd.to_datetime(animated_data['Date'])
animated_data['Date'] = animated_data['Date'].dt.strftime('%m/%d/%Y')
animated_data['size'] = animated_data['Confirmed cases'].pow(0.3)

fig = px.scatter_geo(animated_data, locations="Country", locationmode='country names', 
                     color="Confirmed cases", size='size', hover_name="Country", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="Date", 
                     title='SARS 2003: Spread Over Time', color_continuous_scale="portland")
fig.update_layout(transition_duration=0.0001)
fig.show()


# In[ ]:


fig = px.scatter_geo(animated_data, locations="Country", locationmode='country names', 
                     color="Deaths", size='size', hover_name="Country", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="Date", 
                     title='SARS: Deaths Over Time', color_continuous_scale="peach")
fig.update_layout(transition_duration=0.0001)
fig.show()


# ### Summary ###
# The SARS outbreak in 2003 resulted in more than 8000 cases and 800 deaths. However, the curve started being flattened after only 2.5 months (since 17 Mar 2003) by means of syndromic surveillance, prompt isolation of patients, strict enforcement of quarantine of all contacts... Hopefully, we can contain Covid19 as the way we fought with SARS. Stay safe and healthy guys!!!
