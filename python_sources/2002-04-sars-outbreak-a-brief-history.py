#!/usr/bin/env python
# coding: utf-8

# # 2002-04 SARS Outbreak: A brief history

# In 2002-04, the world was hit by an outbreak involving Severe Acute Respiratory Syndrome (SARS), which was caused by a strain of coronavirus known as SARS-CoV. Now in 2020, its relative, the SARS-CoV-2 is giving us a grim reminder of the terrors of living in a pandemic with Covid-19. 
# 
# Many variables have changed since our last encounter with a coronavirus pandemic, from the contagiousness of the disease to the preparedness of our healthcare system in dealing with this. Nevertheless, it's worth taking a peak at the data and remind ourselves of the lesson that we learned from previous outbreak. 

# ## Preparing the data

# In[ ]:


# Load packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


# import data & rename columns
sars = pd.read_csv("/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")
sars.columns = ['Date', 'Country', 'Cases', 'Deaths', 'Recovered']


# In[ ]:


# add a column for active cases
sars['Active'] = sars['Cases'] - (sars['Recovered'] + sars['Deaths'])


# ## How did SARS progress around the world over time?

# In[ ]:


sars_over_time = sars.groupby(['Date'])['Cases', 'Deaths', 'Recovered', 'Active'].sum().reset_index()


# In[ ]:


# Plotting the time series graph of the outbreak
fig = go.Figure()
fig.add_trace(go.Scatter(x=sars_over_time.Date, y=sars_over_time['Cases'], name="Cases",
                         line_color='purple'))

fig.add_trace(go.Scatter(x=sars_over_time.Date, y=sars_over_time['Deaths'], name="Death",
                         line_color='black'))

fig.add_trace(go.Scatter(x=sars_over_time.Date, y=sars_over_time['Recovered'], name="Recovered",
                         line_color='yellow'))

fig.add_trace(go.Scatter(x=sars_over_time.Date, y=sars_over_time['Active'], name="Active",
                         line_color='red'))

fig.update_layout(title_text="SARS count worldwide")
fig.show()


# The dataset covers the number of reported cases of SARS from March 23rd, 2003 to Jul 11, 2003. The number of recovered and deaths were also reported, which helped us inferred the number of active cases at a given time using the formula Active = Cases - (Death + Recovered) 
# 
# We can see that the number of active cases peaked on May 12, 2003 at 3700 cases. This is the point where the number of new SARS cases are outpaced by the number of recovery (instead of the number of death, fortunately, as our graph show). Nevertheless, this timestamp would serve to indicate that the effort to pushback SARS is finally slowing down the spread of the virus 
# 
# There are some other timestamp worth noting from this chart as well:
# - May 16, 2003, the number of recovery outpaces the number of active cases. This could indicate that the tide is turning in the battle against SARS, as there are now more resources available to treat active cases as more people have recovered from the disease. 
# - June 16, 2003, the number of active cases drop below the number of total deaths. Not sure if this has any significant meaning, but it should at least that more people are recovering rather than dying.
# - April 10, 2003, the number of recovered patients rocketted from 0 to 1337. Was this spike due to not having numbers reported in earlier dates or was it due to some magical cure (not likely)? 
# 

# ## How was each country affected by SARS?

# In[ ]:


sars_by_country = sars.groupby(['Country'])['Cases', 'Deaths', 'Recovered'].max().reset_index().sort_values('Cases', ascending=False)


# In[ ]:


sars_by_country.head(10)


# As we see from here, China, Hong Kong, Taiwan, Canada and Singapore were most affected by the 2002-04 SARS outbreak, all experienced 200+ cases. These countries will be examined further to see how the outbreak progressed in each country and whether the changes correlates with any measures taken by the country at the time.
# 
# United States also has 200+ cases in this dataset, but the official data from WHO [only recorded roughly 27 cases](http://)https://www.who.int/csr/sars/country/table2004_04_21/en/. Hence, it will not be examined here.

# In[ ]:


def visualize_country(country_name):
    country_data = sars[sars.Country == country_name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=country_data.Date, y=country_data['Cases'], name="Cases",
                         line_color='purple'))
    fig.add_trace(go.Scatter(x=country_data.Date, y=country_data['Deaths'], name="Death",
                         line_color='black'))
    fig.add_trace(go.Scatter(x=country_data.Date, y=country_data['Recovered'], name="Recovered",
                         line_color='yellow'))
    fig.add_trace(go.Scatter(x=country_data.Date, y=country_data['Active'], name="Active",
                         line_color='red'))
    fig.update_layout(title_text=f'SARS count in {country_name}')
    fig.show()


# ### China's SARS progression

# In[ ]:


visualize_country('China')


# China's progress seems to mimic that of the world, with peak active cases occuring at the same time. 

# ### Hong Kong's SARS progression

# In[ ]:


visualize_country('Hong Kong SAR, China')


# Hong Kong appears to experience the peak in active cases earlier than the world, around April 17, 2002 at 960. 

# ### Taiwan's SARS progression

# In[ ]:


visualize_country('Taiwan, China')


# Taiwan experienced peak active cases later than the world, at around Jun 2, 2003 at 469. 

# ### Canada's SARS progression

# In[ ]:


visualize_country('Canada')


# The progression for Canada seems rather odd. The number of active cases peaked at 84 on April 4 and decreases until May 24, 2003 where the number of cases rise drastically again to 69 on Jun 9, 2003

# ### Singapore's SARS progression

# In[ ]:


visualize_country('Singapore')


# Like Hong Kong, Singapore peaked in active cases earlier on April 9 2003 at 109 cases. 

# In[ ]:




