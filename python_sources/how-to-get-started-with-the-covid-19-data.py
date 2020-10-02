#!/usr/bin/env python
# coding: utf-8

# # How to get started with the COVID-19 data
# 
# 

# *Step 1: Import Python packages and load the data*

# In[ ]:


# Import Python Packages
import pandas as pd
import numpy as np
import plotly.express as px
import warnings 
warnings.filterwarnings('ignore')

# Load Data
df_global = pd.read_csv('/kaggle/input/coronavirus-covid19-mortality-rate-by-country/global_covid19_mortality_rates.csv')
df_usa = pd.read_csv('/kaggle/input/coronavirus-covid19-mortality-rate-by-country/usa_covid19_mortality_rates.csv')
todays_date = '7/11/2020' # Update this line every time that you rerun the notebook


# *Step 2: Map Spread of COVID-19 for Every Country*

# In[ ]:


fig = px.choropleth(df_global, 
                    locations="Country", 
                    color="Confirmed", 
                    locationmode = 'country names', 
                    hover_name="Country",
                    range_color=[0,1000000],
                    title='Global COVID-19 Infections as of '+todays_date)
fig.show()

fig = px.choropleth(df_global, 
                    locations="Country", 
                    color="Deaths", 
                    locationmode = 'country names', 
                    hover_name="Country",
                    range_color=[0,100000],
                    title='Global COVID-19 Deaths as of '+todays_date)
fig.show()

fig = px.choropleth(df_global, 
                    locations="Country", 
                    color="Mortality Ratio", 
                    locationmode = 'country names', 
                    hover_name="Country",
                    range_color=[0,10],
                    title='Global COVID-19 Mortality Ratios as of '+todays_date)
fig.show()


# *Step 3: Plot Spread of COVID-19 for Every Country*

# In[ ]:


fig = px.bar(df_global.sort_values('Confirmed',ascending=False)[0:20], 
             x="Country", 
             y="Confirmed",
             title='Global COVID-19 Infections as of '+todays_date)
fig.show()

fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:20], 
             x="Country", 
             y="Deaths",
             title='Global COVID-19 Deaths as of '+todays_date)
fig.show()

fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:20], 
             x="Country", 
             y="Mortality Ratio",
             title='Global COVID-19 Mortality Ratios as of '+todays_date+' for Countries with Top 20 Most Deaths')
fig.show()


# *Step 4: Map Spread of COVID-19 for USA State*

# In[ ]:


fig = px.choropleth(df_usa, 
                    locations="USA_State_Code", 
                    color="Confirmed", 
                    locationmode = 'USA-states', 
                    hover_name="State",
                    range_color=[0,300000],scope="usa",
                    title='Global COVID-19 Infections as of '+todays_date)
fig.show()

fig = px.choropleth(df_usa, 
                    locations="USA_State_Code", 
                    color="Deaths", 
                    locationmode = 'USA-states', 
                    hover_name="State",
                    range_color=[0,20000],scope="usa",
                    title='Global COVID-19 Deaths as of '+todays_date)
fig.show()

fig = px.choropleth(df_usa, 
                    locations="USA_State_Code", 
                    color="Mortality Ratio", 
                    locationmode = 'USA-states', 
                    hover_name="State",
                    range_color=[0,10],scope="usa",
                    title='Global COVID-19 Mortality Ratios as of '+todays_date)
fig.show()


# *Step 5: Plot Spread of COVID-19 for USA State*

# In[ ]:


fig = px.bar(df_usa.sort_values('Confirmed',ascending=False)[0:20], 
             x="State", 
             y="Confirmed",
             title='USA COVID-19 Infections as of '+todays_date)
fig.show()

fig = px.bar(df_usa.sort_values('Deaths',ascending=False)[0:20], 
             x="State", 
             y="Deaths",
             title='USA COVID-19 Deaths as of '+todays_date)
fig.show()

fig = px.bar(df_usa.sort_values('Deaths',ascending=False)[0:20], 
             x="State", 
             y="Mortality Ratio",
             title='USA COVID-19 Mortality Ratios as of '+todays_date+' for USA States with Top 20 Most Deaths')
fig.show()


# *Step 6: Plot COVID-19 vs Latitude for Every Country*

# In[ ]:


df_global2 = df_global
df_global2['Latitude'] = abs(df_global2['Latitude'])
df_global2 = df_global2[df_global2['Country']!='China']

fig = px.scatter(df_global2.sort_values('Deaths',ascending=False), 
             x="Latitude", 
             y="Confirmed",
             title='Global COVID-19 Infections vs Absolute Value of Latitude Coordinate as of '+todays_date)
fig.show()

fig = px.scatter(df_global2.sort_values('Deaths',ascending=False), 
             x="Latitude", 
             y="Deaths",
             title='Global COVID-19 Deaths vs Absolute Value of Latitude Coordinate as of '+todays_date)
fig.show()
fig = px.scatter(df_global2.sort_values('Deaths',ascending=False), 
             x="Latitude", 
             y="Mortality Ratio",
             title='Global COVID-19 Mortality Ratios vs Absolute Value of Latitude Coordinate as of '+todays_date)
fig.show()
df_global.sort_values('Mortality Ratio', ascending= False).head(10)


# *Step 7: Plot COVID-19 vs Latitude for Every USA State*

# In[ ]:


fig = px.scatter(df_usa.sort_values('Deaths',ascending=False), 
             x="Latitude", 
             y="Mortality Ratio",
             title='USA States COVID-19 Mortality Ratios vs Absolute Value of Latitude Coordinate as of '+todays_date)
fig.show()
df_usa.sort_values('Mortality Ratio', ascending= False).head(10)


# In[ ]:




