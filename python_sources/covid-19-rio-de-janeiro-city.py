#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.imgur.com/H6b3KTo.png" width="13%" align="left"><br>
# 
# If you find this notebook useful, please support data sharing by referencing [its original source](https://www.kaggle.com/resendeacm/covid19-rj). Don't forget to upvote, please. :)
# ***
# 
# ### Context - [World Health Organization (WHO)](https://www.who.int/health-topics/coronavirus#tab=tab_1)
# 
# + Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus (2019-nCoV).
# + At this time, there are no specific vaccines or treatments for COVID-19. However, there are many ongoing clinical trials evaluating potential treatments.
# + The best way to prevent and slow down transmission is to be well informed about the COVID-19 virus, the disease it causes, and how it spreads. 
# + As of 28 April 2020, 71.886 cases have been confirmed in Brazil.
# 

# ### Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Pre-Processing

# In[ ]:


# Working Folders
inp_folder = '/kaggle/input/covid19-rj/'
out_folder = '/kaggle/working/'

# Original File: rio_covid19_kaggle.csv (https://www.kaggle.com/resendeacm/covid19-rj/)
filename = inp_folder + 'rio_covid19_kaggle.csv'

# Retrieve Main Dataset
df = pd.read_csv(filename, header=0)

# Time Course: Rio de Janeiro (City)
rio_city_general = df.groupby(['Date']).sum().reset_index()

# Last Recorded Day: All Neighborhoods
last_day = df['Date'][-1:].iloc[0]
last_recorded = df[df['Date'] == last_day]
last_recorded_cases = last_recorded.sort_values(by=['Cases'], ascending=False)
last_recorded_deaths = last_recorded.sort_values(by=['Deaths'], ascending=False)
last_recorded_recovered = last_recorded.sort_values(by=['Recovered'], ascending=False)


# ### Time Course: Rio de Janeiro (City)

# In[ ]:


# Time Course Table: Rio de Janeiro (City)
rio_city_general.style.background_gradient(cmap='Blues', subset=["Cases"])                      .background_gradient(cmap='Reds', subset=["Deaths"])                      .background_gradient(cmap='Greens', subset=["Recovered"])


# In[ ]:


# Time Course Figure: Rio de Janeiro (City)
fig, axes = plt.subplots(figsize=(10,8))

# Plot Quantities
rio_city_general.plot(x='Date', y='Cases', style='--bo', markersize=5, ax=axes)
rio_city_general.plot(x='Date', y='Deaths', style='--ro', markersize=5, ax=axes)
rio_city_general.plot(x='Date', y='Recovered', style='--go', markersize=5, ax=axes)
plt.grid(alpha=0.3)

# Fill Curves
plt.fill_between(rio_city_general['Date'], 0, rio_city_general['Deaths'], color='red', alpha=0.3)
plt.fill_between(rio_city_general['Date'], 0, rio_city_general['Recovered'], color='green', alpha=0.15)
plt.fill_between(rio_city_general['Date'], 0, rio_city_general['Cases'], color='blue', alpha=0.1)

# Plot Labels & Title
plt.ylabel('Confirmed Cases, Deaths & Recovered')
plt.title('Rio de Janeiro (City): Status COVID-19')

# Save Figure
plt.savefig(out_folder + 'rio_city_general.png')
plt.show()
plt.close()


# ### Last Recorded Day: Neighborhood Analysis

# In[ ]:


# Last Recorded Day: All Neighborhoods - Sorted (Confirmed Cases)
last_recorded_cases.style.background_gradient(cmap='Blues',subset=["Cases"])                         .background_gradient(cmap='Reds',subset=["Deaths"])                         .background_gradient(cmap='Greens',subset=["Recovered"])


# In[ ]:


# Last Recorded Day: TOP10 Neighborhoods (Confirmed Cases)
fig, axes = plt.subplots(figsize=(10,8))
plt.axes(axisbelow=True)

# Plot Quantities
plt.barh(last_recorded_cases['Neighborhood'].values[:10], last_recorded_cases['Cases'].values[:10], color="slateblue")
plt.grid(alpha=0.3)

# Plot Labels & Title
plt.xlabel("Confirmed Cases")
plt.title("TOP10 Neighborhoods (COVID-19 Confirmed Cases): " + last_day)

# Save Figure
plt.tight_layout()
plt.savefig(out_folder + 'top10_confirmed_cases.png')
plt.show()
plt.close()


# In[ ]:


# Last Recorded Day: All Neighborhoods - Sorted (Deaths)
last_recorded_deaths.style.background_gradient(cmap='Blues',subset=["Cases"])                          .background_gradient(cmap='Reds',subset=["Deaths"])                          .background_gradient(cmap='Greens',subset=["Recovered"])


# In[ ]:


# Last Recorded Day: TOP10 Neighborhoods (Deaths)
fig, axes = plt.subplots(figsize=(10,8))
plt.axes(axisbelow=True)

# Plot Quantities
plt.barh(last_recorded_deaths['Neighborhood'].values[:10], last_recorded_deaths['Deaths'].values[:10], color="crimson")
plt.grid(alpha=0.3)

# Plot Labels & Title
plt.xlabel("Deaths")
plt.title("TOP10 Neighborhoods (COVID-19 Deaths): " + last_day)

# Save Figure
plt.tight_layout()
plt.savefig(out_folder + 'top10_deaths.png')
plt.show()
plt.close()


# In[ ]:


# Last Recorded Day: All Neighborhoods - Sorted (Recovered)
last_recorded_recovered.style.background_gradient(cmap='Blues',subset=["Cases"])                             .background_gradient(cmap='Reds',subset=["Deaths"])                             .background_gradient(cmap='Greens',subset=["Recovered"])


# In[ ]:


# Last Recorded Day: TOP10 Neighborhoods (Recovered)
fig, axes = plt.subplots(figsize=(10,8))
plt.axes(axisbelow=True)

# Plot Quantities
plt.barh(last_recorded_recovered['Neighborhood'].values[:10], last_recorded_recovered['Recovered'].values[:10], color="darkcyan")
plt.grid(alpha=0.3)

# Plot Labels & Title
plt.xlabel("Recovered")
plt.title("TOP10 Neighborhoods (COVID-19 Recovered): " + last_day)

# Save Figure
plt.tight_layout()
plt.savefig(out_folder + 'top10_recovered.png')
plt.show()
plt.close()


# ### Time Course: Some Neighborhoods

# In[ ]:


# Last Recorded Day: TOP10 Neighborhoods (Confirmed Cases)
top10_confirmed = last_recorded_cases['Neighborhood'].values[:10]

# Subplot Options
cols = 2
rows = int(len(top10_confirmed)/2)
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,20))

# TOP10 Plot
for cont, neighborhood in enumerate(top10_confirmed):
    # Neighborhood/Position Relation
    i = int(cont/cols)
    j = cont - i*cols
    
    # Select Neighborhood & Plot Quantities
    dfplot = df[df['Neighborhood'] == neighborhood]
    dfplot.plot(x='Date', y='Cases', style='--bo', markersize=5, ax=axes[i,j])
    dfplot.plot(x='Date', y='Deaths', style='--ro', markersize=5, ax=axes[i,j])
    dfplot.plot(x='Date', y='Recovered', style='--go', markersize=5, ax=axes[i,j])
    axes[i,j].grid(alpha=0.3)
    
    # Fill Curves
    axes[i,j].fill_between(dfplot['Date'], 0, dfplot['Deaths'], color='red', alpha=0.3)
    axes[i,j].fill_between(dfplot['Date'], 0, dfplot['Recovered'], color='green', alpha=0.15)
    axes[i,j].fill_between(dfplot['Date'], 0, dfplot['Cases'], color='blue', alpha=0.1)
    
    # Plot Labels & Title
    axes[i,j].set_title(neighborhood + ' (COVID-19)')
    axes[i,j].set_ylabel('Confirmed Cases, Deaths & Recovered')

# Save Figure
fig.tight_layout(pad=3.0)
plt.savefig(out_folder + 'top10_time_course.png')
plt.show()
plt.close()

