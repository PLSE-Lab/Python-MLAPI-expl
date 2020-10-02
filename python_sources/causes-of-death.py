#!/usr/bin/env python
# coding: utf-8

# # Causes of Death
# 
# How does COVID-19 rank versus other causes of death in Ontario?
# 

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px


# ## Import: Causes of Death
# 
# First, we will import and sort causes of death in Ontario from Stats Canada.
# 
# Source for causes of death: Statistics Canada.  [Table  13-10-0801-01   Leading causes of death, total population (age standardization using 2011 population, 2018 is the most current year available)](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310080101)
# DOI:   https://doi.org/10.25318/1310080101-eng
# 
# NB: raw number of deaths used, *not* age-standardized per 100k.  Both values are located on this StatsCan table.
# 

# In[ ]:


stats_df = pd.read_csv('/kaggle/input/causes-of-death-data/statscan_on_causes_death.csv')
stats_df.head()


# In[ ]:


causes_df = stats_df.loc[stats_df['Characteristics']=='Number of deaths', ['Leading causes of death (ICD-10)', 'VALUE']]
causes_df.rename(columns={
    'Leading causes of death (ICD-10)' : 'Cause',
    'VALUE' : 'Deaths',
}, inplace=True)
causes_df = causes_df[causes_df['Cause'] != 'Other causes of death']
causes_df.sort_values(by='Deaths', ascending=False, inplace=True)
causes_df


# ## Import: Ontario's Current and 2018 Population
# 
# Ontario's population has grown since the causes of death data was recorded in 2018.  We use the StatsCan estimated current population and estimated 2018 population to scale the number of deaths to match the current population.
# 
# 2020Q1 population used for "current."  Average of 2018 quarterly population estimates used for 2018.
# 
# Ontario's population estimated using:
# Statistics Canada.  [Table  17-10-0009-01   Population estimates, quarterly](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901)
# DOI:   https://doi.org/10.25318/1710000901-eng
# 

# In[ ]:


ONTARIO_POPULATION_2020Q1 = 14711827
ONTARIO_POPULATION_2018 = (14188919+14241379+14318545+14405726)/4


# ## Import: Ontario Deaths Due to COVID-19
# 
# For COVID-19 data, we will use Isha Berry et al's COVID-19 Canada Open Data Working Group data set.
# 
# https://howsmyflattening.ca/#/data

# In[ ]:


mortality_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_mortality.csv')
mortality_df['date'] = pd.to_datetime(mortality_df['date'], dayfirst=True)
mortality_df.head()


# In[ ]:


daily_death_ser = mortality_df[mortality_df['province']=='Ontario'].groupby('date')['death_id'].count()
daily_death_ser.name = 'daily_deaths_ontario'
daily_death_ser


# ## Analysis
# 
# In order to get a sense of the magnitude of COVID-19 mortality relative to other causes, we chose to analyse on a per day basis since the first death in Ontario occured on Mar 11, 2020. Here we are assuming the deaths are evenly distributed across days throughout the year. This assumption breaks down for highly seasonal disease such as influenza.
# 

# Convert the 2018 number of deaths to 2020 daily estimates using a simple population scaling and division by number of days in the year.

# In[ ]:


causes_df['Daily_Deaths'] = causes_df['Deaths']/365*ONTARIO_POPULATION_2020Q1/ONTARIO_POPULATION_2018
causes_df.head(10)


# Put some nice names on disease categories.

# In[ ]:


causes_df['Cause'].head(10).to_list()


# In[ ]:


# Make some friendly names
def cause_to_friendly(cause):
  c2f = {
    'Malignant neoplasms [C00-C97]' : 'All Cancers',
    'Diseases of heart [I00-I09, I11, I13, I20-I51]' : 'Heart Disease',
    'Accidents (unintentional injuries) [V01-X59, Y85-Y86]' : 'Accidents / Injuries',
    'Cerebrovascular diseases [I60-I69]' : 'Stroke and Related Diseases',
    'Chronic lower respiratory diseases [J40-J47]' : 'Chronic Lower Respiratory Diseases',
    'Influenza and pneumonia [J09-J18]' : 'Influenza and Pneumonia',
    'Diabetes mellitus [E10-E14]' : 'Diabetes',
    "Alzheimer's disease [G30]" : "Alzheimer's Disease",
    'Intentional self-harm (suicide) [X60-X84, Y87.0]' : 'Suicide',
    'Chronic liver disease and cirrhosis [K70, K73-K74]' : 'Cirrhosis and Other Chronic Liver Diseases'
  }
  try:
    return c2f[cause]
  except KeyError:
    return np.nan


# In[ ]:


causes_df['Friendly_Name'] = causes_df['Cause'].apply(cause_to_friendly)


# ## Plot

# In[ ]:


fig = px.bar(x=daily_death_ser.index, y=daily_death_ser.values, 
             text=daily_death_ser.values,
             )
for _, row in causes_df.head(8).iterrows():
  if row['Cause'] != 'Influenza and pneumonia [J09-J18]':
    fig.add_shape(
            # Line Horizontal
                type="line",
                x0=daily_death_ser.index.min(),
                y0=row['Daily_Deaths'],
                x1=daily_death_ser.index.max(),
                y1=row['Daily_Deaths'],
                line=dict(
                    color="Black",
                    width=1,
                    dash="dot",
                ),
                layer="below",
        )
    y_text = row['Daily_Deaths']+0.1
    # fix overlap
    if row['Cause'] == 'Chronic lower respiratory diseases [J40-J47]':
      y_text = row['Daily_Deaths']-1.7
    fig.add_trace(go.Scatter(
      x=[daily_death_ser.index.min()],
      y=[y_text],
      text=[row['Friendly_Name']],
      mode="text",
      textposition="top right",
      showlegend=False,
    ))
fig.update_layout(
    autosize=False,
    width=800,
    height=900,
    title='Daily Deaths in Ontario: Comparison of COVID-19 and Estimates of non-COVID-19 Leading Causes',
    xaxis_title="Date (since first Ontario death)", 
    yaxis_title="Estimated Deaths per Day",    
)
number_of_days = (daily_death_ser.index.max() - daily_death_ser.index.min()).days + 1
fig.update_xaxes(tickangle=-90, nticks=number_of_days)
fig.show()


# ## How does COVID-19 mortality in Ontario compare to other diseases
# 
# This chart puts COVID-19 deaths in context, relative to other leading causes of mortality in Ontario.
# 
# The figure shows the number of deaths that have occurred each day in Ontario due to COVID-19, since the first death on March 11, 2020. As a comparison, we have also estimated the number of deaths expected each day in Ontario for 7 of the top 8 causes of death in Ontario (from Statistics Canada).
# 
# Deaths due to influenza and pneumonia were the sixth leading cause of death in Ontario in 2018 at 3,055 dead, but are excluded from this analysis as the StatsCan data does not account for seasonality of deaths  (Stay tuned for future work on COVID-19 vs influenza comparisons.)
# 
# This analysis shows that COVID-19 went from causing no deaths in early March to the #3 cause of death in early April, and reached the #2 cause of death on April 14, 2020.  This illustrates a few key take aways:
# 1. COVID-19 is currently the second leading cause of deaths per day in Ontario, behind cancer. 
# 2. COVID-19 has ascended rank rapidly over 2 weeks.
# 3. This figure demonstrates helps understand the rapid shift in clinical care that has occured in the last few weeks, putting sudden pressure on front-line personnel and facilities. Implications of this shift are yet to be fully understood.
# 
# Sources of data:
# * COVID-19 Deaths in Ontario: COVID-19 Canada Open Data Working Group. [Epidemiological Data from the COVID-19 Outbreak in Canada.](https://github.com/ishaberry/Covid19Canada)
# * Non-COVID-19 Causes of Death: 2018 Ontario Averages from Statistics Canada.  [Table  13-10-0801-01   Leading causes of death, total population (age standardization using 2011 population)](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310080101) DOI:   https://doi.org/10.25318/1310080101-eng (NB: raw number of deaths used, not age-adjusted rates)
# * Population of Ontario: 2018 and 2020 Q1 estimate from Statistics Canada.  [Table  17-10-0009-01   Population estimates, quarterly](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901)
# DOI:   https://doi.org/10.25318/1710000901-eng

# In[ ]:




