#!/usr/bin/env python
# coding: utf-8

# # Covid-19 in the US : Analysis, Interactive Plotting and Visualization
# ### Upvotes would be highly appreciated
# 

# ![street_art_covid_19](https://drive.google.com/uc?id=1mR6lkYcjxfpJMc_SeffFaXLKoOJUf8Xm) <br />
# *A coronavirus street art piece by artist **Pony Wave** on Venice Beach in Los Angeles showing a couple kissing with face masks.*

# # Table of Content:
# * [Preparation for Visualization](#first-bullet) :
#     * [Dataset Exploration](#second-bullet)
#     * [Data Preprocessing and Feature Engineering](#third-bullet)
# * [Visualilizations](#fourth-bullet) : 
#     * [Covid-19 pandemic evolution since first record](#fifth-bullet)
#     * [US Interactive mapping](#sixth-bullet)
#     * [Animated Visualizations](#seventh-bullet)
# * [Next Steps](#eighth-bullet) : 

# Note that most of the visualizations are interactive. Hover over to see more information.

# ---

# # Preparation for Visualization <a class="anchor" id="first-bullet"></a>

# * ## Dataset Exploration <a class="anchor" id="second-bullet"></a>

# In[ ]:


import pandas as pd
import numpy as np
import math


# In[ ]:


df = pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isna().sum()


# In[ ]:


df.describe()


# In[ ]:


df['county'].value_counts()


# In[ ]:


print("""Dataset contains \033[1m{}\033[0m rows and \033[1m{}\033[0m columns. \n\033[1m{}\033[0m missing values have been encountered in the "fips" column. \nAlthough there are no missing values in the "county" column, \033[1m{}\033[0m records are containing an "Unknown" value.
""".format(len(df.index), len(df.columns), df['fips'].isna().sum(), len(df[df.county == 'Unknown'])))


# * ## **Data Preprocessing and Feature Engineering <a class="anchor" id="third-bullet"></a>**

# Creating, combining and modifying features of the dataset will help us to plot some graphs.
# 
# * 1)  State Postal Code feature creation <br/>New dataset: each state with its respective postal code

# In[ ]:


df_st_codes = pd.read_csv('../input/usstatescodes/us-states-codes.csv', sep=';')


# In[ ]:


"""
For each row of the original dataset('df'), finding the matching state value within the 'df_st_codes' dataset
and recovering its associated code. A new column in the original dataset is created to store those postal codes.
"""
df['code'] = (df.state.apply(lambda x: df_st_codes.query('@x in state').index.values)
                         .apply(lambda x: x[0] if len(x) else "NaN")).apply(lambda x: df_st_codes.code[x] if x != "NaN" else x)


# * 2) Combining counties and codes into a new feature <br />
# We noticed that counties from different states have the same name. In order to facilitate our work in the coming steps, <br /> let's create a new feature gathering the name of each county present in the dataset with its respective state code.

# In[ ]:


""" 
Example showing same county name in different states
It results that the states of New York, Pennsylvania and West Virginia have a county named Wyoming.
"""
dupl_test= df[df.county == 'Wyoming']
dupl_test.state.unique()


# In[ ]:


"""
Creating a bew feature combining the name of each county present in the dataset with its respective state code
"""
df['agg_county_code'] = df['county'] + ', ' + df['code']


# * 3) Day-to-day new cases and new deaths registered <br />
# Each row of the original dataset reports day-to-day cumulative counts per county. In the following operation, we are creating two new features (new_cases, new_deaths). New cases is substracting the results of day *n* for each county with day *n-1*. Same for new deaths. It results that we will get both new cases and new deaths recorded, each day, per county.

# In[ ]:


df = df.sort_values(by=['state', 'county', 'date'], ascending=True)


# In[ ]:


unique_agg = list(df['agg_county_code'].unique())
new_cases_list = []
new_deaths_list = []

for agg in unique_agg:
    df_filtered = df[df['agg_county_code'] == agg]
    cases_list = list(df_filtered.iloc[:, 4])
    deaths_list = list(df_filtered.iloc[:, 5])
    for j in range(0, len(cases_list)):
        if j == 0:
            new_cases_list.append(cases_list[0])
            new_deaths_list.append(deaths_list[0])
        else:
            new_cases_list.append(cases_list[j] - cases_list[j-1])
            new_deaths_list.append(deaths_list[j] - deaths_list[j-1])
    
new_cases_s = pd.Series(new_cases_list)
new_deaths_s = pd.Series(new_deaths_list)

df['new_cases'] = new_cases_list
df['new_deaths'] = new_deaths_list


# From now on, here is what the dataset looks like:

# In[ ]:


df.head(20)


# ---

# # Visualizations <a class="anchor" id="fourth-bullet"></a>

# * ## **Covid-19 pandemic evolution since first record <a class="anchor" id="fifth-bullet"></a>**

# In[ ]:


import plotly.graph_objects as go

df = df.sort_values('date')

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'].unique(), 
                         y=df.groupby('date')['cases'].sum(), fill='tozeroy', name="Cases")) # fill down to xaxis
fig.add_trace(go.Scatter(x=df['date'].unique(), 
                         y=df.groupby('date')['deaths'].sum(), fill='tozeroy', name="Deaths")) # fill down to xaxis
fig.update_layout(
    annotations=[
        dict(
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Day",
            xref="paper",
            yref="paper"
        ),
        dict(
            x=-0.07,
            y=0.5,
            showarrow=False,
            text="Cases/Deaths Recorded",
            textangle=-90,
            xref="paper",
            yref="paper"
        )
    ],
    title_text="Daily Cumulative Cases/Deaths Recorded"
)

fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df['date'].unique(), 
                         y=df.groupby('date')['new_cases'].sum(), fill='tozeroy', name="New Cases")) # fill down to xaxis
fig2.add_trace(go.Scatter(x=df['date'].unique(), 
                         y=df.groupby('date')['new_deaths'].sum(), fill='tozeroy', name="New Deaths")) # fill down to xaxis
fig2.update_layout(
    annotations=[
        dict(
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Day",
            xref="paper",
            yref="paper"
        ),
        dict(
            x=-0.07,
            y=0.5,
            showarrow=False,
            text="New cases/New Deaths Recorded",
            textangle=-90,
            xref="paper",
            yref="paper"
        )
    ],
    title_text="Daily New Cases/New Deaths Recorded"
)

fig2.show()


# In[ ]:


print('''Unfortunately, we can easily notice an increase in the number of new cases recorded since \033[1mmid-June 2020\033[0m. \nThe highest number of new cases (for now) has been recorded on \033[1m{}\033[0m (with almost \033[1m{}\033[0m new cases in one day). \nAs of today, there were almost \033[1m{}\033[0m cases and more than \033[1m{}k\033[0m deaths.''' 
      .format(df.groupby('new_cases')['date'].sum().max(), round(df.groupby('date')['new_cases'].sum().max(), -4), \
              round(df.groupby('date')['cases'].sum().max(), -5), str(df.groupby('date')['deaths'].sum().max())[:3]))


# * ## US Interactive Mapping <a class="anchor" id="sixth-bullet"></a>

# In[ ]:


import plotly.graph_objects as go
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df_last_record = df[df['date'] == df['date'].max()]

df_last_record = df_last_record.sort_values('state', ascending=True)

fig = go.Figure(data=go.Choropleth(
    locations=df_last_record['code'].unique(), # Spatial coordinates
    z = df_last_record.groupby("state")["cases"].sum(), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    text = df_last_record['state'].unique(),
    colorbar_title = "Millions USD",
))

fig.update_layout(
    title_text = df_last_record.iloc[0,0] + ' : ' + 'Cases Count by State since first record',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)

fig.show()

df_last_record['text_graph_2'] = 'County: ' + df_last_record['agg_county_code'] + '<br>' +                 'Death Rate: ' + round(((df_last_record['deaths'] / df_last_record['cases']) * 100), 2).astype(str) + '%' + '<br>' +                 'Cases number: ' + df_last_record['cases'].astype(str) + '<br>' +                 'Deaths number: ' + df_last_record['deaths'].astype(str) + '<br>' +                 'fips: ' + df_last_record['fips'].astype(str)

fig_2 = go.Figure(go.Choroplethmapbox(geojson=counties, locations=df_last_record['fips'], z=(df_last_record['deaths'] / df_last_record['cases']) * 100,
                                    colorscale='YlOrRd', zmin=0, zmax=8, hoverinfo='text', text=df_last_record['text_graph_2'],
                                    marker_opacity=0.5, marker_line_width=0.2))
fig_2.update_layout(mapbox_style="carto-positron",
                    mapbox_zoom=2.8, 
                    mapbox_center = {"lat": 41.0902, "lon": -97.7129},
                    title_text = df_last_record.iloc[0,0] + ' : ' + 'Death Rate due to Coronavirus since first record'
                   )

#fig_2.show()


# **Aside note :**
# * On map 2, fips county code have not been reviewed. Some may differ from GeoJSON (which is used to display the data), some are missing.
# * Some county names from the original dataset differ from GeoJSON county names.
# 
# For these reasons mentioned above, some counties are not displayed on map 2.

# * ## Animated Visualizations <a class="anchor" id="seventh-bullet"></a>

# For some of the coming visualizations, we built Bar Charts Race with [Flourish Studio](https://flourish.studio/). These are really useful when trying to visualize evolutions and rankings on different dimensions.

# In[ ]:


from IPython.core.display import HTML


# In[ ]:


# Bar Chart Race 1: Reshaping our dataset to fit with Flourish structure requirements.

df_an_chart1 = df.groupby(
   ['date', 'state']
).agg(
    {
         'cases':sum
    }
)

df_an_chart1 = df_an_chart1.reset_index()
df_an_chart1 = df_an_chart1.pivot(index='state', columns='date', values='cases')
df_an_chart1 = df_an_chart1.fillna(0)
#df_an_chart1.to_csv('df_pivot_1.csv', header=True)
df_an_chart1.head(20)


# In[ ]:


HTML("<iframe src='https://flo.uri.sh/visualisation/3052538/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/3052538/?utm_source=embed&utm_campaign=visualisation/3052538' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>")


# In[ ]:


# Bar Chart Race 2: Reshaping our dataset to fit with Flourish structure requirements.

df_an_chart2 = df.pivot(index='agg_county_code', columns='date', values='cases')
df_an_chart2 = df_an_chart2.fillna(0)
#df_an_chart2.to_csv('df_pivot_2.csv', header=True)
df_an_chart2.head(20)


# In[ ]:


HTML("<iframe src='https://flo.uri.sh/visualisation/3018715/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/3018715/?utm_source=embed&utm_campaign=visualisation/3018715' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>")


# In[ ]:


import plotly.express as px

df_mi_counties = df_last_record.nlargest(50, 'cases')
list_mi_counties = list(df_mi_counties.iloc[:, 1])
list_associated_states = list(df_mi_counties.iloc[:, 2])

start_date = '2020-04-01'
end_date = df['date'].max()
df_mi_counties = df[df['county'].isin(list_mi_counties) & df['state'].isin(list_associated_states)]
mask = (df_mi_counties['date'] >= start_date) & (df_mi_counties['date'] <= end_date)
df_mi_counties = df_mi_counties[mask]
df_mi_counties = df_mi_counties[~df_mi_counties['county'].isin(['New York City', 'Unknown'])]
df_mi_counties

px.scatter(df_mi_counties, x=df_mi_counties['cases'], y=df_mi_counties['deaths'], animation_frame=df_mi_counties['date'], animation_group=df_mi_counties['county'],
           size=round((df_mi_counties['deaths'] / df_mi_counties['cases']), 3) * 100, color='state', hover_name=df_mi_counties['county'],
           log_x=False, size_max = 40, range_x=[0,df_mi_counties['cases'].max()], range_y=[0,df_mi_counties['deaths'].max()],
           title='50 Most Impacted Counties since ' + start_date + ': Death Rate Evolution')


# **Aside note :**
# * New York City, as well as 'Unknown' counties have been removed from the third animated graph in order to keep it relevant
# * Bubble sizes correspond to death rate

# # Next Steps <a class="anchor" id="eighth-bullet"></a> 

# This is version 1 of this kernel. Further analyses and visualizations will complete this study. <br />Many thanks for having read it. Hope you found it useful.
