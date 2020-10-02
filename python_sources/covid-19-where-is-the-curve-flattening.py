#!/usr/bin/env python
# coding: utf-8

# # Where is the curve flattening?
# 
# Inflection-sensitive chart for detecting successful interventions, from the article "How To Tell If We're Beating COVID-19".
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from datetime import datetime
from IPython.display import HTML
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


states_url = "https://covidtracking.com/api/states/daily"
case_threshold = 100 # TODO I don't want to drop states below 100

r = requests.get(states_url)
states_df = pd.DataFrame(r.json())
states_df['date'] = pd.to_datetime(states_df.date, format="%Y%m%d")
states_df = states_df[['date', 'state', 'positive']].sort_values('date')
states_df = states_df.rename(columns={'positive': 'confirmed'})
cols = {}
for state in states_df.state.unique():
    cases = states_df[(states_df.state == state) & (states_df.confirmed > case_threshold)]
    cases = cases.reset_index().confirmed.reset_index(drop=True)
    if len(cases) > 1:
        cols[state] = cases

df = states_df.reset_index()


# In[ ]:


df = (df.assign(daily_new=df.groupby('state', as_index=False)[['confirmed']]
                            .diff().fillna(0)
                            .reset_index(0, drop=True)))


# In[ ]:


df = (df.assign(avg_daily_new=df.groupby('state', as_index=False)[['daily_new']]
                                .rolling(7).mean()
                                .reset_index(0, drop=True)))


# In[ ]:


state_names = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AS": "American Samoa",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District Of Columbia",
    "FM": "Federated States Of Micronesia",
    "FL": "Florida",
    "GA": "Georgia",
    "GU": "Guam",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MH": "Marshall Islands",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "MP": "Northern Mariana Islands",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PW": "Palau",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VI": "Virgin Islands",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming"
}


# In[ ]:


df['day'] = df.date.apply(lambda x: x.date()).apply(str)
df = df.sort_values(by='day')
dfc = df[df.avg_daily_new > 0]


# In[ ]:


days = dfc.day.unique().tolist()
states = dfc.state.unique().tolist()
states.sort()


# In[ ]:


fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
}

# fill in most of layout
fig_dict["layout"]["height"] = 700
fig_dict["layout"]["width"] = 900
fig_dict["layout"]["xaxis"] = {"range": [np.log10(5), np.log10(dfc['confirmed'].max() + 5000)], "title": "Total Confirmed Cases (log scale)", "type": "log"}
fig_dict["layout"]["yaxis"] = {"range": [np.log10(1), np.log10(dfc['avg_daily_new'].max() + 500)], "title": "Average Daily New Cases (log scale)", "type": "log"}
fig_dict["layout"]["hovermode"] = "closest"
fig_dict["layout"]["sliders"] = {
    "args": [
        "transition", {
            "duration": 100,
            "easing": "cubic-in-out"
        }
    ],
    "initialValue": min(days),
    "plotlycommand": "animate",
    "values": days,
    "visible": True
}

# buttons
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 300, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 300,
                                                                    "easing": "linear"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.05,
        "xanchor": "right",
        "y": 0.05,
        "yanchor": "top"
    }
]

# sliders
sliders_dict = {
    "active": len(days)-1,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
#         "prefix": "Date: ",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 100},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}


# In[ ]:


day = max(days)
for state in states:
    dataset_by_day = dfc[dfc["day"] <= day]
    dataset_by_day_and_state = dataset_by_day[ dataset_by_day["state"]==state ]
    
    data_dict = {
        "x": list(dataset_by_day_and_state["confirmed"]),
        "y": list(dataset_by_day_and_state["avg_daily_new"]),
        "mode": "lines",
        "text": dataset_by_day_and_state[['confirmed', 'avg_daily_new']],
        "name": state,
        'hoverlabel': {'namelength': 0},
        'hovertemplate': '<b>%{hovertext}</b><br>Confirmed: %{x:,d}<br>Average Daily: %{y:,.2f}',
        'hovertext': dataset_by_day_and_state['state'].apply(lambda s: state_names.get(s, '??') + f' ({s})')
    }
    fig_dict["data"].append(data_dict)

# make frames
for day in days:
    frame = {"data": [], "name": day}
    for state in states:
        dataset_by_day = dfc[dfc["day"] <= day]
        dataset_by_day_and_state = dataset_by_day[
            dataset_by_day["state"] == state]

        data_dict = {
            "x": list(dataset_by_day_and_state["confirmed"]),
            "y": list(dataset_by_day_and_state["avg_daily_new"]),
            "mode": "lines",
            "text": dataset_by_day_and_state[['confirmed', 'avg_daily_new']],
            "name": state
        }
        frame["data"].append(data_dict)

    fig_dict["frames"].append(frame)
    slider_step = {"args": [
        [day],
        {"frame": {"duration": 100, "redraw": True},
         "mode": "immediate",
         "transition": {"duration": 100, 'easing': 'linear'}}
    ],
        "label": day,
        "method": "animate"}
    sliders_dict["steps"].append(slider_step)


# In[ ]:


fig_dict["layout"]["sliders"] = [sliders_dict]
fig = go.Figure(fig_dict)


# States/countries will drift off the diagonal when they are flattening the curve.
# 
# Only entries with at least 100 confirmed cases are considered.
# 
# The top 5 entries are initially highlighted.

# In[ ]:


africa_names = [
    'Algeria',
    'Angola',
    'Benin',
    'Botswana',
    'Burkina Faso',
    'Burundi',
    'Cabo Verde',
    'Cameroon',
    'Central African Republic',
    'Chad',
    'Congo (Brazzaville)',
    'Congo (Kinshasa)',
    'Djibouti',
    'Egypt',
    'Equatorial Guinea',
    'Eritrea',
    'Eswatini',
    'Ethiopia',
    'Gabon',
    'Gambia',
    'Ghana',
    'Guinea',
    'Guinea-Bissau',
    'Ivory Coast',
    'Kenya',
    'Liberia',
    'Libya',
    'Madagascar',
    'Malawi',
    'Mali',
    'Mauritania',
    'Mauritius',
    'Morocco',
    'Mozambique',
    'Namibia',
    'Niger',
    'Nigeria',
    'Rwanda',
    'Sao Tome and Principe',
    'Senegal',
    'Seychelles',
    'Sierra Leone',
    'Somalia',
    'South Africa',
    'South Sudan',
    'Sudan',
    'Tanzania',
    'Togo',
    'Tunisia',
    'Uganda',
    'Western Sahara',
    'Zambia',
    'Zimbabwe'
]


# In[ ]:


america_names = [
    'Antigua and Barbuda',
    'Argentina',
    'Bahamas',
    'Barbados',
    'Belize',
    'Bolivia',
    'Brazil',
    'Canada',
    'Chile',
    'Colombia',
    'Costa Rica',
    'Cuba',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'El Salvador',
    'Grenada',
    'Guatemala',
    'Guyana',
    'Haiti',
    'Honduras',
    'Jamaica',
    'Mexico',
    'Nicaragua',
    'Panama',
    'Paraguay',
    'Peru',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Vincent and the Grenadines',
    'Suriname',
    'Trinidad and Tobago',
    'United States of America',
    'Uruguay',
    'Venezuela'
]


# In[ ]:



asiapacific_names = [
    'Afghanistan',
    'Armenia',
    'Australia',
    'Azerbaijan',
    'Bahrain',
    'Bangladesh',
    'Bhutan',
    'Brunei',
    'Cambodia',
    'China',
    'Cyprus',
    'East Timor',
    'Fiji',
    'Georgia',
    'Hong Kong',
    'India',
    'Indonesia',
    'Iran',
    'Iraq',
    'Israel',
    'Japan',
    'Jordan',
    'Kazakhstan',
    'Kuwait',
    'Kyrgyzstan',
    'Laos',
    'Lebanon',
    'Malaysia',
    'Maldives',
    'Mongolia',
    'Myanmar',
    'Nepal',
    'New Zealand',
    'Oman',
    'Pakistan',
    'Papua New Guinea',
    'Philippines',
    'Qatar',
    'Russia',
    'Saudi Arabia',
    'Singapore',
    'South Korea',
    'Sri Lanka',
    'Syria',
    'Taiwan',
    'Thailand',
    'Turkey',
    'United Arab Emirates',
    'Uzbekistan',
    'Vietnam',
    'West Bank and Gaza'
]


# In[ ]:



europe_names = [
    'Albania',
    'Andorra',
    'Armenia',
    'Austria',
    'Azerbaijan',
    'Belarus',
    'Belgium',
    'Bosnia Herzegovina',
    'Bulgaria',
    'Croatia',
    'Cyprus',
    'Czechia',
    'Denmark',
    'Estonia',
    'Finland',
    'France',
    'Georgia',
    'Germany',
    'Greece',
    'Hungary',
    'Iceland',
    'Ireland',
    'Italy',
    'Kazakhstan',
    'Kosovo',
    'Latvia',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'Malta',
    'Moldova',
    'Monaco',
    'Montenegro',
    'Netherlands',
    'North Macedonia',
    'Norway',
    'Poland',
    'Portugal',
    'Romania',
    'Russia',
    'San Marino',
    'Serbia',
    'Slovakia',
    'Slovenia',
    'Spain',
    'Sweden',
    'Switzerland',
    'Turkey',
    'Ukraine',
    'United Kingdom',
    'Vatican City'
]


# In[ ]:


import pandas as pd
import numpy as np

def load_individual_timeseries(name):
    base_url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series'
    url = f'{base_url}/time_series_covid19_{name}_global.csv'
    df = pd.read_csv(url, 
                     index_col=['Country/Region', 'Province/State', 'Lat', 'Long'])
    df['type'] = name.lower()
    df.columns.name = 'date'
    
    df = (df.set_index('type', append=True)
            .reset_index(['Lat', 'Long'], drop=True)
            .stack()
            .reset_index()
            .set_index('date')
         )
    df.index = pd.to_datetime(df.index)
    df.columns = ['country', 'state', 'type', 'cases']
    
    # Move HK to country level
    df.loc[df.state =='Hong Kong', 'country'] = 'Hong Kong'
    df.loc[df.state =='Hong Kong', 'state'] = np.nan
    
    # Aggregate large countries split by states
    df = pd.concat([df, 
                    (df.loc[~df.state.isna()]
                     .groupby(['country', 'date', 'type'])
                     .sum()
                     .rename(index=lambda x: x+' (total)', level=0)
                     .reset_index(level=['country', 'type']))
                   ])
    return df

def load_data(drop_states=False, p_crit=.05, filter_n_days_100=None):
    df = load_individual_timeseries('confirmed')
    df = df.rename(columns={'cases': 'confirmed'})
    if drop_states:
        # Drop states for simplicity
        df = df.loc[df.state.isnull()]
        
    # Estimated critical cases
    df = df.assign(critical_estimate=df.confirmed*p_crit)

    # Compute days relative to when 100 confirmed cases was crossed
    df.loc[:, 'days_since_100'] = np.nan
    for country in df.country.unique():
        if not df.loc[(df.country == country), 'state'].isnull().all():
            for state in df.loc[(df.country == country), 'state'].unique():
                df.loc[(df.country == country) & (df.state == state), 'days_since_100'] =                     np.arange(-len(df.loc[(df.country == country) & (df.state == state) & (df.confirmed < 100)]), 
                              len(df.loc[(df.country == country) & (df.state == state) & (df.confirmed >= 100)]))
        else:
            df.loc[(df.country == country), 'days_since_100'] =                 np.arange(-len(df.loc[(df.country == country) & (df.confirmed < 100)]), 
                          len(df.loc[(df.country == country) & (df.confirmed >= 100)]))

    # Add recovered cases
#     df_recovered = load_individual_timeseries('Recovered')
#     df_r = df_recovered.set_index(['country', 'state'], append=True)[['cases']]
#     df_r.columns = ['recovered']

    # Add deaths
    df_deaths = load_individual_timeseries('deaths')
    df_d = df_deaths.set_index(['country', 'state'], append=True)[['cases']]
    df_d.columns = ['deaths']

    df = (df.set_index(['country', 'state'], append=True)
#             .join(df_r)
            .join(df_d)
            .reset_index(['country', 'state'])
    )
    
    if filter_n_days_100 is not None:
        # Select countries for which we have at least some information
        countries = pd.Series(df.loc[df.days_since_100 >= filter_n_days_100].country.unique())
        df = df.loc[lambda x: x.country.isin(countries)]

    return df


# In[ ]:


df = load_individual_timeseries('confirmed')
df = df[~df['country'].str.contains(' \(total\)')].drop(['state', 'type'], axis=1).reset_index()

# clean data
df['country'] = df['country'].replace({'Bosnia and Herzegovina':'Bosnia Herzegovina'      })
df['country'] = df['country'].replace({'Timor-Leste'           :'East Timor'              })
df['country'] = df['country'].replace({"Cote d'Ivoire"         :'Ivory Coast'             })
df['country'] = df['country'].replace({'Burma'                 :'Myanmar'                 })
df['country'] = df['country'].replace({'Korea, South'          :'South Korea'             })
df['country'] = df['country'].replace({'Taiwan*'               :'Taiwan'                  })
df['country'] = df['country'].replace({'US'                    :'United States of America'})
df['country'] = df['country'].replace({'Holy See'              :'Vatican City'            })

# append usa-by-states data
dfc['country'] = dfc['state'].map(state_names)
df = (df.rename(columns={'cases': 'confirmed'})
        .append(dfc.drop(['index', 'state', 'daily_new', 'avg_daily_new', 'day'], axis=1)
                   .astype({'confirmed': 'int64'})))

# aggregate data
df = (df.sort_values(by=['country', 'date'])
        .groupby(['country', 'date'])['confirmed']
        .agg(sum)).reset_index()

# additional measurements
df = df.assign(daily_new_abs=(df.groupby('country', as_index=False)[['confirmed']]
                                .diff()
                                .fillna(0)
                                .astype('int64')))
df = df.assign(daily_new_avg=(df.groupby('country', as_index=False)[['daily_new_abs']]
                                .rolling(7)
                                .mean()
                                .fillna(0)
                                .round(decimals=2)
                                .reset_index(drop=True)))

# slice data
df_usa         = df[(df.confirmed > case_threshold) & (df.daily_new_avg > 0) & (df['country'].isin(state_names.values()))]
df_africa      = df[(df.confirmed > case_threshold) & (df.daily_new_avg > 0) & (df['country'].isin(africa_names        ))]
df_america     = df[(df.confirmed > case_threshold) & (df.daily_new_avg > 0) & (df['country'].isin(america_names       ))]
df_asiapacific = df[(df.confirmed > case_threshold) & (df.daily_new_avg > 0) & (df['country'].isin(asiapacific_names   ))]
df_europe      = df[(df.confirmed > case_threshold) & (df.daily_new_avg > 0) & (df['country'].isin(europe_names        ))]

df.to_csv('data.csv')


# In[ ]:


#hide
import altair as alt

def make_chart(data=df):

    countries = data.country.unique().tolist()

    highlighted = data.sort_values('confirmed', ascending=False).groupby('country').head(1).country.tolist()[:5]

    selection = alt.selection_multi(bind='legend',
                                    fields=['country'],
                                    init=[{'country': x} for x in highlighted])

    base = (alt.Chart(data=data)
               .properties(width=550)
               .encode(x=alt.X(scale=alt.Scale(type='log'),
                               shorthand='confirmed:Q',
                               title='Total Confirmed Cases (log scale)'),
                       y=alt.Y(scale=alt.Scale(type='log'),
                               shorthand='daily_new_avg:Q',
                               title='Average Daily New Cases (log scale)'),
                       color=alt.Color(legend=alt.Legend(columns=3,
                                                         symbolLimit=len(countries),
                                                         title='Country/State:'),
                                       scale=alt.Scale(scheme='category20b'),
                                       shorthand='country:N'),
                       tooltip=list(data),
                       opacity=alt.condition(selection, alt.value(1), alt.value(0.05))))

    chart = (base.mark_line()
                 .add_selection(selection)
                 .configure_legend(labelFontSize=10,
                                   titleFontSize=12)
                 .configure_axis(labelFontSize=10,
                                 titleFontSize=12))

    return chart


# #  United States of America

# In[ ]:


#hide_input
make_chart(df_usa)


# # Africa

# In[ ]:


#hide_input
make_chart(df_africa)


# # South and North America

# In[ ]:


#hide_input
make_chart(df_america)


# # Asia-Pacific

# In[ ]:


#hide_input
make_chart(df_asiapacific)


# # Europe

# In[ ]:


#hide_input
make_chart(df_europe)


# ## Explanation
# 
# The exponential growth stage of a pandemic must end sometime, either as the virus runs out of people to infect, or as societies get it under control. However, it can be difficult to tell exactly when exponential growth is ending, for several reasons:
# 
# * Humans aren't wired to understand exponentials at a glance.
# * It can be difficult to compare regions with differing first-infection dates, testing rates, and populations.
# * The news tends to report individual data points, without the contextual information necessary to interpret it.
# * If the plot doesn't explicitly plot the rate of new cases, a change must be quite dramatic before it becomes distinguishable.
# 
# This visualization plots the (sliding average of) daily new cases against the total cases, for each US state (with other countries and regions to come). This has the advantage of aligning all of them onto a baseline trajectory of exponential growth, with a very clear downward plummet when a given state gets the virus under control. As explained in the caveats below, this visualization has a very specific purpose: to make it clear whether a given state has managed to exit the exponential trajectory or not.
# 
# _minutephysics_ has an excellent video on this visualization type, [How To Tell If We're Beating COVID-19](https://youtu.be/54XLXg4fYsc).

# ## Caveats
# 
# 1. The logarithmic scales can make it seem as if states are closer together than they actually are. For example, at time of writing (April 5th) New York (the leader in US cases) and New Jersey (the runner-up) look as though it's a close race, but New York has over three times as any cases as New Jersey.
# 2. The logarithmic scale can also obscure a resurgence of infections after a significant downturn, since the trace won't move much to the right during a short period late in time.
# 3. Time is represented by the animation, not by the x-axis, which is unusual for most charts made about COVID-19. This is the plot's main advantage, because it aligns states onto _roughly_ the same trajectory regardless of population or testing rate, but it may be unexpected.
# 4. The true number of cases is unknown, so the actual slope of the log-log change plot is unknown. All states are also increasing their testing rate over time, so these data may imply that the infection rate is increasing faster than it actually is.
# 5. The data these plots rely on are incomplete, and come in less smoothly than they may imply. Healthcare systems around the world collect and report data when they can.
# 6. This chart plots the logarithm of the sliding window average of the daily growth rate on the y-axis, not the raw daily growth rate, because there's too much variability day-to-day to visually detect the trend. This also makes the plot a pessimistic estimate of where each state is on its trajectory.

# ## Appendix USA - Animated visualization of growth
# 

# In[ ]:


#hide
fig.show()


# The animated visualization for the US and the descriptions were made by Daniel Cox, with thanks to Henry of minutephysics for How To Tell If We're Beating COVID-19, and covidtracking.com for US data.
# 
# **Code Referance & courtesy & Credit : Daniel Cox, Martin Boehler**
# 
# 
# The static visualizations for the US, Africa, America, Asia-Pacific and Europe were made by Martin Boehler, with thanks to Daniel Cox for this great inspiration and implementation, and Johns Hopkins University CSSE for the 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository.

# **Note: This work is highly inspired from few other kaggle kernels , github sources and other data science resources. Any traces of replications, which may appear , is purely co-incidental. Due respect & credit to all my fellow kagglers.**
