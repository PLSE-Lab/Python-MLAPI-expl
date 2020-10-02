#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px


# # Cases

# ## Canadian Data
# 
# Data Source: Berry I, Soucy J-PR, Tuite A, Fisman D. Open access epidemiologic data and an interactive dashboard to monitor the COVID-19 outbreak in Canada. CMAJ. 2020 Apr 14;192(15):E420. doi: https://doi.org/10.1503/cmaj.75262 (updated on Kaggle via the HowsMyFlattening team) [[GitHub]](https://github.com/ishaberry/Covid19Canada)

# In[ ]:


canada_cases_df = pd.read_csv('/kaggle/input/covid19-challenges/test_data_canada.csv')
provincial_cases = canada_cases_df.groupby(['province', 'date'])['case_id'].count().unstack().T.fillna(0)
all_regional_cases = canada_cases_df.groupby(['region', 'date'])['case_id'].count().unstack().T.fillna(0)
on_regional_cases = canada_cases_df[canada_cases_df['province']=='Ontario'].groupby(['region', 'date'])['case_id'].count().unstack().T.fillna(0)
can_summ = pd.concat((provincial_cases, all_regional_cases), axis=1)


# We love the [famous FT.com visualization](https://www.ft.com/coronavirus-latest), so we're stealing its methodology.  Tracks start when the rolling 7-day (equally weighted) average of confirmed cases hits 30.  This lines up our time series to this common starting criterion.

# In[ ]:


window_in_days = 7
rolling_can_summ = can_summ.rolling(window_in_days).mean()
rolling_can_days_since_30_cases_df = pd.DataFrame(index=range(len(rolling_can_summ.dropna())), columns=rolling_can_summ.columns)
for sr in rolling_can_days_since_30_cases_df.columns:
    if sr in ['NWT', 'Yukon', 'Nunavut']:
        # Territories are listed twice, once as a province and once as a region.  Neither hits the threshold
        continue
    srser = rolling_can_summ[sr]
    idx_first_30_cases = np.argmax(srser.values >= 30)
    # argmax returns 0 when there is no match, so check that this is real
    if srser.iloc[idx_first_30_cases] >= 30:
        diff_length = len(rolling_can_days_since_30_cases_df.index) - len(srser.iloc[idx_first_30_cases:])
        rolling_can_days_since_30_cases_df[sr] = list(srser.iloc[idx_first_30_cases:]) + [np.nan]*diff_length


# Next we pick out "interesting" tracks to plot in colour.  For these, I choose all provinces, all Ontario health regions (because [howsmyflattening.ca](https://howsmyflattening.ca/) focuses on Ontario), Vancouver Coastal, and Calgary.  Vancouver is interesting because of the highly successful flattening.  Calgary is interesting because I have family there.

# In[ ]:


# identify regions we want to colour on the chart... provinces and regions of Ontario, 
# plus Vancouver Coastal, which really stands out as seriously flattened
# and Calgary, because Alf's family lives there :-P
interesting_regions = ['Vancouver Coastal', 'Calgary']
pmax = provincial_cases.rolling(7).mean().max()
interesting_regions = interesting_regions + pmax[pmax>=30].index.to_list()
rmax = on_regional_cases.rolling(7).mean().max()
interesting_regions = interesting_regions + rmax[rmax>=30].index.to_list()
interesting_regions


# In[ ]:


colour_palette = px.colors.qualitative.G10
traces = []
colour_i = 0
for c in rolling_can_days_since_30_cases_df.columns:
    srser = rolling_can_days_since_30_cases_df[c].dropna()
    if len(srser) > 0:
        if c in interesting_regions:
            colour = colour_palette[colour_i]
        else:
            colour = '#B3B3B3'
        traces.append(go.Scatter(x=srser.index, y=srser.values, name=c,
                                 text=[c]*len(srser.index), mode='lines',
                                 hovertemplate = "<b>%{text}</b> Days Since 30 Cases: %{x}; Rolling Daily Cases: %{y:.1f}",
                                 line=dict(color=colour),
        ))
        if c in interesting_regions:
            traces.append(go.Scatter(
                x=[srser.index[-1]],
                y=[srser.values[-1]],
                text=[c],
                mode="text",
                textposition="top right",
                textfont=dict(color=colour),
                showlegend=False,
            ))
            colour_i += 1
            if colour_i >= len(colour_palette):
                colour_i = 0
layout = go.Layout(title='Daily COVID-19 Cases: %d-Day Rolling Average' % window_in_days,
                   xaxis_title="Days Since 30 Cases", 
                   yaxis_title="Average Daily Cases (%d-day rolling)" % window_in_days,
                   yaxis_type='log',
                  )
fig = go.Figure(data=traces, layout=layout)
#fig.update_yaxes(range=[0, 12])
fig.show()


# # Canadian Outbreak Visualization
# 
# This chart shows the state of the COVID-19 outbreak in Canada.  It shows the evolution of the outbreak for every province and health region that has met or exceeded a rolling 7-day average of 30 cases per day detected.  This is the same method used to generate the [famous FT.com coronavirus charts](https://www.ft.com/coronavirus-latest).
# 
# Each line starts on the day that the 7-day rolling average of cases found first hits 30 or more.  The vertical axis shows how this rolling average number of cases found per day is changing.  Declining detections, such as seen with the Vancouver Coastal region, show that while new cases are still being found, there are now fewer found than in the past.  Quebec and Ontario show a tendency to "flat" curves, meaning that each day is similar to the last in terms of cases found.
# 
# The goal of "flattening the curve" is first to achieve this steady state, and then to enter declining detections.  Look for provinces or regions to "turn the corner" and start a decline.
# 
# Data Source: Berry I, Soucy J-PR, Tuite A, Fisman D. Open access epidemiologic data and an interactive dashboard to monitor the COVID-19 outbreak in Canada. CMAJ. 2020 Apr 14;192(15):E420. doi: https://doi.org/10.1503/cmaj.75262 (updated on Kaggle via the HowsMyFlattening team) [[GitHub]](https://github.com/ishaberry/Covid19Canada)

# # Deaths
# 
# Now rinse and repeat using the mortality data set.

# In[ ]:


canada_mortality_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_mortality.csv')
provincial_deaths = canada_mortality_df.groupby(['province', 'date'])['death_id'].count().unstack().T.fillna(0)
all_regional_deaths = canada_mortality_df.groupby(['region', 'date'])['death_id'].count().unstack().T.fillna(0)
on_regional_deaths = canada_mortality_df[canada_mortality_df['province']=='Ontario'].groupby(['region', 'date'])['death_id'].count().unstack().T.fillna(0)
can_death_summ = pd.concat((provincial_deaths, all_regional_deaths), axis=1)


# As before, line the time series up using a common starting criterion.  Again, following [the FT.com charts](https://www.ft.com/coronavirus-latest), use a starting point of the day when the 7-day (equally weighted) rolling average deaths reached 3.

# In[ ]:


window_in_days = 7
rolling_can_deaths = can_death_summ.rolling(window_in_days).mean()
rolling_can_days_since_3_deaths_df = pd.DataFrame(index=range(len(rolling_can_deaths.dropna())), columns=rolling_can_deaths.columns)
for sr in rolling_can_days_since_3_deaths_df.columns:
    if sr in ['NWT', 'Yukon', 'Nunavut']:
        # Territories are listed twice, once as a province and once as a region.  Neither hits the threshold
        continue
    srser = rolling_can_deaths[sr]
    idx_first_3_deaths = np.argmax(srser.values >= 3)
    # argmax returns 0 when there is no match, so check that this is real
    if srser.iloc[idx_first_3_deaths] >= 3:
        diff_length = len(rolling_can_days_since_3_deaths_df.index) - len(srser.iloc[idx_first_3_deaths:])
        rolling_can_days_since_3_deaths_df[sr] = list(srser.iloc[idx_first_3_deaths:]) + [np.nan]*diff_length


# In[ ]:


colour_palette = px.colors.qualitative.G10
traces = []
colour_i = 0
for c in rolling_can_days_since_3_deaths_df.columns:
    srser = rolling_can_days_since_3_deaths_df[c].dropna()
    if len(srser) > 0:
        if c in interesting_regions:
            colour = colour_palette[colour_i]
        else:
            colour = '#B3B3B3'
        traces.append(go.Scatter(x=srser.index, y=srser.values, name=c,
                                 text=[c]*len(srser.index), mode='lines',
                                 hovertemplate = "<b>%{text}</b> Days Since 3 Deaths: %{x}; Rolling Daily Deaths: %{y:.1f}",
                                 line=dict(color=colour),
        ))
        if c in interesting_regions:
            traces.append(go.Scatter(
                x=[srser.index[-1]],
                y=[srser.values[-1]],
                text=[c],
                mode="text",
                textposition="top right",
                textfont=dict(color=colour),
                showlegend=False,
            ))
            colour_i += 1
            if colour_i >= len(colour_palette):
                colour_i = 0
layout = go.Layout(title='Daily COVID-19 Deaths: %d-Day Rolling Average' % window_in_days,
                   xaxis_title="Days Since 3 Deaths", 
                   yaxis_title="Average Daily Deaths (%d-day rolling)" % window_in_days,
                   yaxis_type='log',
                  )
fig = go.Figure(data=traces, layout=layout)
#fig.update_yaxes(range=[0, 12])
fig.update_xaxes(range=[0, 18])
fig.show()


# # Canadian Outbreak Visualization
# 
# This chart shows the state of the COVID-19 outbreak in Canada.  It shows the evolution of the outbreak for every province and health region that has met or exceeded a rolling 7-day average of 3 deaths per day.  This is the same method used to generate the [famous FT.com coronavirus charts](https://www.ft.com/coronavirus-latest).
# 
# Each line starts on the day that the 7-day rolling average of death  first hits 3 or more.  The vertical axis shows how this rolling average number of deaths found per day is changing.  Declining deaths, such as seen with the Haliburton Kawartha Pine Ridge region, show that deaths can eventually taper off.
# 
# The goal of "flattening the curve" is first to achieve a steady state, and then to enter declining deaths.  Look for provinces or regions to "turn the corner" and start a decline.
# 
# Data Source: Berry I, Soucy J-PR, Tuite A, Fisman D. Open access epidemiologic data and an interactive dashboard to monitor the COVID-19 outbreak in Canada. CMAJ. 2020 Apr 14;192(15):E420. doi: https://doi.org/10.1503/cmaj.75262 (updated on Kaggle via the HowsMyFlattening team) [[GitHub]](https://github.com/ishaberry/Covid19Canada)
