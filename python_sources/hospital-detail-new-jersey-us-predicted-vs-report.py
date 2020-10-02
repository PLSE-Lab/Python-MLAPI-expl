#!/usr/bin/env python
# coding: utf-8

# ## I looked into the prediction made by IHME and compared it with state reported values for New Jersey State, USA.
# [https://covid19.healthdata.org/united-states-of-america/new-jersey](http://)
# 
# [https://www.nj.gov/health/cd/topics/covid2019_dashboard.shtml](http://)
# # Let me know if there are dataset sources related to daily update on hospital admissions, number of ICU patients, ventilator usage etc details for specific locality. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objects as go

import plotly.offline as ply
ply.init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


base_path = "/kaggle/input/"
ihme_path_04_09 = base_path + "ihme-covid/ihme_covid/2020_04_09/" + "Hospitalization_all_locs.csv"
ihme_path_03_30 = base_path + "ihme-covid/ihme_covid/2020_03_30/" + "Hospitalization_all_locs.csv"

ihme_df_04_09 = pd.read_csv(ihme_path_04_09)
ihme_df_03_30 = pd.read_csv(ihme_path_03_30)
ihme_df_04_09.head(2)


# In[ ]:


ihme_df_04_09_nj=ihme_df_04_09[ihme_df_04_09.location_name=='New Jersey']
ihme_df_03_30_nj=ihme_df_03_30[ihme_df_03_30.location_name=='New Jersey']
ihme_df_04_09_nj.loc[:,'ICUbed_available']=465.0


# ## Values for April 09

# In[ ]:


ihme_df_04_09_nj.loc[ihme_df_04_09_nj['date']=='2020-04-09'].T[3:12].T


# In[ ]:


ihme_df_04_09_nj.loc[ihme_df_04_09_nj['date']=='2020-04-09'].T[12:21].T


# In[ ]:


ihme_df_04_09_nj.loc[ihme_df_04_09_nj['date']=='2020-04-09'].T[21:].T


# ## How the prediction has changed from one made on March 30th to the one on April 9th

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["allbed_mean"],
    line_color='rgb(0,100,80)',
    name='04_09',
))
fig.add_trace(go.Scatter(
    x=ihme_df_03_30_nj["date"], y=ihme_df_03_30_nj["allbed_mean"],
    line_color='rgb(231,107,243)',
    name='03_30',
))
fig.update_traces(mode='lines')
fig.show()


# ### ICU bed requirement prediction

# In[ ]:


# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["newICU_mean"],
#     line_color='rgb(0,100,80)',
# #     name='ICUbed mean',
# ))
# fig.add_trace(go.Scatter(
#     x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["ICUbed_mean"],
#     line_color='rgba(0,100,80,0.5)',
# #     name='ICUbed available',
# ))
# fig.update_traces(mode='lines')
# fig.show()


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ihme_df_04_09_nj["date"].tolist()+ihme_df_04_09_nj["date"][::-1].tolist(),
    y=ihme_df_04_09_nj['ICUbed_upper'].tolist()+(ihme_df_04_09_nj['ICUbed_lower'][::-1]).tolist(),
    fill='toself',
    fillcolor='rgba(0,100,80,0.2)',
    line_color='rgba(255,255,255,0)',
    showlegend=False,
    name='ICUbed',
))
fig.add_trace(go.Scatter(
    x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["ICUbed_mean"],
    line_color='rgb(0,100,80)',
    name='ICUbed mean',
))
fig.add_trace(go.Scatter(
    x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["ICUbed_available"],
    line_color='rgba(0,100,80,0.5)',
    name='ICUbed available',
))
fig.update_traces(mode='lines')
fig.show()


# ## Actual Reports
# 
# [https://www.nj.com/coronavirus/2020/04/nj-coronavirus-death-toll-hits-2350-as-cases-climb-to-61850-officials-announce-3733-new-positive-tests.html](http://)
# 
# [https://www.nj.gov/health/cd/topics/covid2019_dashboard.shtml](http://)
# 
# 
# #### There were 7,604 people hospitalized in New Jersey with confirmed or suspected coronavirus cases, with 1,914 of the patients in critical care or intensive care, as of 10 p.m. Saturday(2020-04-11), according to the state Department of Health.
# #### 5057 people is in general care
# 
# #### The state has 1,644 patients on ventilators, which is about 56% of the total capacity for the life-saving machines (So around 3000 ventilators are in the state). In addition, 658 patients have been discharged from hospitals in the last 24 hours.
# 
# #### The latest numbers include 64,885 negative tests for coronavirus.
# 
# #### Further availability of beds: critical care: 98, intensive care:306, general care:703; So the medical capacity has not been overwhelmed so far, i think.
# 
# Positive: 61,850, 
# Deaths: 2,350
# 
# Total Negatives: 64,885
# Major Lab Positivity: 44.5%
# Major Lab Positives*:52,106
# Major Labs Total Tests*: 116,991

# ## Observation:
# 
# ### All beds: prediction: 11248 Actual: 7604; 
# ### ICU beds: prediction: 2052 Actual: 1914; (Intensive care, Actual: 823,Critical care, Actual: 1091) 
# ### Ventilators: prediction: 1782 Actual: 1644; 
# 
# ### There is some discrepancy in availability of intensive care beds between IHME and NJ health department reports. IHME data says ICUbeds available as only 465 but the NJ health department shows still 306 beds available which means it has capacity of around 2210.
# 
# #### Note: Actual data corresponds to report on 04-11 and prediction is on 04-09.

# In[ ]:


def plot_eleven_columns_using_plotly_regular(dataframe,
                                             column_one,
                                             column_two,
                                             column_three,
                                             column_four,
                                             column_five,
                                             column_six,
                                             column_seven,
                                             column_eight,
                                             column_nine,
                                             column_ten,
                                             column_eleven,
                                             title):    
    '''
    This function plots four numerical columns against a date column.
    It using the regular plotly library instead of plotly express.
    '''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_one],
                        mode='lines+markers',name=column_one))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_two],
                        mode='lines+markers',name=column_two))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_three],
                        mode='lines+markers',name=column_three))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_four],
                        mode='lines+markers',name=column_four))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_five],
                        mode='lines+markers',name=column_five))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_six],
                        mode='lines+markers',name=column_six))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_seven],
                        mode='lines+markers',name=column_seven))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_eight],
                        mode='lines+markers',name=column_eight))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_nine],
                        mode='lines+markers',name=column_nine))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_ten],
                        mode='lines+markers',name=column_ten))
    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_eleven],
                        mode='lines+markers',name=column_eleven))
    fig.update_layout(title={'text':title},
                      xaxis_title='Date',yaxis_title='Average of upper and lower predictions',
                      legend_orientation="h",showlegend=True)
    #fig.update_layout(xaxis=dict(range=[lower_axis_limit,upper_axis_limit]))
    fig.show()    
    
def plot_eleven_columns_using_plotly_express(dataframe,
                                             column_one,
                                             column_two,
                                             column_three,
                                             column_four,
                                             column_five,
                                             column_six,
                                             column_seven,
                                             column_eight,
                                             column_nine,
                                             column_ten,
                                             column_eleven,
                                             title):
    '''
    This function plots four numerical columns against a date column.
    It using the plotly express library instead of the normal plotly library.
    '''
    df_melt = dataframe.melt(id_vars='date', value_vars=[column_one,
                                                         column_two,
                                                         column_three,
                                                         column_four,
                                                         column_five,
                                                         column_six,
                                                         column_seven,
                                                         column_eight,
                                                         column_nine,
                                                         column_ten,
                                                         column_eleven])
    fig = px.line(df_melt, x="date", y="value", color="variable",title=title).update(layout=dict(xaxis_title='date',yaxis_title='Average of upper and lower predictions',legend_orientation="h",showlegend=True))
    #fig.update_xaxes(range=[lower_axis_limit,upper_axis_limit])
    fig.show()


# In[ ]:


todays_date = '4/09/2020'


# In[ ]:


plot_eleven_columns_using_plotly_regular(dataframe=ihme_df_04_09_nj,
                                        column_one='allbed_mean',
                                        column_two='ICUbed_mean',
                                        column_three='InvVen_mean',
                                        column_four='deaths_mean',
                                        column_five='admis_mean',
                                        column_six='newICU_mean',
                                        column_seven='newICU_lower',
                                        column_eight='newICU_upper',
                                        column_nine='totdea_mean',
                                        column_ten='bedover_mean',
                                        column_eleven='bedover_mean',
                                        title='Mean of Upper and Lower Predictions from IHME for New Jersey as of '+todays_date)


# In[ ]:


plot_eleven_columns_using_plotly_express(dataframe=ihme_df_04_09_nj,
                                        column_one='allbed_mean',
                                        column_two='ICUbed_mean',
                                        column_three='InvVen_mean',
                                        column_four='deaths_mean',
                                        column_five='admis_mean',
                                        column_six='newICU_mean',
                                        column_seven='newICU_lower',
                                        column_eight='newICU_upper',
                                        column_nine='totdea_mean',
                                        column_ten='bedover_mean',
                                        column_eleven='bedover_mean',
                                        title='Mean of Upper and Lower Predictions from IHME for New Jersey as of '+todays_date)

