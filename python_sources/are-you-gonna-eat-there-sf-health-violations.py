#!/usr/bin/env python
# coding: utf-8

# ## A Dashboard for San Francisco Restaurant Health Inspections

# In[ ]:


# Imports
import datetime as dt
import folium
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import sys

# Settings
init_notebook_mode()
pd.set_option('max_rows', 8)

# Files
raw_data_filename = '../input/restaurant-scores-lives-standard.csv'

# Load inspection data
inspection_data = pd.read_csv(raw_data_filename,
    usecols=['business_id', 'inspection_id', 'inspection_date', 
             'inspection_type', 'violation_id', 'violation_description',
             'risk_category'],
    parse_dates=['inspection_date'], infer_datetime_format=True)
                              
# Validate inspection data
risk_categories = ['No Violations', 'Low Risk', 'Moderate Risk', 'High Risk']
invalid_risk_categories = (~inspection_data['risk_category']
                           .isin(risk_categories + [np.nan]))
if invalid_risk_categories.any():
    print('Found entries with unknown risk categories')
    print('Acceptable risk categories are {0}'.format(risk_categories))
    print('Dropped offending entries')
    display(inspection_data.loc[invalid_risk_categories])
    inspection_data.drop(invalid_risk_categories.index, inplace=True)

# Load business data in a separate, less redundent, dataframe
business_info = (pd.read_csv(raw_data_filename, 
                             usecols=['business_id', 'business_name', 
                                      'business_latitude', 'business_longitude'])
                   .drop_duplicates('business_id')
                   .set_index('business_id'))

# Clean up business name format
business_info['business_name'] = (business_info['business_name']
                                  .str.title().str.replace("'S", "'s"))

# Set any latitude and longitude that is obviously outside San Fran to Na
# Ideally, at some point, code in an address based geocatcher to find Na
business_info.loc[(business_info['business_latitude'] < 37.65) | 
                  (business_info['business_latitude'] > 37.85) |
                  (business_info['business_longitude'] < -122.55) |
                  (business_info['business_longitude'] > -122.30),
                   ['business_latitude', 'business_longitude']] =\
                     [np.nan, np.nan]

# Focusing on "Routine - Unscheduled" inspections for now
# These dominate the inspection types and are less likely to have special cases
routine_data = (inspection_data
 .loc[inspection_data['inspection_type'] == 'Routine - Unscheduled'].copy())

# Assign risk_category to categorical type for sorting
routine_data['risk_category'] = (routine_data['risk_category'].astype('category'))
routine_data['risk_category'].cat.set_categories(risk_categories, 
                                                 ordered=True, inplace=True)

# Drop duplicate rows
routine_data.drop_duplicates(inplace=True)


# In[ ]:


# Plot violations per inspection per day

def count_inspections(df, rule, label=None, loffset=None):
    '''Count unique inspections per time period. 
       Takes inspections dataframe and time perios rule.
       Returns series of counts per period. '''

    df['n'] = 1
    inspection_counts = (df
     .groupby(['inspection_date', 'inspection_id'])
     .n.count()
     .reset_index(level=1)
     .resample(rule, label=label, loffset=loffset).count()['n'])
    inspection_counts.drop(inspection_counts.loc[inspection_counts == 0].index,
                           inplace=True)
    
    return inspection_counts


def count_violations(df, rule, label=None, loffset=None):
    '''Count number of occurances of each risk level by time period.
       Takes inspections dataframe and time period rule. 
       Returns dataframe with inspection_date as index and 
       risk_categories as columns. '''
    
    df['n'] = 1
    df.fillna({'risk_category':'No Violations'}, inplace=True)
    date_risk_count = (pd.pivot_table(df, 'n', 'inspection_date', 'risk_category',
                                      'count', fill_value=0)
                       .resample(rule, loffset=loffset, label=label).sum())
    
    return date_risk_count


def violations_per_inspection(df, rule, label=None, loffset=None):
    vio_insp = (count_violations(df, rule, label, loffset)
                .divide(count_inspections(df, rule, label, loffset), axis=0)
                .dropna())

    return vio_insp


# Plot data
plot_data = violations_per_inspection(routine_data, 'D')
data = []
risk_levels = ['No Violations', 'Low Risk', 'Moderate Risk', 'High Risk'][::-1]
colors = ['black', 'blue', 'orange', 'red'][::-1]
visibilities = ['legendonly', True, True, True][::-1]
x = plot_data.index

for risk_level, color, visible in zip(risk_levels, colors, visibilities):
    y = plot_data[risk_level]
    data.append(go.Scatter(x=x, y=y, name=risk_level, mode='lines',
                           line=dict(color=color), visible=visible))

# Figure layout
layout = dict(title='Violations per inspection (daily average)',
              width=700, height=400,
              xaxis=dict(title='Inspection date', ticklen=8, 
                         hoverformat='%d', 
                         range=[routine_data['inspection_date'].max() -
                                dt.timedelta(30), 
                                routine_data['inspection_date'].max()]),
              yaxis=dict(hoverformat='.2f'))

fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


# Plot violations per inspection per month

#  Use data from the most recent two years of completed months
end_date = routine_data['inspection_date'].max().replace(day=1)
begin_date = end_date - dt.timedelta(2 * 365)
twoyear_data = (routine_data.loc[(routine_data['inspection_date'] >= begin_date) &
                                 (routine_data['inspection_date'] < end_date)]
                .copy())
        
plot_data = violations_per_inspection(twoyear_data, 'M', label='left', 
                                      loffset=dt.timedelta(1))

# Plot violations per inspection
data = []
risk_levels = ['No Violations', 'Low Risk', 'Moderate Risk', 'High Risk'][::-1]
colors = ['black', 'blue', 'orange', 'red'][::-1]
x = plot_data.index
for risk_level, color in zip(risk_levels, colors):
    y = plot_data[risk_level]
    data.append(go.Scatter(x=x, y=y, name=risk_level, mode='lines',
                           line=dict(color=color)))

# Specify figure layout
layout = dict(title='Violations per inspection (monthly average)',
              width=700, height=400,
              xaxis=dict(title='Inspection date', ticklen=8, 
                         hoverformat='%b %Y', 
                         range=[begin_date + dt.timedelta(355),
                                end_date - dt.timedelta(20)]),
              yaxis=dict(hoverformat='.2f'))
fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


# Plot geographic map of most recent violations
# Get violations within the past n days of the most recently reported violation
n_days = 30

def inspection_df(insp):
    ''' Generate dataframe with one row per inspection for plotting on 
        geographic map '''    
    # Retain business id to join with business info
    business_id = insp['business_id'].iloc[0]
    # Color by highest risk violation in inspection
    color_map = {'Low Risk':'blue', 'Moderate Risk':'orange', 
                 'High Risk':'red'}
    color = color_map[insp['risk_category'].iloc[0]]
    # List all violations in inspection
    violations = [': '.join(text) for text in zip(insp['risk_category'], 
                                                  insp['violation_description'])]
    violations_str = '<br>'.join(violations)
    
    return pd.DataFrame({'business_id':business_id, 'color':color, 
                         'violations':violations_str}, index=[0])

# Gather info to plot for selected date range
recent_violations = (routine_data
 .loc[(routine_data['inspection_date'] > 
       routine_data['inspection_date'].max() - dt.timedelta(n_days)) &
      (routine_data['violation_id'].notnull())]
 .sort_values('risk_category', ascending=False)
 .groupby('inspection_id')
 .apply(inspection_df)
 .drop_duplicates()
 .join(business_info[['business_name', 'business_latitude', 
                      'business_longitude']], on='business_id')
)

# For now, just drop the missing latitudes and longitudes, focusing on making map
# first ideally come back and try to geolocate from address
recent_violations.dropna(inplace=True)

# Add data to map
restaurant_map = folium.Map(location=(37.76, -122.44),
                            width='100%', height='90%',
                            zoom_start=12, min_zoom=12, max_zoom=15)
(recent_violations
 #.iloc[:5] # for troubleshooting
 .apply(lambda row: folium
        .Marker([row['business_latitude'], row['business_longitude']], 
            popup=('<p><b>{0}</b><br>{1}</p>'.format(row['business_name'],
                                                     row['violations'])),
            icon=folium.Icon(icon_color='black', color=row['color'], 
                             icon='bug', prefix='fa'))
        .add_to(restaurant_map), axis='columns')
)

# Add legend and title via html
legend_html =  '''
    <div style="position: fixed; bottom: 20px; left: 20px; 
                width: 108px; height: 120px;
                background-color:silver; border:2px solid black; z-index:9999;
                font-size:14px;
      "<h2 align="center"><b>Health Risk</b></h2><br>
      <p align="left">
        &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i> 
        &nbsp; High <br>
        &nbsp; <i class="fa fa-map-marker fa-2x" style="color:orange"></i> 
        &nbsp; Moderate <br>
        &nbsp; <i class="fa fa-map-marker fa-2x" style="color:#0099FF"></i> 
        &nbsp; Low 
      </p>
    </div>
    ''' 
title_html = '''
    <div style="position: static; margin: auto; z-index:9998; font-size:20px;
      "<h2 align="center"><b>Health Violations (from most recent {0} days of 
                             inspections)</b></h2><br>
    </div>
    '''.format(n_days)
restaurant_map.get_root().html.add_child(folium.Element(title_html))
restaurant_map.get_root().html.add_child(folium.Element(legend_html))

display(restaurant_map)


# In[ ]:


# Plot geographic map for recent inspections with no violations

# Gather info to plot for selected date range
recent_no_violations = (routine_data
 .loc[(routine_data['inspection_date'] > 
       routine_data['inspection_date'].max() - dt.timedelta(n_days)) &
      (routine_data['violation_id'].isnull())]
 .loc[:, ['business_id']]
 .join(business_info[['business_name', 'business_latitude', 
                      'business_longitude']], on='business_id')
)

# For now, just drop the missing latitudes and longitudes, focusing on making map
# first ideally come back and try to geolocate from address
recent_no_violations.dropna(inplace=True)

# Add data to map
restaurant_map = folium.Map(location=(37.76, -122.44), 
                            width='100%', height='90%',
                            zoom_start=12, min_zoom=12, max_zoom=15)

(recent_no_violations
  #.iloc[:5] # for troubleshooting
  .apply(lambda row: folium
         .Marker([row['business_latitude'], row['business_longitude']],
              popup=('<p><b>{0}</b></p>'.format(row['business_name'])),
             icon=folium.Icon(icon_color='black', color='green', 
                              icon='thumbs-up', prefix='fa'))
         .add_to(restaurant_map), axis='columns')
)

# Add title html
title_html = '''
    <div style="position: static; margin: auto; z-index:9998; font-size:20px;
      "<h2 align="center"><b>Inspected, No Violations (from most recent {0} 
                             days) </b></h2><br>
    </div>
    '''.format(n_days)
restaurant_map.get_root().html.add_child(folium.Element(title_html))
display(restaurant_map)


# In[ ]:


# Test violations per inspection per day
run_tests = False

if run_tests:
    test_data = pd.DataFrame([[ 0,  0, np.NaN,           '2001-01-01'], # Day 1
                              [ 1,  1, 'Low Risk',       '2001-01-01'],
                              [ 2,  2, 'Moderate Risk',  '2001-01-01'],
                              [ 3,  3, 'High Risk',      '2001-01-01'],
                              [ 4,  4, 'Low Risk',       '2001-01-02'], # Day 2
                              [ 4,  5, 'Moderate Risk',  '2001-01-02'],
                              [ 5,  6, 'High Risk',      '2001-01-02'],                                                     
                              [ 6,  7, 'Moderate Risk',  '2001-01-03'], # Day 3
                              [ 6,  8, 'High Risk',      '2001-01-03'],
                                                                        # Day 4 - none
                              [ 7,  9, 'Moderate Risk',  '2001-01-05'], # Day 5
                              [ 7, 10, 'Moderate Risk',  '2001-01-05'],
                              [ 7, 11, 'Moderate Risk',  '2001-01-05'],
                              [ 8, 12, np.NaN,           '2001-01-05'],
                                                                        # Day 6 - none
                              [ 9, 13, 'Low Risk',       '2001-01-07'], # Day 7
                              [ 9, 14, 'Low Risk',       '2001-01-07'],
                              [10, 15, 'Low Risk',       '2001-01-07'],
                              [10, 16, 'Moderate Risk',  '2001-01-07'],
                              [10, 17, 'High Risk',      '2001-01-07']],
                             columns=['inspection_id', 'violation_id', 
                                      'risk_category', 'inspection_date'])

    test_data['inspection_date'] = pd.to_datetime(test_data['inspection_date'])
    
    output_correct = pd.DataFrame({'High Risk':     [0.25, 0.50, 1.00, 0.0, 0.5],
                                   'Low Risk':      [0.25, 0.50, 0.00, 0.0, 1.5],
                                   'Moderate Risk': [0.25, 0.50, 1.00, 1.5, 0.5],
                                   'No Violations': [0.25, 0.00, 0.00, 0.5, 0.0]},
                                  index=['2001-01-01', '2001-01-02', '2001-01-03',
                                         '2001-01-05', '2001-01-07'])

    output_correct.index = pd.to_datetime(output_correct.index)
    output_correct.columns.name = 'risk_category'
    output_correct.index.name = 'inspection_date'
        
    pd.testing.assert_frame_equal(output_correct, 
                                  violations_per_inspection(test_data, 'D'))


# In[ ]:


# Test violations per inspection per day

if run_tests:
    test_data = pd.DataFrame([[ 0,  0, np.NaN,           '2001-01-01'], # Month 1
                              [ 1,  1, 'Low Risk',       '2001-01-08'],
                              [ 2,  2, 'Moderate Risk',  '2001-01-12'],
                              [ 3,  3, 'High Risk',      '2001-01-31'],
                              [ 4,  4, np.NaN,           '2001-02-01'], # Month 2
                              [ 5,  5, np.NaN,           '2001-02-02'],
                              [ 6,  6, 'Moderate Risk',  '2001-02-03'],
                              [ 7,  7, 'Moderate Risk',  '2001-02-04'],
                              [ 8,  8, 'Low Risk',       '2001-03-15'], # Month 3
                              [ 8,  9, 'Low Risk',       '2001-03-15'],
                              [ 8, 10, 'Moderate Risk',  '2001-03-15'],
                              [ 8, 11, 'Moderate Risk',  '2001-03-15'],
                                                                        # Month 4 - none
                              [ 9, 12, 'Low Risk',       '2001-05-01'], # Month 5
                              [ 9, 13, 'Low Risk',       '2001-05-01'],
                              [10, 10, 'High Risk',      '2001-05-31'],
                              [10, 11, 'Moderate Risk',  '2001-05-31']],
                             columns=['inspection_id', 'violation_id', 
                                      'risk_category', 'inspection_date'])  
    
    test_data['inspection_date'] = pd.to_datetime(test_data['inspection_date'])

    output_correct = pd.DataFrame({'High Risk':     [0.25, 0.00, 0.00, 0.50],
                                   'Low Risk':      [0.25, 0.00, 2.00, 1.00],
                                   'Moderate Risk': [0.25, 0.50, 2.00, 0.50],
                                   'No Violations': [0.25, 0.50, 0.00, 0.00]},
                                  index=['2001-01-01', '2001-02-01', '2001-03-01',
                                         '2001-05-01'])

    output_correct.index = pd.to_datetime(output_correct.index)
    output_correct.columns.name = 'risk_category'
    output_correct.index.name = 'inspection_date'
    
    pd.testing.assert_frame_equal(output_correct, 
                                  violations_per_inspection(test_data, 'M', 'left',
                                                            dt.timedelta(1)))


# The dataset, provided by the City of San Francisico Health Department, logs violations identified during restaurant inspections and their corresponding health risk. The current version of this notebook looks at the results of  "Routine - Unschedule inspecitons" which dominate the dataset.
# 
# The dashboard was built following the [Dashboarding with Notebooks tutorial](http://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5) by Rachael Tatman
