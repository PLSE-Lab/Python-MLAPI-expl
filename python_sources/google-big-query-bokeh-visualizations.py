#!/usr/bin/env python
# coding: utf-8

# # BigQuery ML Intersection Data Analysis
# ![Intersection](https://c.wallhere.com/photos/d6/0e/city_architecture_cityscape_New_York_City_USA_building_car_street-81238.jpg!d)

# In[ ]:


from google.cloud import bigquery
from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')

from bokeh.plotting import figure, show, output_notebook
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import ColumnDataSource, HoverTool, CDSView, GroupFilter
from bokeh.models.annotations import Title
from bokeh.models.widgets import Tabs, Panel

#For inline plotting of bokeh plots
output_notebook()


# In[ ]:


PROJECT_ID = 'bigquery-geotab-competition'
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('bqml_example', exists_ok=True)

magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

# create a reference to our table
table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.train")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()


# # Exploratory Data Analysis

# ## Dataset Columns
# The training dataset has 28 columns with the following names

# In[ ]:


table.schema


# The BigQuery Python client library provides a magic command that allows you to run queries with minimal code.

# In[ ]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# The BigQuery client library provides a cell magic, %%bigquery, which runs a SQL query and returns the results as a Pandas DataFrame

# ## Mapping the intersections

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'int_counts', 'SELECT\n    City,\n    count(IntersectionId) as Intersections\nFROM (\n    SELECT DISTINCT\n        City,\n        IntersectionId\n    FROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\n    )\nGROUP BY City\nORDER BY Intersections')


# In[ ]:


sns.barplot(x='City', y='Intersections', data=int_counts)
plt.show()


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'intersections_map', 'SELECT\n    City,\n    IntersectionId,\n    Latitude,\n    Longitude,\n    avg(TotalTimeStopped_p80) as AverageTimeSpent\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City, IntersectionId, Latitude, Longitude\nORDER BY City ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'intersection_details', 'SELECT\n    City,\n    IntersectionId,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nORDER BY CITY ASC')


# In[ ]:


#Converting the co-ordinates to something that bokeh likes
def merc(Coords):
    lat = Coords[0]
    lon = Coords[1]
    
    r_major = 6378137.000
    x = r_major * math.radians(lon)
    scale = x/lon
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + 
        lat * (math.pi/180.0)/2.0)) * scale
    return (x, y)


# In[ ]:


intersections_map['Location'] = tuple(zip(intersections_map['Latitude'], intersections_map['Longitude']))
intersections_map['coords_x'] = intersections_map['Location'].apply(lambda x: merc(x)[0])
intersections_map['coords_y'] = intersections_map['Location'].apply(lambda x: merc(x)[1])
intersections_map.head()


# In[ ]:


merged_df = pd.merge(intersections_map,intersection_details, how='inner', on=['City', 'IntersectionId'])
merged_df.head()


# In[ ]:


merged_df.drop_duplicates(subset=['City','IntersectionId'], keep='first', inplace=True)


# In[ ]:


def build_city_int_map(city, colour):
    title_provider = get_provider(Vendors.CARTODBPOSITRON)

    data_map = merged_df[merged_df['City']== city]

    source = ColumnDataSource(data=dict(
                            x=list(data_map['coords_x']), 
                            y=list(data_map['coords_y']),
                            AvgWaitTime=list(data_map['AverageTimeSpent']),
                            sizes=list(data_map['AverageTimeSpent']/5),                                               
                            intersection=list(data_map['Path'])))

    hover = HoverTool(tooltips=[
        ("Wait Time","@AvgWaitTime"),
        ("Intersection","@intersection")
    ])

    p = figure(x_axis_type="mercator", y_axis_type="mercator",
           tools=[hover, 'wheel_zoom','save','pan','box_zoom','reset'])

    p.add_tile(title_provider)

    p.circle(x = 'x',
             y = 'y',
             source=source,
             size = 'sizes',
             line_color=colour, 
             fill_color=colour,
             fill_alpha=0.15)
    t = Title()
    t.text = city + ' Intersections with Average wait times'
    p.title = t
    return(p)


# In[ ]:


#Retrieve the plots for all the cities
at_map = build_city_int_map('Atlanta','green')
bo_map = build_city_int_map('Boston', 'orange')
ch_map = build_city_int_map('Chicago', 'red')
ph_map = build_city_int_map('Philadelphia', 'blue')

#Set the plot width
at_map.plot_width = bo_map.plot_width = ch_map.plot_width = ph_map.plot_width = 800

#Create 4 panels, one for each city
at_panel = Panel(child=at_map, title='Atlanta')
bo_panel = Panel(child=bo_map, title='Boston')
ch_panel = Panel(child=ch_map, title='Chicago')
ph_panel = Panel(child=ph_map, title='Philadelphia')

# Assign the panels to Tabs
tabs = Tabs(tabs=[at_panel, bo_panel, ch_panel, ph_panel])

show(tabs)


# In[ ]:


#Top most conjusted intersections for each city
cities = ['Atlanta', 'Boston', 'Chicago', 'Philadelphia']
lat_long_df = pd.DataFrame(columns=['Latitude','Longitude'])
for city in cities:
    lat_long_df =  lat_long_df.append(intersections_map[intersections_map['City']==city]                            .nlargest(1, 'AverageTimeSpent', keep='first')                            .loc[:,['Latitude','Longitude']], ignore_index=True, sort=False)
Latitudes = tuple(lat_long_df['Latitude'])
Longitudes = tuple(lat_long_df['Longitude'])


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'top_ints', 'SELECT * FROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\n    WHERE Latitude IN (33.77384, 42.27153, 41.91044, 39.9773) \n    AND Longitude IN (-84.34896, -71.17265, -87.67753, -75.22695)')


# In[ ]:


#Get Monthly Distribution of measures for the most conjusted intersections
top_monthly = top_ints.groupby(['City','Month','Path','EntryStreetName','ExitStreetName','EntryHeading','ExitHeading'])                .mean()                .loc[:,'TotalTimeStopped_p20':'DistanceToFirstStop_p80']                .sort_values(['City','Month'])                .reset_index()


# In[ ]:


#Get the Hourly Distribution of measures for the most conjusted intersections
top_hourly = top_ints.groupby(['City','Hour','Weekend','Path','EntryStreetName','ExitStreetName','EntryHeading','ExitHeading'])                .mean()                .loc[:,'TotalTimeStopped_p20':'DistanceToFirstStop_p80']                .sort_values(['City','Hour','Weekend'])                .reset_index()


# In[ ]:


def create_feature_dist(source_df, feature, percentile, monthly):
    #CDS Source
    source = ColumnDataSource(source_df)
    #Group Filters
    at_filters = [GroupFilter(column_name='City', group='Atlanta')]
    bo_filters = [GroupFilter(column_name='City', group='Boston')]
    ch_filters = [GroupFilter(column_name='City', group='Chicago')]
    ph_filters = [GroupFilter(column_name='City', group='Philadelphia')]
    #CDS Views
    at_view = CDSView(source=source, filters=at_filters)
    bo_view = CDSView(source=source, filters=bo_filters)
    ch_view = CDSView(source=source, filters=ch_filters)
    ph_view = CDSView(source=source, filters=ph_filters)

    hover = HoverTool(tooltips=[
            ("Entry Heading","@EntryHeading"),
            ("Exit Heading","@ExitHeading"),
            ("Entry Street","@EntryStreetName"),
            ("Exit Street", "@ExitStreetName"),
            ("Value", "@"+feature+'_p'+str(percentile))
    ])

    select_tools=[hover,'box_select', 'lasso_select', 'poly_select', 'tap', 'reset']
    
    if monthly:
        x_axis_label = 'Months'
        x = 'Month'
        plot_title = 'Monthly Distribution of '+feature+'_p'+str(percentile)
    else:
        x_axis_label = 'Hours'
        x = 'Hour'
        plot_title = 'Hourly Distribution of '+feature+'_p'+str(percentile)

    common_fig_kwargs = {
        'x_axis_label': x_axis_label,
        'y_axis_label': feature+'_p'+str(percentile),
        #'toolbar_location': 'below',
        'tools': select_tools
    }

    common_marker_kwargs = {
        'x': x,
        'y': feature+'_p'+str(percentile),
        'source': source,
        'size': 10,
        'nonselection_color': 'lightgray',
        'nonselection_alpha': 0.3,
        'fill_alpha': 0.15
    }

    common_at_kwargs = {
        'view': at_view,
        'color': 'blue',
        'legend': 'Atlanta'
    }

    common_bo_kwargs = {
        'view': bo_view,
        'color': 'orange',
        'legend': 'Boston'
    }

    common_ch_kwargs = {
        'view': ch_view,
        'color': 'red',
        'legend': 'Chicago'
    }

    common_ph_kwargs = {
        'view': ph_view,
        'color': 'green',
        'legend': 'Philadelphia'
    }

    fig = figure(**common_fig_kwargs, title=plot_title)
    fig.square(**common_marker_kwargs, **common_at_kwargs, muted_alpha=0.1)
    fig.circle(**common_marker_kwargs, **common_bo_kwargs, muted_alpha=0.1)
    fig.triangle(**common_marker_kwargs, **common_ch_kwargs, muted_alpha=0.1)
    fig.diamond(**common_marker_kwargs, **common_ph_kwargs, muted_alpha=0.1)
    fig.legend.click_policy = 'mute'
    fig.legend.location = 'top_left'
    return fig


# In[ ]:


def create_feature_grid(feature, percs, monthly=True):
    tbs=[]
    if monthly:
        source_df=top_monthly
    else:
        source_df=top_hourly
    for perc in percs:
        tbs.append(Panel(child=create_feature_dist(source_df, feature, perc, monthly), title='p'+str(perc)))
    return Tabs(tabs=tbs)


# In[ ]:


show(create_feature_grid('TotalTimeStopped',(20,40,50,60,80)))


# In[ ]:


show(create_feature_grid('TimeFromFirstStop',(20,40,50,60,80)))


# In[ ]:


show(create_feature_grid('DistanceToFirstStop',(20,40,50,60,80)))


# In[ ]:


show(create_feature_grid('TotalTimeStopped',(20,40,50,60,80),monthly=False))


# In[ ]:


show(create_feature_grid('TimeFromFirstStop',(20,40,50,60,80),monthly=False))


# In[ ]:


show(create_feature_grid('DistanceToFirstStop',(20,40,50,60,80),monthly=False))


# # Training Model Creation

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `bqml_example.model_tts20`\nOPTIONS(MODEL_TYPE='LINEAR_REG',\n    LS_INIT_LEARN_RATE=.3,\n    L1_REG=1,\n    MAX_ITERATIONS=10) AS\nSELECT\n    TotalTimeStopped_p20 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour, \n    Weekend,\n    Month,\n    City\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE RowId < 2600000")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `bqml_example.model_tts50`\nOPTIONS(MODEL_TYPE='LINEAR_REG',\n    LS_INIT_LEARN_RATE=.3,\n    L1_REG=1,\n    MAX_ITERATIONS=10) AS\nSELECT\n    TotalTimeStopped_p50 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour,\n    Weekend,\n    Month,\n    City\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE RowId < 2600000")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `bqml_example.model_tts80`\nOPTIONS(MODEL_TYPE='LINEAR_REG',\n    LS_INIT_LEARN_RATE=.3,\n    L1_REG=1,\n    MAX_ITERATIONS=10) AS\nSELECT\n    TotalTimeStopped_p80 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour,\n    Weekend,\n    Month,\n    City\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE RowId < 2600000")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `bqml_example.model_dtfs20`\nOPTIONS(MODEL_TYPE='LINEAR_REG',\n    LS_INIT_LEARN_RATE=.3,\n    L1_REG=1,\n    MAX_ITERATIONS=10) AS\nSELECT\n    DistanceToFirstStop_p20 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour,\n    Weekend,\n    Month,\n    City\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE RowId < 2600000")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `bqml_example.model_dtfs50`\nOPTIONS(MODEL_TYPE='LINEAR_REG',\n    LS_INIT_LEARN_RATE=.3,\n    L1_REG=1,\n    MAX_ITERATIONS=10) AS\nSELECT\n    DistanceToFirstStop_p50 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour,\n    Weekend,\n    Month,\n    City\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE RowId < 2600000")


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `bqml_example.model_dtfs80`\nOPTIONS(MODEL_TYPE='LINEAR_REG',\n    LS_INIT_LEARN_RATE=.3,\n    L1_REG=1,\n    MAX_ITERATIONS=10) AS\nSELECT\n    DistanceToFirstStop_p80 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour,\n    Weekend,\n    Month,\n    City\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE RowId < 2600000")


# # Training Statistics

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.TRAINING_INFO(MODEL `bigquery-geotab-competition.bqml_example.model_tts20`)')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.TRAINING_INFO(MODEL `bigquery-geotab-competition.bqml_example.model_tts50`)')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.TRAINING_INFO(MODEL `bigquery-geotab-competition.bqml_example.model_tts80`)')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.TRAINING_INFO(MODEL `bigquery-geotab-competition.bqml_example.model_dtfs20`)')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.TRAINING_INFO(MODEL `bigquery-geotab-competition.bqml_example.model_dtfs50`)')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.TRAINING_INFO(MODEL `bigquery-geotab-competition.bqml_example.model_dtfs80`)')


# # Model Evaluation

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.EVALUATE(MODEL `bqml_example.model_tts20`, (\n  SELECT\n    TotalTimeStopped_p20 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour, \n    Weekend,\n    Month,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE RowId > 2600000))')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.EVALUATE(MODEL `bqml_example.model_tts50`, (\n  SELECT\n    TotalTimeStopped_p50 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour, \n    Weekend,\n    Month,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE RowId > 2600000))')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.EVALUATE(MODEL `bqml_example.model_tts80`, (\n  SELECT\n    TotalTimeStopped_p80 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour, \n    Weekend,\n    Month,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE RowId > 2600000))')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.EVALUATE(MODEL `bqml_example.model_dtfs20`, (\n  SELECT\n    DistanceToFirstStop_p20 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour, \n    Weekend,\n    Month,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE RowId > 2600000))')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.EVALUATE(MODEL `bqml_example.model_dtfs50`, (\n  SELECT\n    DistanceToFirstStop_p50 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour, \n    Weekend,\n    Month,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE RowId > 2600000))')


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT * FROM ML.EVALUATE(MODEL `bqml_example.model_dtfs80`, (\n  SELECT\n    DistanceToFirstStop_p80 AS label,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Hour, \n    Weekend,\n    Month,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE RowId > 2600000))')


# # Predict Outcomes

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'df1', 'SELECT\n  RowId,\n  predicted_label as TotalTimeStopped_p20\nFROM\n  ML.PREDICT(MODEL `bqml_example.model_tts20`,\n    (\n    SELECT\n        RowId,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Hour, \n        Weekend,\n        Month,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'df2', 'SELECT\n  RowId,\n  predicted_label as TotalTimeStopped_p50\nFROM\n  ML.PREDICT(MODEL `bqml_example.model_tts50`,\n    (\n    SELECT\n        RowId,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Hour, \n        Weekend,\n        Month,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'df3', 'SELECT\n  RowId,\n  predicted_label as TotalTimeStopped_p80\nFROM\n  ML.PREDICT(MODEL `bqml_example.model_tts80`,\n    (\n    SELECT\n        RowId,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Hour, \n        Weekend,\n        Month,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'df4', 'SELECT\n  RowId,\n  predicted_label as DistanceToFirstStop_p20\nFROM\n  ML.PREDICT(MODEL `bqml_example.model_dtfs20`,\n    (\n    SELECT\n        RowId,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Hour, \n        Weekend,\n        Month,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'df5', 'SELECT\n  RowId,\n  predicted_label as DistanceToFirstStop_p50\nFROM\n  ML.PREDICT(MODEL `bqml_example.model_dtfs50`,\n    (\n    SELECT\n        RowId,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Hour, \n        Weekend,\n        Month,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'df6', 'SELECT\n  RowId,\n  predicted_label as DistanceToFirstStop_p80\nFROM\n  ML.PREDICT(MODEL `bqml_example.model_dtfs80`,\n    (\n    SELECT\n        RowId,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Hour, \n        Weekend,\n        Month,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# # Output as csv

# In[ ]:


#Change RowId in accordance with submission file
df1['RowId'] = df1['RowId'].apply(str) + '_0'
df2['RowId'] = df2['RowId'].apply(str) + '_1'
df3['RowId'] = df3['RowId'].apply(str) + '_2'
df4['RowId'] = df4['RowId'].apply(str) + '_3'
df5['RowId'] = df5['RowId'].apply(str) + '_4'
df6['RowId'] = df6['RowId'].apply(str) + '_5'

#Renaming the columns
df1.rename(columns={'RowId': 'TargetId', 'TotalTimeStopped_p20': 'Target'}, inplace=True)
df2.rename(columns={'RowId': 'TargetId', 'TotalTimeStopped_p50': 'Target'}, inplace=True)
df3.rename(columns={'RowId': 'TargetId', 'TotalTimeStopped_p80': 'Target'}, inplace=True)
df4.rename(columns={'RowId': 'TargetId', 'DistanceToFirstStop_p20': 'Target'}, inplace=True)
df5.rename(columns={'RowId': 'TargetId', 'DistanceToFirstStop_p50': 'Target'}, inplace=True)
df6.rename(columns={'RowId': 'TargetId', 'DistanceToFirstStop_p80': 'Target'}, inplace=True)

#Concatenate all the dataframes into one
df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)


# In[ ]:


df.to_csv(r'submission.csv', index=False)

