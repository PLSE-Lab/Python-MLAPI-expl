#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from bokeh.plotting import figure, show

from bokeh.io import output_notebook

from bokeh.models import ColumnDataSource, HoverTool

from bokeh.transform import dodge

output_notebook()


# In[ ]:


df = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200129.csv')
df.head()


# In[ ]:


# only check mainland and remove inaccurate suspected numbers
df = df[df['Country/Region'] == 'Mainland China']
df = df.drop(['Suspected', 'Country/Region'], axis=1)
df = df.fillna(0)


# In[ ]:


# convert the last update field type to datatime
def convert_last_update_to_day(x):
    return pd.to_datetime(x.split(' ')[0])

df['Last Update'] = df['Last Update'].apply(convert_last_update_to_day)
df = df.groupby(['Province/State', 'Last Update']).max()
df.head()


# In[ ]:


def get_df_by_name(provinces, df=df):
    ''' get dataframe by province list
    '''
    dfs = list()
    for p in provinces:
        df_tmp = df.loc[p]
        df_tmp = df_tmp.sort_values(by='Last Update')
        dfs.append(df_tmp)
    return dfs, provinces


# In[ ]:


def draw_trend(source, province):
    ''' draw trend line with bokeh source data
    '''
    plot = figure(x_axis_type="datetime", title="{} Trend".format(province), plot_width=800, plot_height=350)

    plot.line('Last Update', 'Confirmed', line_color='red', source=source, legend_label='Confirmed')
    plot.circle('Last Update', 'Confirmed', fill_color="white", size=8, source=source)   

    plot.line('Last Update', 'Recovered', line_color='green', source=source, legend_label='Recovered')
    plot.circle('Last Update', 'Recovered', fill_color="white", size=8, source=source)      

    plot.line('Last Update', 'Death', line_color='black', source=source, legend_label='Death')
    plot.circle('Last Update', 'Death', fill_color="white", size=8, source=source)      

    plot.legend.location = "top_left"
    plot.add_tools(HoverTool(tooltips=[("# Confirmed", "@Confirmed"), ("# Recovered", "@Recovered"), ("# Death", "@Death")]))

    show(plot)


# # Trends

# In[ ]:


# prepare dataframes

df_hubei = df.loc['Hubei']
df_hubei = df_hubei.sort_values(by='Last Update')
source_hubei = ColumnDataSource(df_hubei)

df_shanghai = df.loc['Shanghai']
df_shanghai = df_shanghai.sort_values(by='Last Update')
source_shanghai = ColumnDataSource(df_shanghai)

df_sum = df.groupby(by='Last Update').sum()
source_china = ColumnDataSource(df_sum)

source_others = ColumnDataSource(df_sum - df_hubei)


# ## Shanghai Trends

# In[ ]:


draw_trend(source_shanghai, 'Shanghai')


# ## China Trends

# In[ ]:


draw_trend(source_china, 'China')


# ## Hubei Trends

# In[ ]:


draw_trend(source_china, 'Hubei')


# ## Comparing The Trends

# In[ ]:


plot = figure(x_axis_type="datetime", title="Trends Comparison", plot_width=800, plot_height=350)

plot.line('Last Update', 'Confirmed', line_color='red', source=source_china, legend_label='# Confirmed China')
plot.circle('Last Update', 'Confirmed', fill_color="white", size=8, source=source_china)   

plot.line('Last Update', 'Confirmed', line_color='orange', source=source_hubei, legend_label='# Confirmed Hubei')
plot.circle('Last Update', 'Confirmed', fill_color="white", size=8, source=source_hubei)      

plot.line('Last Update', 'Confirmed', line_color='green', source=source_shanghai, legend_label='# Confirmed Shanghai')
plot.circle('Last Update', 'Confirmed', fill_color="white", size=8, source=source_shanghai)     

plot.line('Last Update', 'Confirmed', line_color='pink', source=source_others, legend_label='# Confirmed Non-Hubei')
plot.circle('Last Update', 'Confirmed', fill_color="white", size=8, source=source_others)   


plot.legend.location = "top_left"
plot.add_tools(HoverTool(tooltips=[("# Confirmed", "@Confirmed")]))

show(plot)


# # Distribution of Confirmed without Hubei 

# In[ ]:


df_sum_by_province = df.groupby(by='Province/State').max()
df_sum_by_province = df_sum_by_province[df_sum_by_province.index!='Hubei']
df_sum_by_province.head()
df_sum_by_province = df_sum_by_province.sort_values(by='Confirmed')


# In[ ]:


p = figure(y_range=list(df_sum_by_province.index), title="Distribution of Confirmed", plot_width=800, plot_height=550)
p.hbar(y=list(df_sum_by_province.index), right=list(df_sum_by_province['Confirmed']), height=0.6)

p.xgrid.grid_line_color = None
p.x_range.start = 0

p.add_tools(HoverTool(tooltips=[("Province", "@y"), ("Confirmed", "@right")]))

show(p)


# # Increasing Rate

# In[ ]:


a = df_sum['Confirmed'].iloc[:-1].reset_index(drop=True)
b = df_sum['Confirmed'].iloc[1:].reset_index(drop=True)
chg = b - a
df_increase = df_sum.iloc[1:,:]
pd.options.mode.chained_assignment = None
df_increase['chg'] = chg.values
df_increase['date'] = df_increase.index.map(lambda x: x.strftime('%Y-%m-%d'))
df_increase


# In[ ]:


p = figure(x_range=list(df_increase.index.format()), title="Increased Number of China Confirmed", plot_width=800, plot_height=350)
p.vbar(x=list(df_increase.index.format()), top=list(df_increase['chg']), width=0.2)
p.line(x=list(df_increase.index.format()), y=list(df_increase['chg']))
p.circle(x=list(df_increase.index.format()), y=list(df_increase['chg']), fill_color="white", size=8)    

p.xgrid.grid_line_color = None
p.y_range.start = 0

p.add_tools(HoverTool(tooltips=[("Date", "@x"), ("Increaed Number", "@top")]))

show(p)


# In[ ]:


df_others = df_sum - df_hubei
a = df_others['Confirmed'].iloc[:-1].reset_index(drop=True)
b = df_others['Confirmed'].iloc[1:].reset_index(drop=True)
chg = b - a
df_others_increase = df_others.iloc[1:,:]
pd.options.mode.chained_assignment = None
df_others_increase['chg'] = chg.values
df_others_increase['date'] = df_others_increase.index.map(lambda x: x.strftime('%Y-%m-%d'))
df_others_increase


# In[ ]:


p = figure(x_range=list(df_others_increase.index.format()), title="Increased Number of Non-Hubei Confirmed", plot_width=800, plot_height=350)
p.vbar(x=list(df_others_increase.index.format()), top=list(df_others_increase['chg']), width=0.2)
p.line(x=list(df_others_increase.index.format()), y=list(df_others_increase['chg']))
p.circle(x=list(df_others_increase.index.format()), y=list(df_others_increase['chg']), fill_color="white", size=8)    

p.xgrid.grid_line_color = None
p.y_range.start = 0

p.add_tools(HoverTool(tooltips=[("Date", "@x"), ("Increaed Number", "@top")]))

show(p)


# In[ ]:


source1 = ColumnDataSource(df_increase)
source2 = ColumnDataSource(df_others_increase)

p = figure(x_range=list(df_increase.index.format()), title="Increased Number Comparison", plot_width=800, plot_height=350)

p.vbar(x=dodge('date', 0, range=p.x_range), top='chg', width=0.2, color="#718dbf", legend_label="China", source=source1)
p.vbar(x=dodge('date', 0.25, range=p.x_range), top='chg', width=0.2, color="#e84d60", legend_label="China without Hubei", source=source2)

p.xgrid.grid_line_color = None
p.y_range.start = 0
p.legend.location = "top_left"

p.add_tools(HoverTool(tooltips=[("Date", "@date"), ("Increaed Number", "@chg")]))

show(p)


# In[ ]:




