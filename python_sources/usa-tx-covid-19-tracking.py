#!/usr/bin/env python
# coding: utf-8

# Ignore how it says "no data sources" above. That means there are no *static* data sources. Here are the data sources for this notebook, which are accessed at runtime to create the graphs below:
# * [JHU github repository](https://github.com/CSSEGISandData/COVID-19)
# * [Texas HHS](https://dshs.texas.gov/news/updates.shtm)
# * [Walker County, TX Office of Emergency Management](https://www.co.walker.tx.us/department/index.php?structureid=17)
# 
# This is a notebook for visualizing trends in COVID-19 data that I have not seen elsewhere. This notebook is meant to complement (not replace) other trackers, such as [worldometer](https://www.worldometers.info/coronavirus/). Other great trackers include [ProPublica's tracker](https://projects.propublica.org/reopening-america/), [covid19-projections](https://covid19-projections.com/about/), [CovidActNow](https://covidactnow.org/us/tx/?s=44750), and the [Texas Tribune tracker](https://apps.texastribune.org/features/2020/texas-coronavirus-cases-map/).
# 
# This notebook will be updated with new graphs on most days around 3pm CST.
# 
# This notebook is under active development, so if there is a visualization you would like to see, please let me know and I will consider adding it. 
# 
# County-level visualizations will remain focused on areas important to me. If you would like to see areas other than those depicted below and you know me, please ask. Otherwise, you may fork the notebook and modify it accordingly. The necessary modifications should be straightforward.

# In[ ]:


#TODO:
#-test incr vs cases incr, last 7 and 14 days - the computations are done, but think about the presentation
#-automate TX passing/failing/neither clearly passing nor clearly failing
import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} -c plotly plotly-orca ')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import urllib.request
import datetime
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.subplots import make_subplots

#data source: https://www.co.walker.tx.us/department/index.php?structureid=17
#unfortunately I have to update this one by hand. UGH.
df_Walker_TX_free = pd.DataFrame({'Province_State':['Texas'], 'Admin2':['Walker Free'], 'Country_Region':['US'],
                                "3/22/20":[1],
                                "3/23/20":[1],
                                "3/24/20":[1],
                                "3/25/20":[1],
                                "3/26/20":[2],
                                "3/27/20":[2],
                                "3/28/20":[3],
                                "3/29/20":[3],
                                "3/30/20":[3],
                                "3/31/20":[6],
                                "4/1/20":[7],
                                "4/2/20":[9],
                                "4/3/20":[10],
                                "4/4/20":[10],
                                "4/5/20":[10],
                                "4/6/20":[15],
                                "4/7/20":[15],
                                "4/8/20":[16],
                                "4/9/20":[18],
                                "4/10/20":[18],
                                "4/11/20":[18],
                                "4/12/20":[18],
                                "4/13/20":[21],
                                "4/14/20":[26],
                                "4/15/20":[41],
                                "4/16/20":[43],
                                "4/17/20":[44],
                                "4/18/20":[44],
                                "4/19/20":[44],
                                "4/20/20":[49],
                                "4/21/20":[51],
                                "4/22/20":[59],
                                "4/23/20":[61],
                                "4/24/20":[65],
                                "4/25/20":[65],
                                "4/26/20":[65],
                                "4/27/20":[69],
                                "4/28/20":[83],
                                "4/29/20":[83],
                                "4/30/20":[104],
                                "5/1/20":[108],
                                "5/2/20":[108],
                                "5/3/20":[108],
                                "5/4/20":[114],
                                "5/5/20":[128],
                                "5/6/20":[128],
                                "5/7/20":[128],
                                "5/8/20":[128],
                                "5/9/20":[128],
                                "5/10/20":[128],
                                "5/11/20":[131],
                                "5/12/20":[131],
                                "5/13/20":[137],
                                "5/14/20":[137],
                                "5/15/20":[137],
                                "5/16/20":[137],
                                "5/17/20":[137],
                                "5/18/20":[144],
                                "5/19/20":[150],
                                "5/20/20":[151],
                                "5/21/20":[151],
                                "5/22/20":[151],
                                "5/23/20":[151],
                                "5/24/20":[151],
                                "5/25/20":[159],
                                "5/26/20":[167],
                                "5/27/20":[172],
                                "5/28/20":[172],
                                "5/29/20":[172],
                                "5/30/20":[172],
                                "5/31/20":[184],
                                "6/1/20":[188],
                                "6/2/20":[193],
                                "6/3/20":[194],
                                "6/4/20":[201],
                                "6/5/20":[201],
                                "6/6/20":[201],
                                "6/7/20":[210],
                                "6/8/20":[210],
                                "6/9/20":[211],
                                "6/10/20":[217],
                                "6/11/20":[220],
                                "6/12/20":[223],
                                "6/13/20":[223],
                                "6/14/20":[223],
                                  "6/15/20":[242],
                                  "6/16/20":[248],
                                  "6/17/20":[265],
                                  "6/18/20":[274],
                                  "6/19/20":[283],
                                  "6/20/20":[283],
                                  "6/21/20":[283],
                                  "6/22/20":[306],
                                  "6/23/20":[320],
                                  "6/24/20":[373],
                                  "6/25/20":[393],
                                  "6/26/20":[415],
                                  "6/27/20":[415],
                                  "6/28/20":[415],
                                  "6/29/20":[452],
                                  "6/30/20":[465],
                                  "7/1/20":[499],
                                  "7/2/20":[541],
                                  "7/3/20":[541],
                                  "7/4/20":[541],
                                  "7/5/20":[541],
                                  "7/6/20":[628],
                                  "7/7/20":[656],
                                  "7/8/20":[665],
                                  "7/9/20":[697],
                                  "7/10/20":[715],
                                  "7/11/20":[715],
                                  "7/12/20":[715],
                                  "7/13/20":[773],
                                  "7/14/20":[773],
                                  "7/15/20":[831],
                                  "7/16/20":[848],
                                  "7/17/20":[848],
                                  "7/18/20":[848],
                                  "7/19/20":[848],
                                  "7/20/20":[934],
                                  "7/21/20":[970],
                                  "7/22/20":[1007],
                                  "7/23/20":[1021],
                                  "7/24/20":[1030],
                                  "7/25/20":[1030],
                                  "7/26/20":[1030],
                                  "7/27/20":[1053],
                                  "7/28/20":[1076],
                                  "7/29/20":[1089],
                                  "7/30/20":[1098],
                                  "7/31/20":[1107],
                                  "8/1/20":[1107],
                                  "8/2/20":[1107],
                                  "8/3/20":[1136],
                                  "8/4/20":[1145],
                                  "8/5/20":[1157],
                                  "8/6/20":[1172],
                                  "8/7/20":[1185],
                                  "8/8/20":[1185],
                                  "8/9/20":[1185],
                                  "8/10/20":[1239],
                                  "8/11/20":[1286],
                                  "8/12/20":[1316],
                                  "8/13/20":[1347],
                                  "8/14/20":[1401],
                                  "8/15/20":[1401],
                                  "8/16/20":[1401],
                                  "8/17/20":[1422],
                                  "8/18/20":[1426],
                                  "8/19/20":[1433],
                                  "8/20/20":[1447],
                                 })

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def clear_working_directory():
    for a,b,c in os.walk('/kaggle/working'):
        for f in c:
            if f[0]!='_':
                target = os.path.join(a,f)
                os.remove(target)
#plotting imports
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#display settings
pd.options.display.width = 1200
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)


# In[ ]:


#download the data
#clear_working_directory() 

today_datetime = datetime.datetime.today()
today = str(today_datetime).split(' ')[0]
target = '/kaggle/working/JHU_TS_US_confirmed_'+today+'.csv'
if not os.path.isfile(target): #if we haven't already downloaded the data today...
    urllib.request.urlretrieve('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv', target)
target = '/kaggle/working/JHU_TS_US_deaths_'+today+'.csv'
if not os.path.isfile(target):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv', target)
target = '/kaggle/working/TX_HHS_'+today+'.xlsx'    
if not os.path.isfile(target):
    urllib.request.urlretrieve('https://dshs.texas.gov/coronavirus/TexasCOVID19CaseCountData.xlsx', target)
target = '/kaggle/working/TX_HHS_cumulative_tests_county_'+today+'.xlsx'    
if not os.path.isfile(target):
    urllib.request.urlretrieve('https://dshs.texas.gov/coronavirus/TexasCOVID-19CumulativeTestsOverTimebyCounty.xlsx', target)
def datetime_to_JHU_date(d):
    '''d an object that has the form of datetime.datetime.today()'''
    s = str(d).split(' ')[0]
    s = s.split('-')
    s = '-'.join([s[1],s[2],s[0]])
    return s
    
target = '/kaggle/working/JHU_daily_us_most_recent.csv'
JHU_datetime = today_datetime
if not os.path.isfile(target):
    done, attempts_remaining = False, 10
    while (not done) and (attempts_remaining > 0):
        JHU_datestr = datetime_to_JHU_date(JHU_datetime)
        try:
            urllib.request.urlretrieve('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+JHU_datestr+'.csv', target)
            #fourteen_days_ago = date_JHU - datetime.timedelta(days=14) #Actually, I don't need this yet
            #urllib.request.urlretrieve('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+'....'+'.csv', target)
            done = True
        except:
            JHU_datetime = JHU_datetime - datetime.timedelta(days=1)
            attempts_remaining-=1
    if attempts_remaining == 0:
        print('Warning: JHU daily_reports_us has not been updated in over 8 days. Many graphing attempts below will fail.')

target = '/kaggle/working/JHU_daily_us_7_days_ago.csv'
if not os.path.isfile(target):
    JHU_datestr_minus_7 = datetime_to_JHU_date(JHU_datetime - datetime.timedelta(days=7))
    urllib.request.urlretrieve('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+JHU_datestr_minus_7+'.csv', target)
target = '/kaggle/working/JHU_daily_us_14_days_ago.csv'
if not os.path.isfile(target):
    JHU_datestr_minus_14 = datetime_to_JHU_date(JHU_datetime - datetime.timedelta(days=14))
    urllib.request.urlretrieve('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+JHU_datestr_minus_14+'.csv', target)
target = '/kaggle/working/JHU_daily_us_21_days_ago.csv'
if not os.path.isfile(target):
    JHU_datestr_minus_21 = datetime_to_JHU_date(JHU_datetime - datetime.timedelta(days=21))
    urllib.request.urlretrieve('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+JHU_datestr_minus_21+'.csv', target)
target = '/kaggle/working/JHU_daily_us_28_days_ago.csv'
if not os.path.isfile(target):
    JHU_datestr_minus_28 = datetime_to_JHU_date(JHU_datetime - datetime.timedelta(days=28))
    urllib.request.urlretrieve('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'+JHU_datestr_minus_28+'.csv', target)
        
#Read in the data
df_confirmed = pd.read_csv('/kaggle/working/JHU_TS_US_confirmed_'+today+'.csv')
df_deaths = pd.read_csv('/kaggle/working/JHU_TS_US_deaths_'+today+'.csv')
df_TX_tests = pd.read_excel('/kaggle/working/TX_HHS_cumulative_tests_county_'+today+'.xlsx', skiprows=[0]) #get first (only) sheet, skip first row
df_TX_pos_rate = pd.read_excel('/kaggle/working/TX_HHS_'+today+'.xlsx', sheet_name = 'Tests by day', skiprows = [0])
#df_TX_pos_rate = pd.read_excel('/kaggle/working/TX_HHS_'+today+'.xlsx', sheet_name = 'Tests by day', skiprows = [0,1])
#df_TX_hospitalizations = pd.read_excel('/kaggle/working/TX_HHS_'+today+'.xlsx', sheet_name = 'Hospitalization by Day', skiprows = [0,1])
df_TX_hospitalizations = pd.read_excel('/kaggle/working/TX_HHS_'+today+'.xlsx', sheet_name = 'Hospitalization by Day', skiprows = [0])
df_TX_hospitalizations.drop('Obs', axis=1, inplace=True)
df_JHU_daily_us = pd.read_csv('/kaggle/working/JHU_daily_us_most_recent.csv')
df_JHU_daily_us_m7 = pd.read_csv('/kaggle/working/JHU_daily_us_7_days_ago.csv')
df_JHU_daily_us_m14 = pd.read_csv('/kaggle/working/JHU_daily_us_14_days_ago.csv')
df_JHU_daily_us_m21 = pd.read_csv('/kaggle/working/JHU_daily_us_21_days_ago.csv')
df_JHU_daily_us_m28 = pd.read_csv('/kaggle/working/JHU_daily_us_28_days_ago.csv')
for k, df in enumerate([df_JHU_daily_us, df_JHU_daily_us_m7, df_JHU_daily_us_m14, df_JHU_daily_us_m21, df_JHU_daily_us_m28]):
    to_drop_list = df['Country_Region']!='US'
    to_drop = []
    for i in range(len(df)):
        if to_drop_list[i]:
            to_drop.append(i)
    if len(to_drop)>0:
        print('Warning: Extra data in JHU daily data index '+str(k)+'. (This is an error on their end that we have corrected for.)')
        df.drop(to_drop, axis=0, inplace=True)

first_date_col_index = 12 #This is the first col index for the dates in the time series for df_deaths and, after the next couple of lines have run, for df_confirmed. Update this if the dataset format changes.
#Make the columns for df_confirmed and df_deaths consistent.
if 'Population' not in df_confirmed.columns:
    df_confirmed.insert(first_date_col_index-1, 'Population', df_deaths['Population'])

#Do some data validation.
for col in zip(df_deaths.columns, df_confirmed.columns):
    if not col[0]==col[1]:
        print('Problem here', col)
if df_deaths.columns[first_date_col_index-1]!='Population':
    print('Problem: Population column is not correct in df_deaths.')
if df_confirmed.columns[first_date_col_index-1]!='Population':
    print('Problem: Population column is not correct in df_confirmed.')
if df_deaths.columns[first_date_col_index]!='1/22/20':
    print('First date is not 1/22/20 in df_deaths.')
if df_confirmed.columns[first_date_col_index]!='1/22/20':
    print('First date is not 1/22/20 in df_confirmed.')
if 'April 21' not in df_TX_tests.columns[1]:
    print('First date in df_TX_tests is not April 21')

#regularize the dates in df_TX_tests to be the same as those in df_deaths and df_confirmed
start_cols = df_TX_tests.columns.tolist()[1:]
cols = df_confirmed.columns.tolist()
cols = cols[cols.index('4/21/20'): len(cols)]
assert(len(cols) >= len(start_cols))
renamer = dict(zip(start_cols, cols))
df_TX_tests = df_TX_tests.rename(columns = renamer)

#fill na values
df_TX_tests.replace('-',np.nan, inplace=True)
df_TX_tests.replace('--',np.nan, inplace=True)
df_TX_tests = df_TX_tests.fillna(method = 'ffill', axis = 1)

df_TX_pos_rate.replace('.',np.nan, inplace=True)
df_TX_pos_rate.replace('-',np.nan, inplace=True)
df_TX_pos_rate.replace('--',np.nan, inplace=True)
#df_TX_pos_rate[['COVID-19\nPositivity\nRate' ,'New Viral Tests Reported* (Average of previous 7 days)']] = df_TX_pos_rate[['COVID-19\nPositivity\nRate' ,'New Viral Tests Reported* (Average of previous 7 days)']].bfill()

#data exploration
pt = 0
if pt:
    print(df_confirmed.head(90),'\n')
    print(df_deaths.head(5),'\n')
    for i, col in enumerate(df_confirmed.columns):
        print(i, col)
        if i>12:
            print('...')
            break

#define processing functions and constants
all_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
dates = df_confirmed.columns[first_date_col_index:] 
day_MA = 7
day_MA_2 = 14

def time_series(df, states='all', counties='all'):
    '''states, counties are "all" or a list'''
    if states == 'all':
        this_slice = df
    else:
        this_slice = df.loc[df.Province_State.isin(states), :] #first get all columns for these states
    
    if counties=='all':
        this_slice = this_slice
    else:
        this_slice = this_slice.loc[df.Admin2.isin(counties), :] #then get all the columns for these counties

    this_slice = this_slice.iloc[:, first_date_col_index:] #next, get all the day-by-day rows (1/22/2020 - present)
    TS = this_slice.sum(axis=0) #finally, sum these rows
    TS = pd.Series(TS, name='thing')
    
    return TS

def get_population(states = 'all', counties='all'):
    df = df_confirmed
    '''states, counties are "all" or a list'''
    if states == 'all':
        this_slice = df
    else:
        this_slice = df.loc[df.Province_State.isin(states), :] #first get all columns for these states
    
    if counties=='all':
        this_slice = this_slice
    else:
        this_slice = this_slice.loc[df.Admin2.isin(counties), :] #then get all the columns for these counties
        
    this_slice = this_slice['Population']
    return this_slice.sum()
        

def daily_change_in_time_series(TS):
    res=TS.rolling(window = 2).apply(lambda x: x.iloc[1] - x.iloc[0])
    res.iloc[0] = TS.iloc[0]
    return res

def trailing_moving_average(TS, n=7):
    TS = pd.Series(TS)
    return (TS.rolling(n).mean())

def single_area_graph(df, title, states, counties, difference=None, interactive=False):
    ignore_left_points=39 #initial number of datapoints not plotted (but still computed). Set to 39 to start the graphs at 3/1/2020
    
    TS = time_series(df, states = states, counties = counties)
    if difference is not None:
        TS_2 = time_series(df, states = difference['states'], counties = difference['counties'])
        TS = TS - TS_2
    TS_daily = daily_change_in_time_series(TS)
    MA = trailing_moving_average(TS_daily, day_MA)
    MA_2 = trailing_moving_average(TS_daily, day_MA_2)
    
    what = MA
    trace1 = go.Scatter(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='lines', name='{}-day MA'.format(day_MA), marker={'color':'red'})
    what = TS_daily
    trace2 = go.Scatter(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='markers', name='daily val', marker={'color':'blue', 'size':4, 'opacity':0.5})
    what = MA_2
    trace3 = go.Scatter(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='lines', name='{}-day MA'.format(day_MA_2), marker={'color':'rgba(0,160,0,0.7)'})
    data = [trace1, trace3, trace2]
    legend = {'orientation':'v', 'x':0, 'y':-0.35}
    layout = {'title':{'text':title, 'x':0}, 'xaxis':{'title':'date'}, 'yaxis':{'range':[0,1.2*MA.values[ignore_left_points:].max()]}, 'margin':{'l':0, 'r':0, 't':50}, 'legend':legend}
    fig = {'data':data, 'layout':layout}
    
    if interactive:
        iplot(fig, config={'displayModeBar':False})
    else:
        fig = go.Figure(data = data, layout = layout)
        config = (dict(displayModeBar = False))
        fig.show(config = config, renderer='png', width = 800)


# IMPORTANT NOTE: Beginning around August 1, most states started reporting far fewer daily tests than they did during the month of July. 
# 
# For example, during the last week of July, Texas reported an average of 62546 daily tests, but during the first week of August, reported an average of only 47254 daily tests.
# 
# Therefore, a decreasing number of daily new confirmed cases in the graphs below may correspond more to a decrease in the number of tests performed than a decrease in the true number of new infections.

# In[ ]:


#make graphs
single_area_graph(df_confirmed, 'USA daily new confirmed cases', states='all', counties='all')
single_area_graph(df_confirmed, 'USA minus NY and NJ daily new confirmed cases', states='all', counties='all', difference={'states':['New York', 'New Jersey'], 'counties':'all'})
single_area_graph(df_confirmed, 'TX daily new confirmed cases', states=['Texas'], counties='all')


# In[ ]:


ignore_left_points = 50
data=[]
today_data = []
colorwheel = 3
for state in all_states:
    TS = time_series(df_confirmed, states = [state], counties = 'all')
    TS_daily = daily_change_in_time_series(TS)
    MA = trailing_moving_average(TS_daily, day_MA)
    MA_2 = trailing_moving_average(TS_daily, day_MA_2)
    
    what = MA
    if state=='Texas':
        color = 'red'
        opacity = 1
    elif state=='New York':
        color = 'black'
        opacity = 0.5
    elif state=='Arizona':
        color = 'DarkOrange'
        opacity = 0.75
    elif state=='Florida':
        color = 'rgb(50,100,50)'
        opacity = 0.5
    elif state=='Nevada':
        color = 'DarkBlue'
        opacity = 0.5
    else:
        color = 'hsl({},50%,30%)'.format(colorwheel)
        #color = 'hsl(100,90%,30%)'
        colorwheel += 360/50
        opacity = 0.1

    what = what / get_population(states = [state], counties='all')*100000 #switch to new cases per 100,000
    trace = go.Scatter(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode = 'lines', opacity = opacity, marker = {'color':color}, name = state)
    data.append(trace)
    
    today_data.append(what.values[-1])
today_data = pd.DataFrame(index = all_states, data = {'Daily new cases per 100,000 residents, average of last 7 days (lower is better)': today_data})

#legend = {'orientation':'v', 'x':0, 'y':-0.35}
legend={}
layout={'title':{'text':'Daily new cases per 100,000 residents, 7-day moving average (Texas in red)', 'x':0}, 'legend':legend, 'margin':{'l':0, 'r':0, 't':50}, 'showlegend':True}    
    
fig = go.Figure({'data': data, 'layout':layout})
fig.show()


# In[ ]:


today_data = today_data.sort_values('Daily new cases per 100,000 residents, average of last 7 days (lower is better)', ascending=False)
today_data.head(50)


# In[ ]:


#graph new confirmed cases by state for all 50 states + DC
ignore_left_points=39 #initial number of datapoints not plotted (but still computed). Set to 39 to start the graphs at 3/1/2020

df = df_confirmed

fig = make_subplots(rows = 17, cols = 3, subplot_titles=all_states, shared_xaxes=True, vertical_spacing = 0.01)

for i_state, state in enumerate(all_states):
    states=[state]
    counties = 'all'
    TS = time_series(df, states = states, counties = counties)
    TS_daily = daily_change_in_time_series(TS)
    MA = trailing_moving_average(TS_daily, day_MA)
    MA_2 = trailing_moving_average(TS_daily, day_MA_2)

    row, col = divmod(i_state,3)
    row+=1
    col+=1
    
    what = MA
    fig.add_trace( go.Scattergl(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='lines', marker={'color':'red'}, name='{}-day MA'.format(day_MA)),  row = row, col = col)
    what = TS_daily
    fig.add_trace( go.Scattergl(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='markers', marker={'color':'blue', 'size':2, 'opacity':0.5}, name='daily val'),   row = row, col = col)
    what = MA_2
    fig.add_trace( go.Scattergl(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='lines', marker={'color':'rgba(0,160,0,0.7)'}, name='{}-day MA'.format(day_MA_2)),   row = row, col = col)
    fig.update_yaxes(range = [0,1.2*MA.values[ignore_left_points:].max()], row = row, col = col)
    
    #if i_state>6:
    #    break
    
legend = {'orientation':'v', 'x':0, 'y':-0.35}
layout = {'title':{'text':'Daily new confirmed cases by state', 'x':0}, 'height':3000, 'width':1000, 'showlegend':False} 
#'xaxis':{'title':'date'}, 'yaxis':{'range':[0,1.2*MA.values[ignore_left_points:].max()]}, 'margin':{'l':0, 'r':0, 't':50}, }
fig.update_layout(layout)
fig.update_layout(margin = {'l' : 0, 'r' : 0}) #, row=row, col=col)
config = (dict(displayModeBar = False))

for i in fig['layout']['annotations']:
    i['font']['size']=12 #change title size for the subgraphs

fig.show(config = config, renderer='png', width = 800, height = 2000)

#fig = {'data':data, 'layout':layout}

#iplot(fig, config={'displayModeBar':False})


# IMPORTANT NOTE: The following graph is created from TX HHS data.

# In[ ]:


title = 'TX active COVID-19 hospitalizations'
trace1 = go.Scatter(x = df_TX_hospitalizations.Date, y = df_TX_hospitalizations.Hospitalizations, mode='lines+markers', name='Hopsitalizations',  marker={'color':'red', 'opacity':0.5})
data = [trace1]
legend = {'orientation':'v', 'x':0, 'y':-0.35}
layout = {'title':{'text':title, 'x':0}, 'xaxis':{}, 'yaxis':{'range':[0,1.2*df_TX_hospitalizations.Hospitalizations.max()], 'title':'People hospitalized'}, 'margin':{'l':0, 'r':0, 't':50}, 'legend':legend}
fig = {'data':data, 'layout':layout}
fig = go.Figure(data = data, layout = layout)
config = (dict(displayModeBar = False))
fig.show(config = config, renderer='png', width = 800)


# In[ ]:


if 1: #Why can't these fuckers keep their spreadsheet structure the same from day to day?
    title = 'TX new molecular tests and percentage of new tests positive (7-day moving averages) (TX HHS data)'

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace( go.Scattergl(x = df_TX_pos_rate['Date'], y = 100*df_TX_pos_rate['COVID-19\nPositivity\nRate'], name = '% of new tests positive (7-day MA)', mode='lines+markers', marker={'opacity':0.5}), secondary_y = False,  )
    #fig.add_trace( go.Scattergl(x = df_TX_pos_rate['Date'], y = df_TX_pos_rate['New Viral Tests Reported* (Average of previous 7 days)'], name = '# of new tests (7-day MA)', mode='lines+markers', marker={'opacity':0.5}), secondary_y = True )
    fig.add_trace( go.Scattergl(x = df_TX_pos_rate['Date'], y = df_TX_pos_rate['New\nMolecular\nTests\nReported*'], name = '# of new tests (7-day MA)', mode='lines+markers', marker={'opacity':0.5}), secondary_y = True )


    legend = {'orientation':'v', 'x':0, 'y':-0.35}
    layout = {'title':{'text':title, 'x':0}, 'margin':{'l':0, 'r':0, 't':50}, 'legend':legend}
    fig.update_layout(layout)

    config = (dict(displayModeBar = False))
    fig.show(config = config, renderer='png', width = 800,)

#Based on the information above, 


# According to [the White House's guidelines for reopening America](https://www.npr.org/2020/04/16/836489480/read-white-house-guidelines-to-states-for-reopening), to begin reopening safely a state must first satisfy three criteria. States are to reopen in phases, and to proceed to each new phase the state must satisfy all three criteria again. The second criterion is this:
# * Downward trajectory of new confirmed cases within a 14-day period (with a flat or increasing volume of tests), OR
# * Downward trajectory of positive tests as a percentage of total tests within a 14-day period (with a flat or increasing volume of tests).
# 
# Texas is currently neither clearly passing nor clearly failing the second criterion for reopening safely.

# The following table displays an approximation of the doubling time for the number of active cases. Specifically, it shows the number of days it would take for the cumulative number of cases in each state to increase by the number of currently active cases in that state if the state's case growth rate (see graphs immediately above) does not change. This is not exactly the same as the doubling time for the number of active cases, because some currently-active cases will resolve in this time.
# 
# States with an accelerating number of cases will take less time than displayed, and cases with a decelerating number of cases will take more time. 
# 
# It is possible for a state to see an accelerating number of cases due to an increase in testing. This table includes information on whether daily testing for each state is increasing, decreasing, or flat.
# 
# States with fewer than 1000 active cases are excluded from this table.

# In[ ]:


#examine time for number of cases to increase by number of active cases (which isn't the same thing as active cases doubling)

#This code isn't elegant. Sorry.
df = df_JHU_daily_us[['Province_State', 'Active']]
df = df.loc[df['Province_State'].isin(all_states),:]
df_active = df.set_index('Province_State')

accel_points_1=[]
most_recent_MA_day_case_increase = []
for state in all_states:
    TS = time_series(df_confirmed, states = [state], counties = 'all')
    TS_daily = daily_change_in_time_series(TS)
    MA = trailing_moving_average(TS_daily, day_MA)
    most_recent_MA_day_case_increase.append(MA[-1])
    accel_points_1.append(MA[-14])
df_MA_newest_incr_rate = pd.DataFrame(index=all_states, data={'incr_rate':most_recent_MA_day_case_increase})
df_MA_incr_rate_14_days_ago = pd.DataFrame(index=all_states, data={'incr_rate':accel_points_1} )
df1 = df_active['Active']/df_MA_newest_incr_rate['incr_rate']

accel_points_2=[]
most_recent_MA_2_day_case_increase = []
for state in all_states:
    TS = time_series(df_confirmed, states = [state], counties = 'all')
    TS_daily = daily_change_in_time_series(TS)
    MA = trailing_moving_average(TS_daily, day_MA_2)
    most_recent_MA_2_day_case_increase.append(MA[-1])
    accel_points_2.append(MA[-14])
df_MA_2_newest_incr_rate = pd.DataFrame(index=all_states, data={'incr_rate':most_recent_MA_2_day_case_increase})
df_MA_2_incr_rate_14_days_ago = pd.DataFrame(index=all_states, data={'incr_rate':accel_points_2})
df2 = df_active['Active']/df_MA_2_newest_incr_rate['incr_rate']

r0,r1,rr0,rr1 = df_MA_incr_rate_14_days_ago['incr_rate'], df_MA_newest_incr_rate['incr_rate'], df_MA_2_incr_rate_14_days_ago['incr_rate'], df_MA_2_newest_incr_rate['incr_rate']
is_accel = (1.1*r0 < r1)&(1.1*rr0 < rr1) #vectorized op
is_decel = (1.1*r1 < r0)&(1.1*rr1 < rr0)
is_neither = (~is_accel) & (~is_decel)

def compute_tests_incr_or_decr(return_rate = True):
    df_daily_most_recent = df_JHU_daily_us[['Province_State','People_Tested']].copy()
    df_daily_m7 = df_JHU_daily_us_m7[['Province_State','People_Tested']].copy()
    df_daily_m14 = df_JHU_daily_us_m14[['Province_State','People_Tested']].copy()
    df_daily_m21 = df_JHU_daily_us_m21[['Province_State','People_Tested']].copy()
    df_daily_m28 = df_JHU_daily_us_m28[['Province_State','People_Tested']].copy()
    for df in [df_daily_most_recent, df_daily_m7, df_daily_m14, df_daily_m21, df_daily_m28]:
        df.set_index('Province_State', inplace=True)
    df_MR_14_avg_tests = (df_daily_most_recent - df_daily_m14)/14
    df_MR_7_avg_tests = (df_daily_most_recent - df_daily_m7)/7
    df_7_days_ago_7_avg_tests = (df_daily_m7 - df_daily_m14)/7    
    df_7_days_ago_14_avg_tests = (df_daily_m7 - df_daily_m21)/14    
    df_14_days_ago_14_avg_tests = (df_daily_m14 - df_daily_m28)/14
    #To be increasing, we should be doing at least 3% more tests now than a week ago, and 10% now than two weeks ago.
    tests_incr = (df_MR_7_avg_tests['People_Tested'] > 1.03*df_7_days_ago_7_avg_tests['People_Tested']) & (df_MR_14_avg_tests['People_Tested'] > 1.10*df_14_days_ago_14_avg_tests['People_Tested'])
    tests_decr = (df_MR_7_avg_tests['People_Tested'] < df_7_days_ago_7_avg_tests['People_Tested']) & (df_MR_14_avg_tests['People_Tested'] < df_14_days_ago_14_avg_tests['People_Tested'])
    tests_neither = (~tests_incr) & (~tests_decr)
    to_return = [tests_incr, tests_decr, tests_neither]
    if return_rate:
        to_return.append( ((df_MR_14_avg_tests/df_7_days_ago_14_avg_tests - 1)*100 + (df_MR_14_avg_tests/df_14_days_ago_14_avg_tests - 1)*100)/2 ) #average the testing increase rates measured two different ways
    return to_return

tests_incr, tests_decr, tests_neither, df_tests_incr_rate = compute_tests_incr_or_decr()

df_min = df1.combine(df2, min)
df_min = df_min.rename('min')
df_max = df1.combine(df2, max)
df_max = df_max.rename('max')
df = pd.DataFrame(df_min)
#df['Tests are up by...'] = df_tests_incr_rate
df['max']=df_max
df = df.loc[df_active['Active']>=1000,:].copy() #exclude states with <1000 active cases
df = df.sort_values('min')
df['Days until # cases increases by # active cases (larger is better)'] = df['min'].combine(df['max'], lambda x,y:'{:.2f} to {:.2f}'.format(x,y) )
df.drop(['max','min'], axis = 1, inplace=True)
df['Number of active cases'] = df_active['Active'].astype(int)
df['Cases accelerating?'] = is_accel.apply(lambda x: 'Accelerating' if x is True else None)
df.loc[is_decel,'Cases accelerating?'] = 'Decelerating'
df.loc[is_neither, 'Cases accelerating?'] = 'Neither / Unclear'
df['Tests increasing?'] = tests_incr.apply(lambda x: 'Increasing' if x is True else None)
df.loc[tests_decr, 'Tests increasing?'] = 'Decreasing'
df.loc[tests_neither, 'Tests increasing?'] = 'Flat'


def color_cells(val):
    if val in ['Accelerating', 'Decreasing']:
        color = 'red'
    elif val in ['Decelerating', 'Increasing']:
        color = 'green'
    else:
        color = None
    return 'color: %s' % color
df = df.reset_index() 
df = df.rename(columns = {'Province_State':'State', 'Cases accelerating?':'Cases are...', 'Tests increasing?':'Daily tests are...'})
#df = df.drop(columns = ['Tests are...'], axis = 1)
df = df.set_index('State')
df.head(51).style.applymap(color_cells)


# In[ ]:


#figure out how to describe this, then show the output
#As tests rise, so do the number of cases. If cases are rising faster than tests, that means the percentage of tests that are positive is increasing. When this happens, this is very bad. At minimum, as we expand testing we should hope that the number of cases grows no more rapildy than the number of tests, and ideally, we would like to see the number of tests growing more rapidly than the number of cases.

#Compute 7 and 14 day percentage increase in tests and cases
df_tests_cases = df_JHU_daily_us[['Province_State', 'People_Tested', 'Confirmed']]
df_tests_cases_m7 = df_JHU_daily_us_m7[['Province_State', 'People_Tested', 'Confirmed']]
df_tests_cases_m14 = df_JHU_daily_us_m14[['Province_State', 'People_Tested', 'Confirmed']]
for df_TC in [df_tests_cases, df_tests_cases_m7, df_tests_cases_m14]:
    df_TC.set_index('Province_State',inplace=True)
df_tests_cases_incr_14_days = ((df_tests_cases / df_tests_cases_m14 ) - 1)*100
df_tests_cases_incr_7_days =((df_tests_cases / df_tests_cases_m7) - 1)*100
df_tests_cases_incr_7_days['Total test growth (last 7 days)'] = df_tests_cases_incr_7_days['People_Tested'].apply(lambda x:'{:.1f}%'.format(x))
df_tests_cases_incr_7_days['Total case growth (last 7 days)'] = df_tests_cases_incr_7_days['Confirmed'].apply(lambda x:'{:.1f}%'.format(x))
df_tests_cases_incr_14_days['Total test growth (last 14 days)'] = df_tests_cases_incr_14_days['People_Tested'].apply(lambda x:'{:.1f}%'.format(x))
df_tests_cases_incr_14_days['Total case growth (last 14 days)'] = df_tests_cases_incr_14_days['Confirmed'].apply(lambda x:'{:.1f}%'.format(x))

df = df_tests_cases_incr_7_days[['Total case growth (last 7 days)', 'Total test growth (last 7 days)']].join(
    df_tests_cases_incr_14_days[['Total case growth (last 14 days)', 'Total test growth (last 14 days)']] )
df = df.reset_index()
df = df.loc[df['Province_State'].isin(all_states),:]
df = df.rename(columns = {'Province_State':'State'})
df = df.set_index('State')
df.head(51)


# In[ ]:


#add Walker Free info to df_confirmed. Only add columns that are already in df_confirmed. (We only want to graph dates where we have all the info for everywhere.)
for col in df_confirmed.columns.values:
    if col not in df_Walker_TX_free:
        df_Walker_TX_free[col] = np.nan
for col in df_Walker_TX_free.columns.values:
    if col not in df_confirmed:
        df_Walker_TX_free.drop(col, axis=1, inplace=True)
if sum(df_confirmed['Admin2']=='Walker Free') == 0: #if we haven't added Walker Free info yet
    df_confirmed = df_confirmed.append(df_Walker_TX_free, ignore_index = True)
    
#single_area_graph(df_confirmed, 'Walker County, TX daily new confirmed cases', states=['Texas'], counties=['Walker'])
single_area_graph(df_confirmed, 'Walker County, TX daily new confirmed cases (free population only)', states=['Texas'], counties=['Walker Free'])
single_area_graph(df_confirmed, 'Walker County, TX and surrounding areas daily new confirmed cases (free population only)', states=['Texas'], counties=['Walker Free', 'Harris', 'Montgomery', 'Grimes', 'Brazos', 'San Jacinto', 'Trinity', 'Houston', 'Madison'])
single_area_graph(df_confirmed, 'Harris County, TX daily new confirmed cases', states=['Texas'], counties=['Harris'])
single_area_graph(df_confirmed, 'Montgomery County, TX daily new confirmed cases', states=['Texas'], counties=['Montgomery'])


# In[ ]:


single_area_graph(df_confirmed, 'Nueces County, TX daily new confirmed cases', states=['Texas'], counties=['Nueces'])
single_area_graph(df_confirmed, 'San Patricio County, TX daily new confirmed cases', states=['Texas'], counties=['San Patricio'])
single_area_graph(df_confirmed, 'Douglas County, NV daily new confirmed cases', states=['Nevada'], counties=['Douglas'])
single_area_graph(df_confirmed, 'El Dorado County, CA daily new confirmed cases', states=['California'], counties=['El Dorado'])
single_area_graph(df_confirmed, 'Placer County, CA daily new confirmed cases', states=['California'], counties=['Placer'])
single_area_graph(df_confirmed, 'Sacramento County, CA daily new confirmed cases', states=['California'], counties=['Sacramento'])
single_area_graph(df_confirmed, 'Fulton County, GA daily new confirmed cases', states=['Georgia'], counties=['Fulton'])


# In[ ]:


single_area_graph(df_confirmed, 'Yavapai County, AZ daily new confirmed cases', states=['Arizona'], counties=['Yavapai'])
single_area_graph(df_confirmed, 'Maricopa County, AZ daily new confirmed cases', states=['Arizona'], counties=['Maricopa'])
single_area_graph(df_confirmed, 'Sedgwick County, KS daily new confirmed cases', states=['Kansas'], counties=['Sedgwick'])


# In[ ]:


single_area_graph(df_confirmed, 'Clark County, NV daily new confirmed cases', states=['Nevada'], counties=['Clark'])
single_area_graph(df_confirmed, 'Miami-Dade County, FL daily new confirmed cases', states=['Florida'], counties=['Miami-Dade'])


# In[ ]:


single_area_graph(df_deaths, 'USA daily deaths', states='all', counties='all')
single_area_graph(df_deaths, 'USA minus NY and NJ daily deaths', states='all', counties='all', difference={'states':['New York', 'New Jersey'], 'counties':'all'})
single_area_graph(df_deaths, 'TX daily deaths', states=['Texas'], counties='all')


# In[ ]:


ignore_left_points=39 #initial number of datapoints not plotted (but still computed). Set to 39 to start the graphs at 3/1/2020

df = df_deaths

fig = make_subplots(rows = 17, cols = 3, subplot_titles=all_states, shared_xaxes=True, vertical_spacing = 0.01)

for i_state, state in enumerate(all_states):
    states=[state]
    counties = 'all'
    TS = time_series(df, states = states, counties = counties)
    TS_daily = daily_change_in_time_series(TS)
    MA = trailing_moving_average(TS_daily, day_MA)
    MA_2 = trailing_moving_average(TS_daily, day_MA_2)

    row, col = divmod(i_state,3)
    row+=1
    col+=1
    
    what = MA
    fig.add_trace( go.Scattergl(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='lines', marker={'color':'red'}, name='{}-day MA'.format(day_MA)),  row = row, col = col)
    what = TS_daily
    fig.add_trace( go.Scattergl(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='markers', marker={'color':'blue', 'size':2, 'opacity':0.5}, name='daily val'),   row = row, col = col)
    what = MA_2
    fig.add_trace( go.Scattergl(x = what.index[ignore_left_points:], y = what.values[ignore_left_points:], mode='lines', marker={'color':'rgba(0,160,0,0.7)'}, name='{}-day MA'.format(day_MA_2)),   row = row, col = col)
    fig.update_yaxes(range = [0,1.2*MA.values[ignore_left_points:].max()], row = row, col = col)
    
    #if i_state>6:
    #    break
    
legend = {'orientation':'v', 'x':0, 'y':-0.35}
layout = {'title':{'text':'Daily deaths by state', 'x':0}, 'height':3000, 'width':1000, 'showlegend':False} 
#'xaxis':{'title':'date'}, 'yaxis':{'range':[0,1.2*MA.values[ignore_left_points:].max()]}, 'margin':{'l':0, 'r':0, 't':50}, }
fig.update_layout(layout)
fig.update_layout(margin = {'l' : 0, 'r' : 0}) #, row=row, col=col)
config = (dict(displayModeBar = False))

for i in fig['layout']['annotations']:
    i['font']['size']=12 #change title size for the subgraphs

fig.show(config = config, renderer='png', width = 800, height = 2000)


# In[ ]:


#development area
df_tests_cases = df_JHU_daily_us[['Province_State', 'People_Tested', 'Confirmed']]
df_tests_cases_m7 = df_JHU_daily_us_m7[['Province_State', 'People_Tested', 'Confirmed']]
df_tests_cases_m14 = df_JHU_daily_us_m14[['Province_State', 'People_Tested', 'Confirmed']]
for df_TC in [df_tests_cases, df_tests_cases_m7, df_tests_cases_m14]:
    df_TC.set_index('Province_State',inplace=True)
df_tests_cases_incr_14_days = ((df_tests_cases / df_tests_cases_m14 ) - 1)*100
df_tests_cases_incr_7_days =((df_tests_cases / df_tests_cases_m7) - 1)*100
df_tests_cases_incr_7_days['Tests increase (7 days)'] = df_tests_cases_incr_7_days['People_Tested'].apply(lambda x:'{:.1f}%'.format(x))
df_tests_cases_incr_7_days['Cases increase (7 days)'] = df_tests_cases_incr_7_days['Confirmed'].apply(lambda x:'{:.1f}%'.format(x))
df_tests_cases_incr_14_days['Tests increase (14 days)'] = df_tests_cases_incr_14_days['People_Tested'].apply(lambda x:'{:.1f}%'.format(x))
df_tests_cases_incr_14_days['Cases increase (14 days)'] = df_tests_cases_incr_14_days['Confirmed'].apply(lambda x:'{:.1f}%'.format(x))


# In[ ]:


#df_JHU_daily_us[['Province_State','Confirmed']] - df_JHU_daily_us_m14[['Province_State','Confirmed']]
time_series(df_confirmed, states=['Texas'])
get_population(states=['Texas'])


# In[ ]:


df_TX_pos_rate.columns
#df_TX_hospitalizations.columns


# In[ ]:


time_series(df = df_deaths, states=['Texas'], counties='all').tail(9)


# In[ ]:


get_population(states=['Alaska'])


# In[ ]:


#with pd.option_context('display.max_rows', None, 'display.max_columns', 25):
#    print(df_confirmed.head())

