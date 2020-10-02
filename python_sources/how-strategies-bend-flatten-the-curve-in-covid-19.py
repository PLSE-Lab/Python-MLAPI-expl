#!/usr/bin/env python
# coding: utf-8

# # How is the implementation of existing strategies affecting the rates of COVID-19 infections? 
# 
# This notebook represents a visualization of COVID-19 new cases/deaths align with the dates strategies being implemented. 

# ## Data sources
# 
# - UNCOVER COVID-19 Challenge; up to date version from [Namara platform](https://how-to.namara.io/#introduction)
# 
#     - ACAPS COVID-19: Govenment Measures Dataset    
#     
#     The COVID-19 Govenment Measures Dataset puts together all the measures implemented by govenments worldwide in response to the Coronavirus pandemic.
#     
#     - [Coronavirus Disease (COVID-19) -- Statistics and Research](https://ourworldindata.org/coronavirus)   
#     
#     Confirmed total and daily cases/deaths on country level.

# In[ ]:


# Load essential moduals
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import datetime
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

# Input data files are available in the "../input/" directory.
# List all csv files used in this notebook 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if(filename == "coronavirus-disease-covid-19-statistics-and-research.csv" or
           filename == "acaps-covid-19-government-measures-dataset.csv"):
            print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.

# Set update_ds = True to load dataset from Namara platform,
# otherwise the notebook will read saved csv files as printed in the following.
update_ds = False


# ### Load data from [Namara platform](https://app.namara.io)
# 
# Download up to date data from [Namara platform](https://app.namara.io) and save csv file.

# In[ ]:


def load_namara(url):
    import requests
    import time
    response = requests.get(url)
    print(f"Request returned {response.status_code} : '{response.reason}'")

    i = 0
    while(response.json()['message']!='Exported'):
        time.sleep(1)
        print(response.json()['message'], i)
        i = i + 1
        response = requests.get(url)
        print(f"Request returned {response.status_code} : '{response.reason}'")

    csv_url = response.json()['url']
    csv_file = csv_url.split('?')[0].split('/')[-1]
    print("Read csv file: {}".format(csv_file))
    df = pd.read_csv(csv_url)
    print("with {} lines, {} columns".format(df.shape[0], df.shape[1]))

    df.to_csv(csv_file, index=False)
    print("Save a copy: {}".format(csv_file))

    d = datetime.datetime.today()
    print(d.strftime('%Y-%m-%d'))

    return df


# In[ ]:


my_api_key = "<api_key>"   # set Namara platform api_key


# #### ACAPS COVID-19: Govenment Measures Dataset -- The Humanitarian Data Exchange
# 

# In[ ]:


url_measure = ('https://api.namara.io/v0/data_sets/'
               'f2a2f3a6-83d2-4f58-a606-85f1598884e6/'
               'data/en-3/export?geometry_format=wkt'
               '&api_key=' + my_api_key +
               '&organization_id=5e96c73c6eec7900102a37a9'
               )
if(update_ds): df_measure = load_namara(url_measure)


# #### Coronavirus Disease (COVID-19) -Statistics and Research -- [Our World in Data](https://ourworldindata.org/coronavirus)

# In[ ]:


url_owd = ('https://api.namara.io/v0/data_sets/'
                   'e820187b-708c-4394-a251-8fe61b919624/'
                   'data/en-0/export?geometry_format=wkt'
                   '&api_key=' + my_api_key +
                   '&organization_id=5e96c73c6eec7900102a37a9'
                   )
if(update_ds): df_owd = load_namara(url_owd)


# ### Load data from input directory

# In[ ]:


if(not update_ds):
    df_measure = pd.read_csv("/kaggle/input/uncover/UNCOVER/HDE_update/acaps-covid-19-government-measures-dataset.csv")
    df_owd = pd.read_csv("/kaggle/input/namara-covid-19/coronavirus-disease-covid-19-statistics-and-research.csv")
    print('Data loaded')


# ### An overview of dataset

# **Measures that have been implemented**

# In[ ]:


df_measure.replace(
    to_replace = 'testing policy',
    value      = 'Testing policy',
    inplace    = True
)
df_measure.replace(
    to_replace = 'strengthening the public health system',
    value      = 'Strengthening the public health system',
    inplace    = True
)
df_measure.replace(
    to_replace = 'limit public gatherings',
    value      = 'Limit public gatherings',
    inplace    = True
)


# In[ ]:


list_meas_in_cat = {}
for c in df_measure['category'].unique():
    measures = df_measure[df_measure['category'] == c]['measure'].unique()
    list_meas_in_cat[c] = measures

print("{:<70}{:>10}".format("Measures", "Number of Country"))
for c in list_meas_in_cat:
    print(c)
    for m in list_meas_in_cat[c]:
        n_c = len(df_measure.loc[df_measure['measure'] == m, 'country'].unique())
        print('{:<10}{:<60}{:>10}'.format('', m, n_c))


# ## Strategies are performed at which stage of COVID-19 infection for each country? 

# #### Visualization scripts

# In[ ]:


# define color map
# asign each measure a color and a hatch
color_measure={}
ic = 0
list_hatch = ['/', '\\', '|', 'x', 'o', 'O', '.', '*', '-', '+', '//', '\\\\', '||', '--']
for c in df_measure['category'].unique():
    list_m = df_measure.loc[df_measure['category']==c]['measure'].unique()
    n_m = len(list_m)
    color_c = mcolors.to_rgb('C'+str(ic))
    color_w = mcolors.to_rgb('w')
    im = 0
    for m in list_m:
       color_m = tuple(map(lambda x, y: x + 0.8*im*(y-x)/n_m, color_c, color_w ))
       hatch_m = list_hatch[im]
       color_measure[m] = {'color': color_m, 'hatch': hatch_m, 'category': c}
       im = im + 1
    color_measure[c] = {'color': color_c, 'category': c}
    ic=ic+1


# In[ ]:


# define iso -- country name map
iso_country = {}
for iso in df_measure['iso'].unique():
    country_name = df_measure.loc[df_measure['iso']==iso, 'country'].unique()[0]
    iso_country[iso] = country_name
for iso in df_owd['iso_code'].unique():
    country_name = df_owd.loc[df_owd['iso_code']==iso, 'location'].unique()[0]
    iso_country[iso] = country_name   


# In[ ]:


def plot_curv_meas(ax, iso, col_c, list_meas, col_c2 = None):
    # plot the curves for one country
    df_owd_1 = df_owd.loc[df_owd['iso_code'] == iso,['date', col_c]].copy()
    if(col_c2):
        df_owd_1 = df_owd.loc[df_owd['iso_code'] == iso, ['date', col_c, col_c2]].copy()
    df_owd_1.sort_values(by = 'date',  ascending = True, inplace = True)
    ax.plot('date', col_c, 'b-', data=df_owd_1)
    if(col_c2):
        ax_y2=ax.twinx()
        ax_y2.plot('date', col_c2, 'r-', data=df_owd_1)
        ax_y2.set_ylim(0, ax_y2.get_ylim()[1])
        ax_y2.tick_params(axis = 'y', labelcolor = 'r')
        ax.tick_params(axis = 'y', labelcolor = 'b')

    # mark the dates strategies implemented
    y_min, y_max = ax.get_ylim()
    y_min = 0
    ax.set_ylim(y_min, y_max)
    n_meas = len(list_meas)
    y_sep = (y_max - y_min)/n_meas
    x_min, x_max = ax.get_xlim()
    im = 0
    for m in list_meas:
        xx = df_measure.loc[(df_measure['iso']==iso) & (df_measure['measure']==m), ['date_implemented', 'entry_date']]
        if(xx.shape[0]<1): continue
        xx.sort_values(by='date_implemented', inplace = True)    
        for x0 in xx['date_implemented'].unique():
            if(pd.isna(x0)): continue
            x0 = mdates.date2num(x0)
            y0 = y_sep*im
            x_sep = x_max - x0 # no data for implementation end date
            rect = plt.Rectangle((x0,y0), x_sep, y_sep, alpha = 0.2, color=color_measure[m]['color'], hatch = color_measure[m]['hatch'])
            ax.add_patch(rect)
            im = im + 1
    ax.set_title(iso_country[iso] + "  " + col_c)
    if(col_c2):
        ax.set_title(iso_country[iso] + "  " + col_c + "/" + col_c2)
    
    # format the ticks
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter('%Y-%m')
    months_fmt = mdates.DateFormatter('%m')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(months_fmt)
    ax.grid(True, which = 'both', axis = 'y')
    plt.show()
    
    return


# In[ ]:


def color_legend(ax, list_meas):
    ax.set_xlim(0,1)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x0 = x_min
    x_sep = 0.1*(x_max-x_min)
    y_sep = (y_max-y_min)/len(list_meas)
    im = 0
    for m in list_meas[::-1]:
        y0 = y_max - im*y_sep
        rect = plt.Rectangle((x0,y0), x_sep, 0.8*y_sep, alpha = 0.2, 
                             color = color_measure[m]['color'], hatch = color_measure[m]['hatch'])
        ax.add_patch(rect)
        ax.text(x0+x_sep, y0+0.4*y_sep, m, va = 'center')
        im = im + 1
    ax.axis('off')
    ax.plot()
    return


# In[ ]:


df_measure['date_implemented'] = pd.to_datetime(df_measure['date_implemented'])
df_measure['entry_date'] = pd.to_datetime(df_measure['entry_date'])
df_owd['date'] = pd.to_datetime(df_owd['date'])


# ### Visualization & analysis
# 
# The measures are in six categories as listed below. 
# The visualization will focus on the following categories:
# 
# - Public health measures
# - Social distancing
# - Movement restrictions
# - Lockdown
# 
# The category
# 
# - <font color="gray">Governance and socio-economic measures</font>
# 
# is out of the scope here.
# 
# - <font color="gray">Humanitarian exemption</font>
# 
# only has two entries and no date implemented.
# 
# Some measures appear repeatedly in one plot for one country. 
# a) That could be the same measure with different details. For example, 'Limit public gathering' with different limit numbers.
# b) Or some of the measures are from different administration level (from province or state). The curves are on country level. 
# 
# 
# 

# In[ ]:


list_cat = list(df_measure['category'].unique())
print(list_cat)
list_cat.remove("Governance and socio-economic measures")
list_cat.remove("Humanitarian exemption")
print(list_cat)


# In[ ]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib import rcParams as mrcParams
mrcParams.update({'font.size': 22})

list_meas = []
for cat in list_cat:
    list_meas.extend(df_measure.loc[df_measure['category']==cat]['measure'].unique())

fig, axc = plt.subplots(1,1,figsize=(16,16))
color_legend(axc, list_meas)

for iso in ['CHN', 'USA', 'ITA', 'ESP', 'SWE', 'DNK', 'NOR']:
    fig, ax1 = plt.subplots(1,1,figsize=(16,16))
    plot_curv_meas(ax1, iso, 'new_cases', list_meas, col_c2 = 'new_deaths')


# ## How each measure bend or flatten the curves?
# 
# The daily confirmed cases and deaths are plotted along time and are originated at the day a specific measure was implemented. When a measure is implemented several times with different details, only the earlist one is considered here. 

# In[ ]:


# asign each region a color
region_color = {}
i = 0
for r in df_measure['region'].unique():
    region_color[r] = 'C'+str(i)
    i = i + 1

# define a iso region map
iso_region = {}
for iso in df_measure['iso'].unique():
    iso_region[iso] = df_measure.loc[df_measure['iso']==iso, ['region']].iat[0,0]


# In[ ]:



def plot_curv(ax, col_c, meas):

    list_iso = df_measure[df_measure['measure'] == meas]['iso'].unique()
    
    # Add a column that how many days it has past since a measure was implemented
    handles=[]
    labels=[]
    for iso in list_iso:
        xx = df_measure.loc[(df_measure['measure']==meas) & (df_measure['iso']==iso), ['date_implemented','entry_date']]
        xx.sort_values(by=['date_implemented', 'entry_date'], ascending=True, inplace = True)
        day0 = xx.iat[0,0]
        if(pd.isna(day0)):
            continue
        df_owd_measure_iso = df_owd.loc[df_owd['iso_code']==iso,['date', col_c]].copy()
        if(df_owd_measure_iso.shape[0]<1): continue
        df_owd_measure_iso['days'] = (df_owd_measure_iso['date'] - day0)/np.timedelta64(1, 'D')
        df_owd_measure_iso.sort_values(by=['days'], inplace = True)
        handles_i, = ax.plot('days', col_c, data = df_owd_measure_iso,  
                  color = region_color[iso_region[iso]])
        if(iso_region[iso] not in labels):
            handles.append(handles_i)
            labels.append(iso_region[iso])
        xt = df_owd_measure_iso['days'].iat[-1]
        yt = df_owd_measure_iso[col_c].iat[-1]
        ax.annotate(iso_country[iso], (xt, yt), color = region_color[iso_region[iso]])
    ax.set_ylim(0,None)
    ax.set_title(col_c + '\n' + meas)
    ax.legend(handles, labels, loc = 2)
    plt.show()
    return
    


# In[ ]:


from matplotlib import rcParams as mrcParams
mrcParams.update({'figure.max_open_warning': 0})
for m in list_meas:
    if(m=='Obligatory medical tests not related to COVID-19'): continue
    fig, ax1 = plt.subplots(1,1,figsize=(16,16))
    plot_curv(ax1, 'new_deaths', m)


# ## Summary
# 
# This notebook visualized that at which stage measures are implemented and the daily confirmed cases/deaths trend. Further analysis will be done to try to get insight how these stratigies affecting COVID-19 infection.   
# 
# This notebook will be updated and keeps tracking COVID-19.
# 
# Thanks for your reading!
# 
# 
# 
