#!/usr/bin/env python
# coding: utf-8

# ### the file we are using in this notebook is route.csv 

# In[ ]:


# install the package we need
get_ipython().system('pip install vincent')
get_ipython().system('pip install pmdarima')
get_ipython().system('pip install chart_studio')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import pandas as pd


# In[ ]:


# plotly standard imports
import plotly.graph_objs as go
import chart_studio.plotly as py

# Cufflinks wrapper on plotly
import cufflinks


# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from plotly.offline import iplot
cufflinks.go_offline()

# Set global theme
cufflinks.set_config_file(world_readable=True, theme='ggplot')


# In[ ]:


filename = '../input/coronavirusdataset/'
route = pd.read_csv(filename+'/route.csv')


# In[ ]:


route.head()


# In[ ]:


route.info()


# In[ ]:


route['date'] = pd.to_datetime(route['date'])
route = route.set_index('date')
route.head()


# In[ ]:


#check if data have missing value
missing = route.isnull().sum()
missing


# In[ ]:


province_id = route.groupby('province')['id'].aggregate([np.sum])
city_id = route.groupby('city')['id'].aggregate([np.sum])
visit_id = route.groupby('visit')['id'].aggregate([np.sum])


# In[ ]:


province_id.iplot(kind='bar', title='total id in each province', xTitle='Province',                  yTitle='total id')


# In[ ]:


city_id.iplot(kind='bar', title='total id in each city', xTitle='city',                  yTitle='total id')


# In[ ]:


visit_id.iplot(kind='bar', title='total id in each visit', xTitle='city',                  yTitle='total id')


# In[ ]:



def color(id):
    minimun = int(route['id'].min())
    step = int((route['id'].max() - route['id'].min())/3)
    
    if id in range(minimun, minimun+step):
        col = 'blue'
    elif id in range(minimun+step, minimun+step*2):
        col = 'orange'
    else:
        col = 'red'
    
    return col


# In[ ]:


import folium as fl
import json
import vincent

def geospace():

    mapped = fl.Map(location=[route['latitude'].mean(),route['longitude'].mean()],zoom_start=6,             control_scale=True, world_copy_jump=True, no_wrap=False)

    fg_province= fl.FeatureGroup(name="Province")
    for lat, lon, name, id in zip(route['latitude'], route['longitude'], route['province'],route['id']):
        fl.Marker(location=[lat, lon], popup=(fl.Popup(name+ ' id = '+ str(id))),              icon = fl.Icon(color=color(id))).add_to(fg_province)
    
    fg_city = fl.FeatureGroup(name='City')
    for lat, lon, name, id in zip(route['latitude'], route['longitude'], route['city'],route['id']):
        fl.Marker(location=[lat, lon], popup=(fl.Popup(name+ ' id = '+ str(id))),               icon = fl.Icon(color=color(id))).add_to(fg_city)
    
    fg_visit = fl.FeatureGroup(name="Visit")
    for lat, lon, name, id in zip(route['latitude'], route['longitude'], route['visit'],route['id']):
        fl.Marker(location=[lat, lon], popup=(fl.Popup(name+ ' id = '+ str(id))),              icon = fl.Icon(color=color(id))).add_to(fg_visit)
    
    fg_time = fl.FeatureGroup(name='South Korean covid-19|year=2020')
    for lat, lon, name, id in zip(route['latitude'], route['longitude'], route['province'],route['id']):
    
        y=route['id'][route['province']==name]
    
        date = [d.strftime('%m/%d') for d in y.index.date] #[]
    
        
        multi_iter2 = pd.DataFrame(y.values, index=date).sort_index()
        scatter = vincent.GroupedBar(multi_iter2, height=200, width=350)
        data = json.loads(scatter.to_json())
    
        v = fl.features.Vega(data, width='100%', height='100%')
        p = fl.Popup(name)
        pop =p.add_child(v)
        fl.features.Marker(location=[lat, lon], popup=pop,icon = fl.Icon(color=color(id))).add_to(fg_time)
    
    
    
    fg_province.add_to(mapped)
    fg_city.add_to(mapped)
    fg_visit.add_to(mapped)
    fg_time.add_to(mapped)
    fl.LayerControl().add_to(mapped)
    
    return mapped   


# In[ ]:


geo_map = geospace()
geo_map.save(outfile='South Korean Coronavirus.html')


# In[ ]:


geo_map


# In[ ]:


route_daily = route[['latitude', 'longitude', 'id']]


# In[ ]:


route_daily['day'] = [d.strftime('%m/%d') for d in route.index.date]
route_daily.head() 


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(15.5,5.5))
h=sns.boxplot(x='day', y='id', data=route_daily.sort_values(by=['day']), ax=ax)
ax.set_title('South Korean COVID-19|year=2020 boxplot')


# # Granger causality
# 
# We say X Granger-causes Y if prediction of values of Y based on its own past values and on the past values of X are better than predictions of Y based on its past value alone. 
# 
# or, It is based on the idea that if X causes Y, then the forecast of Y based on previous values of Y AND the previous values of X should outperform the forecast of Y based on previous values of Y alone. 
# 
# we are finding if there exist Granger causality between variable time series

# In[ ]:


# display each time series
from pmdarima import *


# In[ ]:


fx=utils.tsdisplay(route_daily['longitude'],title='longitude')


# In[ ]:


fy =  utils.tsdisplay(route_daily['id'], title='id')


#  Something happens in this three figure. we can see index interval [40, 60] and [60,80] for longitude and id plotting. Also [20,40] and [60, 80] for latitude plotting. we are checking if there exist Granger causality between feature.

# In[ ]:


fv = utils.tsdisplay(route_daily['latitude'], title='latitude')


# In[ ]:


ts_route = route_daily.copy()
ts_route = ts_route.reset_index()
ts_route = ts_route.sort_values(by=['date'])
ts_route.head()


# We are going to implement Granger causality with a statsmodel package below. it accepts only 2D array with 2 columns. The values are in the first column and the predictor (X) is in the second column. 
# 
# > **The Null hypothesis for grangercausalitytests** is that the time series in
# the second column, x2, does NOT Granger cause the time series in the first
# column, x1. Grange causality means that past values of x2 have a
# statistically significant effect on the current value of x1, taking past
# values of x1 into account as regressors. 
# 
# > We reject the null hypothesis that **x2 does not Granger cause x1**
# if the pvalues are below a desired size
# of the test.
# > The null hypothesis for all four test is that the coefficients
# corresponding to past values of the second time series are zero.
# - 'params_ftest', 'ssr_ftest' are based on F distribution
# - 'ssr_chi2test', 'lrtest' are based on chi-square distribution
# 
# Also see: **https://www.statisticshowto.datasciencecentral.com/granger-causality/**

# In[ ]:


def is_GrangerCause(data=None, maxlag=30):
    """This function find if x2 Granger cause x1 vis versa """    
    from statsmodels.tsa.stattools import grangercausalitytests
    gc = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    
    for i in range(maxlag):
        x=gc[i+1][0]
        p1 = x['lrtest'][1] # pvalue for lr test
        p2 = x['ssr_ftest'][1] # pvalue for ssr ftest
        p3 = x['ssr_chi2test'][1] #pvalue for ssr_chi2test
        p4 = x['params_ftest'][1] #pvalue for 'params_ftest'
        
        condition = ((p1 < 0.05 and p2 < 0.05) and (p3 < 0.05 and p4 < 0.05))
        
        if condition == True:
            cols = data.columns
            print('Yes: {} Granger causes {}'.format(cols[0], cols[1]))
            print('maxlag = {}\nResults: {}'.format(i, x))
            break
            
        else:
            if i == maxlag - 1:
                cols = data.columns
                print('No: {} does not Granger cause {}'.format(cols[0], cols[1]))


# In[ ]:


is_GrangerCause(data = ts_route[['longitude', 'id']])


# In[ ]:


is_GrangerCause(data = ts_route[['id','longitude']])


# In[ ]:


is_GrangerCause(data = ts_route[['id','latitude']])


# In[ ]:


is_GrangerCause(data = ts_route[['latitude','id']])


# In[ ]:


is_GrangerCause(data = ts_route[['latitude','longitude']])


# In[ ]:


is_GrangerCause(data = ts_route[['longitude','latitude']])


# **We can therefore say from the results that the prediction of the number of patients confirmed in the future, gives us the place where there will be the next contamination of covid-19. in other words, it is enough to know the number of patients confirmed in the future if we want to know which province, city or visit will be infected in the future.**
# 
# Thank for reading!
# you can download for your own use

# In[ ]:




