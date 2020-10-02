#!/usr/bin/env python
# coding: utf-8

# In[37]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
import plotly.tools as tools
import plotly

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

plotly.__version__
plotly.offline.init_notebook_mode() 

import plotly.plotly as py
import plotly.graph_objs as go


# In[38]:


#Load flat files
BloodTypeDistr = pd.read_csv('../input/bloodtypecountrydistributions/BloodTypeDistributions.csv', encoding='latin1')
CountryCodes = pd.read_csv('../input/countrycodes/CountryCodes.csv')


# In[39]:


BloodTypeDistr.head()


# In[40]:


CountryCodes.head()


# In[41]:


#Create counts for each populations
BloodTypeDistr = BloodTypeDistr.set_index('Country')
CountryCodes = CountryCodes.set_index('Name')


# In[43]:


#Create a population count column for each blood type and country

PopulationCounts = [('O+', 'O+ Population'),
                     ('A+', 'A+ Population'),
                     ('B+', 'B+ Population'),
                     ('AB+', 'AB+ Population'),
                     ('O-', 'O- Population'),
                     ('A-', 'A- Population'),
                     ('B-', 'B- Population'),
                     ('AB-', 'AB- Population')]

for BloodType, BloodTypeCounts in PopulationCounts:
    BloodTypeDistr[BloodTypeCounts] = BloodTypeDistr[BloodType].str.replace('%', '').astype(float)*.01*BloodTypeDistr['Population'].str.replace(',', '').astype(float)
    BloodTypeDistr[BloodTypeCounts] = BloodTypeDistr[BloodTypeCounts].astype(float)   


# In[44]:


BloodTypeDistr.head()


# In[45]:


BloodTypeCountryCodes = BloodTypeDistr.join(CountryCodes, how='right')


# In[46]:


BloodTypeCountryCodes.head()


# ## Choropleths detailing percent of population in each country with a specific blood typ

# In[47]:


data = [ dict(
        type = 'choropleth',
        locations = BloodTypeCountryCodes['alpha-3'],
        z = BloodTypeCountryCodes['O+'].str.replace('%', '').astype(float) + BloodTypeCountryCodes['O-'].str.replace('%', '').astype(float),
        text = BloodTypeCountryCodes.index,
        colorscale = 'Reds',
        autocolorscale = True,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) )
    )]


layout = dict(
    title = 'O global percent distribution',
    geo = dict(
        showframe = True,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
plotly.offline.iplot( fig, validate=False)


# In[14]:


data = [ dict(
        type = 'choropleth',
        locations = BloodTypeCountryCodes['alpha-3'],
        z = BloodTypeCountryCodes['A+'].str.replace('%', '').astype(float) + BloodTypeCountryCodes['A-'].str.replace('%', '').astype(float),
        text = BloodTypeCountryCodes.index,
        colorscale = 'Reds',
        autocolorscale = True,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) )
    )]


layout = dict(
    title = 'A global percent distribution',
    geo = dict(
        showframe = True,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
plotly.offline.iplot( fig, validate=False)


# In[16]:


data = [ dict(
        type = 'choropleth',
        locations = BloodTypeCountryCodes['alpha-3'],
        z = BloodTypeCountryCodes['B+'].str.replace('%', '').astype(float) + BloodTypeCountryCodes['B-'].str.replace('%', '').astype(float),
        text = BloodTypeCountryCodes.index,
        colorscale = 'Reds',
        autocolorscale = True,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) )
    )]


layout = dict(
    title = 'B global percent distribution',
    geo = dict(
        showframe = True,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
plotly.offline.iplot( fig, validate=False)


# In[26]:


data = [ dict(
        type = 'choropleth',
        locations = BloodTypeCountryCodes['alpha-3'],
        z = BloodTypeCountryCodes['AB+'].str.replace('%', '').astype(float) + BloodTypeCountryCodes['AB-'].str.replace('%', '').astype(float),
        text = BloodTypeCountryCodes.index,
        colorscale = 'Reds',
        autocolorscale = True,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) )
    )]


layout = dict(
    title = 'AB global percent distribution',
    geo = dict(
        showframe = True,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
plotly.offline.iplot( fig, validate=False)


# In[32]:


BloodTypeRegionGroup = BloodTypeDistr.join(CountryCodes, how='inner').reset_index()
BloodTypeRegionGroup = BloodTypeRegionGroup[['Population' ,'O+ Population', 'A+ Population',
                                            'B+ Population', 'AB+ Population',
                                            'O- Population', 'A- Population',
                                            'B- Population', 'AB- Population',
                                            'region', 'sub-region']]


# In[33]:


BloodTypeRegionGroup.head(5)


# In[34]:


BloodTypeRegionGroup = BloodTypeRegionGroup.groupby(['region', 'sub-region']).sum()
BloodTypeRegionGroup.head(25)


# In[58]:


BloodTypebyRegion = BloodTypeRegionGroup.copy()
BloodTypebyRegion['O Population'] = (BloodTypebyRegion['O+ Population'] + BloodTypebyRegion['O- Population'])
BloodTypebyRegion['A Population'] = (BloodTypebyRegion['A+ Population'] + BloodTypebyRegion['A- Population'])
BloodTypebyRegion['B Population'] = (BloodTypebyRegion['B+ Population'] + BloodTypebyRegion['B- Population'])
BloodTypebyRegion['AB Population'] = (BloodTypebyRegion['AB+ Population'] + BloodTypebyRegion['AB- Population'])

BloodTypebyRegion['Type_O_%'] = BloodTypebyRegion['O Population'] / (BloodTypebyRegion['O Population'] + BloodTypebyRegion['A Population'] +
                                                                     BloodTypebyRegion['B Population'] + BloodTypebyRegion['AB Population']) * 100
BloodTypebyRegion['Type_A_%'] = BloodTypebyRegion['A Population'] / (BloodTypebyRegion['O Population'] + BloodTypebyRegion['A Population'] +
                                                                     BloodTypebyRegion['B Population'] + BloodTypebyRegion['AB Population']) * 100
BloodTypebyRegion['Type_B_%'] = BloodTypebyRegion['B Population'] / (BloodTypebyRegion['O Population'] + BloodTypebyRegion['A Population'] +
                                                                     BloodTypebyRegion['B Population'] + BloodTypebyRegion['AB Population']) * 100
BloodTypebyRegion['Type_AB_%'] = BloodTypebyRegion['AB Population'] / (BloodTypebyRegion['O Population'] + BloodTypebyRegion['A Population'] +
                                                                       BloodTypebyRegion['B Population'] + BloodTypebyRegion['AB Population']) * 100

BloodTypebyRegion = BloodTypebyRegion[['Type_O_%', 'Type_A_%', 'Type_B_%', 'Type_AB_%']]

BloodTypebyRegion.head(30)


# In[65]:


BloodTypebyRegionIndexReset = BloodTypebyRegion.reset_index()

#Africa
trace1 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Africa'],
    y = BloodTypebyRegionIndexReset['Type_O_%'][BloodTypebyRegionIndexReset.region == 'Africa'],
    name = 'Type O %',
    marker = dict( color = '#373e02')                                           
)
trace2 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Africa'],
    y = BloodTypebyRegionIndexReset['Type_A_%'][BloodTypebyRegionIndexReset.region == 'Africa'],
    name = 'Type A %',
    marker = dict( color = '#fe7b7c')                                           
)
trace3 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Africa'],
    y = BloodTypebyRegionIndexReset['Type_B_%'][BloodTypebyRegionIndexReset.region == 'Africa'],
    name = 'Type B %',
    marker = dict( color = '#49759c')                                           
)
trace4 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Africa'],
    y = BloodTypebyRegionIndexReset['Type_AB_%'][BloodTypebyRegionIndexReset.region == 'Africa'],
    name = 'Type AB %',
    marker = dict( color = '#5ca904')                                           
)

#Americas
trace5 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Americas'],
    y = BloodTypebyRegionIndexReset['Type_O_%'][BloodTypebyRegionIndexReset.region == 'Americas'],
    name = 'Type O %',
    showlegend=False,
    marker = dict( color = '#373e02')                                           
)
trace6 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Americas'],
    y = BloodTypebyRegionIndexReset['Type_A_%'][BloodTypebyRegionIndexReset.region == 'Americas'],
    name = 'Type A %',
    showlegend=False,
    marker = dict( color = '#fe7b7c')                                           
)
trace7 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Americas'],
    y = BloodTypebyRegionIndexReset['Type_B_%'][BloodTypebyRegionIndexReset.region == 'Americas'],
    name = 'Type B %',
    showlegend=False,
    marker = dict( color = '#49759c')                                           
)
trace8 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Americas'],
    y = BloodTypebyRegionIndexReset['Type_AB_%'][BloodTypebyRegionIndexReset.region == 'Americas'],
    name = 'Type AB %',
    showlegend=False,
    marker = dict( color = '#5ca904')                                           
)

#Asia
trace9 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Asia'],
    y = BloodTypebyRegionIndexReset['Type_O_%'][BloodTypebyRegionIndexReset.region == 'Asia'],
    name = 'Type O %',
    showlegend=False,
    marker = dict( color = '#373e02')                                           
)
trace10 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Asia'],
    y = BloodTypebyRegionIndexReset['Type_A_%'][BloodTypebyRegionIndexReset.region == 'Asia'],
    name = 'Type A %',
    showlegend=False,
    marker = dict( color = '#fe7b7c')                                           
)
trace11 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Asia'],
    y = BloodTypebyRegionIndexReset['Type_B_%'][BloodTypebyRegionIndexReset.region == 'Asia'],
    name = 'Type B %',
    showlegend=False,
    marker = dict( color = '#49759c')                                           
)
trace12 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Asia'],
    y = BloodTypebyRegionIndexReset['Type_AB_%'][BloodTypebyRegionIndexReset.region == 'Asia'],
    name = 'Type AB %',
    showlegend=False,
    marker = dict( color = '#5ca904')                                           
)

#Europe
trace13 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Europe'],
    y = BloodTypebyRegionIndexReset['Type_O_%'][BloodTypebyRegionIndexReset.region == 'Europe'],
    name = 'Type O %',
    showlegend=False,
    marker = dict( color = '#373e02')                                           
)
trace14 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Europe'],
    y = BloodTypebyRegionIndexReset['Type_A_%'][BloodTypebyRegionIndexReset.region == 'Europe'],
    name = 'Type A %',
    showlegend=False,
    marker = dict( color = '#fe7b7c')                                           
)
trace15 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Europe'],
    y = BloodTypebyRegionIndexReset['Type_B_%'][BloodTypebyRegionIndexReset.region == 'Europe'],
    name = 'Type B %',
    showlegend=False,
    marker = dict( color = '#49759c')                                           
)
trace16 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Europe'],
    y = BloodTypebyRegionIndexReset['Type_AB_%'][BloodTypebyRegionIndexReset.region == 'Europe'],
    name = 'Type AB %',
    showlegend=False,
    marker = dict( color = '#5ca904')                                           
)

#Europe
trace17 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Oceania'],
    y = BloodTypebyRegionIndexReset['Type_O_%'][BloodTypebyRegionIndexReset.region == 'Oceania'],
    name = 'Type O %',
    showlegend=False,
    marker = dict( color = '#373e02')                                           
)
trace18 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Oceania'],
    y = BloodTypebyRegionIndexReset['Type_A_%'][BloodTypebyRegionIndexReset.region == 'Oceania'],
    name = 'Type A %',
    showlegend=False,
    marker = dict( color = '#fe7b7c')                                           
)
trace19 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Oceania'],
    y = BloodTypebyRegionIndexReset['Type_B_%'][BloodTypebyRegionIndexReset.region == 'Oceania'],
    name = 'Type B %',
    showlegend=False,
    marker = dict( color = '#49759c')                                           
)
trace20 = go.Bar(
    x = BloodTypebyRegionIndexReset['sub-region'][BloodTypebyRegionIndexReset.region == 'Oceania'],
    y = BloodTypebyRegionIndexReset['Type_AB_%'][BloodTypebyRegionIndexReset.region == 'Oceania'],
    name = 'Type AB %',
    showlegend=False,
    marker = dict( color = '#5ca904')                                           
)

fig = tools.make_subplots(rows=5, cols=1, subplot_titles=('Africa', 'Americas', 'Asia', 'Europe', 'Oceania'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 1)
fig.append_trace(trace5, 2, 1)
fig.append_trace(trace6, 2, 1)
fig.append_trace(trace7, 2, 1)
fig.append_trace(trace8, 2, 1)
fig.append_trace(trace9, 3, 1)
fig.append_trace(trace10, 3, 1)
fig.append_trace(trace11, 3, 1)
fig.append_trace(trace12, 3, 1)
fig.append_trace(trace13, 4, 1)
fig.append_trace(trace14, 4, 1)
fig.append_trace(trace15, 4, 1)
fig.append_trace(trace16, 4, 1)
fig.append_trace(trace17, 5, 1)
fig.append_trace(trace18, 5, 1)
fig.append_trace(trace19, 5, 1)
fig.append_trace(trace20, 5, 1)

fig['layout'].update(height=1000, width=1000, barmode='stack')
plotly.offline.iplot(fig, validate=False)


# In[22]:


RHRegionGroup = BloodTypeRegionGroup.copy()
RHRegionGroup['Positive'] = ( RHRegionGroup['O+ Population'] + RHRegionGroup['A+ Population'] +
                              RHRegionGroup['B+ Population'] + RHRegionGroup['AB+ Population'] )
RHRegionGroup['Negative'] = ( RHRegionGroup['O- Population'] + RHRegionGroup['A- Population'] +
                              RHRegionGroup['B- Population'] + RHRegionGroup['AB- Population'] )

RHRegionGroup = RHRegionGroup.drop(RHRegionGroup.columns[:8], axis=1)

RHRegionGroup['Positive_%'] = RHRegionGroup['Positive'] / ( RHRegionGroup['Positive'] + RHRegionGroup['Negative'] ) * 100
RHRegionGroup['Negative_%'] = RHRegionGroup['Negative'] / ( RHRegionGroup['Positive'] + RHRegionGroup['Negative'] ) * 100


# In[23]:


RHRegionGroup.head(30)


# In[24]:


RHRegionGroup.reset_index()[RHRegionGroup.reset_index().region == 'Africa']


# In[25]:


RHRegionGroupIndexReset = RHRegionGroup.reset_index()

trace1 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Africa'],
    y = RHRegionGroupIndexReset['Positive_%'][RHRegionGroupIndexReset.region == 'Africa'],
    name = 'RH + Percentage',
    marker = dict( color = '#047495' )
)
trace2 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Africa'],
    y = RHRegionGroupIndexReset['Negative_%'][RHRegionGroupIndexReset.region == 'Africa'],
    name = 'RH - Percentage',
    marker = dict( color = '#ff474c' )
)
trace3 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Americas'],
    y = RHRegionGroupIndexReset['Positive_%'][RHRegionGroupIndexReset.region == 'Americas'],
    name = 'RH + Percentage',
    showlegend=False,
    marker = dict( color = '#047495' )
)
trace4 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Americas'],
    y = RHRegionGroupIndexReset['Negative_%'][RHRegionGroupIndexReset.region == 'Americas'],
    name = 'RH - Percentage',
    showlegend=False,
    marker = dict( color = '#ff474c' )
)
trace5 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Asia'],
    y = RHRegionGroupIndexReset['Positive_%'][RHRegionGroupIndexReset.region == 'Asia'],
    name = 'RH + Percentage',
    showlegend=False,
    marker = dict( color = '#047495' )
)
trace6 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Asia'],
    y = RHRegionGroupIndexReset['Negative_%'][RHRegionGroupIndexReset.region == 'Asia'],
    name = 'RH - Percentage',
    showlegend=False,
    marker = dict( color = '#ff474c' )
)
trace7 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Europe'],
    y = RHRegionGroupIndexReset['Positive_%'][RHRegionGroupIndexReset.region == 'Europe'],
    name = 'RH + Percentage',
    showlegend=False,
    marker = dict( color = '#047495' )
)
trace8 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Europe'],
    y = RHRegionGroupIndexReset['Negative_%'][RHRegionGroupIndexReset.region == 'Europe'],
    name = 'RH - Percentage',
    showlegend=False,
    marker = dict( color = '#ff474c' )
)
trace9 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Oceania'],
    y = RHRegionGroupIndexReset['Positive_%'][RHRegionGroupIndexReset.region == 'Oceania'],
    name = 'RH + Percentage',
    showlegend=False,
    marker = dict( color = '#047495' )
)
trace10 = go.Bar(
    x = RHRegionGroupIndexReset['sub-region'][RHRegionGroupIndexReset.region == 'Oceania'],
    y = RHRegionGroupIndexReset['Negative_%'][RHRegionGroupIndexReset.region == 'Oceania'],
    name = 'RH - Percentage',
    showlegend=False,
    marker = dict( color = '#ff474c' )
)

fig = tools.make_subplots(rows=5, cols=1, subplot_titles=('Africa', 'Americas', 'Asia', 'Europe', 'Oceania'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 3, 1)
fig.append_trace(trace6, 3, 1)
fig.append_trace(trace7, 4, 1)
fig.append_trace(trace8, 4, 1)
fig.append_trace(trace9, 5, 1)
fig.append_trace(trace10, 5, 1)

fig['layout'].update(height=1000, width=1000, barmode='stack')
plotly.offline.iplot(fig, validate=False)


# In[ ]:





# In[ ]:





# In[ ]:




