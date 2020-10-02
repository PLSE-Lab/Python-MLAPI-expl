#!/usr/bin/env python
# coding: utf-8

# In[18]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from bokeh.io import show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
output_notebook()
import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# In[19]:


wdi_data = pd.read_csv('../input/WDIData.csv')


# In[20]:


wdi_data.tail(4)


# In[21]:


countries = wdi_data['Country Name'].unique()
print(countries[:10])


# For my analysis I will choose top 5 countries with highest GDP as of 2017.

# In[22]:


selected_countries = ['China', 'Japan', 'India', 'Germany', 'United States']


# Since the dataset has many indicators, I chose some common words for filtering and selecting required indicators.

# In[23]:


word_list = ['employment', 'health', 'growth', 'education' , 'birth', 'goods', 'men', 'women', 'income', 'gni', 'male', 'female', 'death', 'ratio', 'gdp', 'energy', 'total']
ind_uniq = wdi_data['Indicator Name'].unique()
required_ind = []
for i in ind_uniq:
    for j in word_list:
        if j in i.lower().split():
            required_ind.append(i)


# In[24]:


# List of indicators matching selection
required_ind


# In[25]:


# Select required indicators
selected_ind = ['Urban population growth (annual %)', 'GDP growth (annual %)', 'Sex ratio at birth (male births per female births)', 
 'Current health expenditure per capita (current US$)', 'Imports of goods and services (current US$)', 'Exports of goods and services (current US$)',
               'GNI per capita, Atlas method (current US$)','Life expectancy at birth, female (years)', 
                'Life expectancy at birth, male (years)','Adolescent fertility rate (births per 1,000 women ages 15-19)',  
                'Death rate, crude (per 1,000 people)', 'School enrollment, primary, female (% gross)', 'School enrollment, primary, male (% gross)',
               'School enrollment, secondary, female (% gross)', 'School enrollment, secondary, male (% gross)', 
                'GDP (current US$)', 'Renewable energy consumption (% of total final energy consumption)',
                'Unemployment, total (% of total labor force) (national estimate)', 'Unemployment, total (% of total labor force) (modeled ILO estimate)',
               'Mortality rate, infant, female (per 1,000 live births)']
selected_ind = pd.Series(selected_ind)
selected_ind


# In[26]:


# Subset data based on selected countries and indicators
subset = wdi_data[wdi_data['Country Name'].isin(selected_countries) & wdi_data['Indicator Name'].isin(selected_ind)]


# In[27]:


# Cleaning
subset = subset.drop(list(subset.columns)[4:51]+list(subset.columns)[-2:], axis=1)
subset = subset.rename(columns={"Indicator Code": "IndiCode", "Country Code": "CountCode", 
                       "Indicator Name": "IndiName", "Country Name": "CountName"})


# In[28]:


# List of columns in dataframe
cols = pd.Series(subset.columns)
cols


# In[29]:


# Dict with (indicator code, indicator name) as (key,value) pair
indicators = dict(zip(subset['IndiCode'], subset['IndiName']))
indicators


# # Analysis of GDP

# In[30]:


from bokeh.core.properties import value
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource

lowestGNI_2017 = subset.query("IndiCode == 'NY.GDP.MKTP.CD'").sort_values(by = '2017', ascending = False)
lowestGNI_2007 = subset.query("IndiCode == 'NY.GDP.MKTP.CD'").sort_values(by = '2007', ascending = False)
tmp = {'Countries': lowestGNI_2017['CountName'].values, '2007': lowestGNI_2007['2007'].values, '2017': lowestGNI_2017['2017'].values}


years = ['2007', '2017']
perChg= tmp['2017']-tmp['2007']
perChg = np.divide(perChg, tmp['2017'])*100

tmp['chg'] = perChg

p = figure(x_range = tmp['Countries'], plot_height=350, title="(A) A Comparsion between 2007/2017 GDP (current US$)")

colors = ['blue', 'green']
p.vbar_stack(years, x='Countries', color= colors, width=0.9, source=tmp,
             legend=[value(x) for x in years])

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_right"
p.legend.orientation = "horizontal"
hover = HoverTool(
    tooltips=[
        ( 'Country',   '@Countries'),
        ( 'GDP 2007',  '$@2007' ),
        ( 'GDP 2017', '$@2017'  ),
        ('%Change', '@chg%')
    ]
)
p.add_tools(hover)

### p2 ####
sorted_chg = sorted(tmp['Countries'], key=lambda x: tmp['chg'][list(tmp['Countries']).index(x)])

p2 = figure(x_range=sorted_chg, plot_height=350, plot_width=750, title="(B) % GDP growth 2007-2017. China is the fastest growing economy with 2nd largest GDP (A) in this period.")

p2.vbar(x=tmp['Countries'], top=tmp['chg'], width=0.9)

p2.xgrid.grid_line_color = None
p2.y_range.start = 0
p2.y_range.end = 200
show(row(p,p2))


# In above plots Fig A shows GDP for years 2007/2017, wheras Fig B displays percent change in GDP between 2007-2017. Although, US has the highest GDP its rate of GDP is slow between this period. Whereas China and India showed highest GDP growth in this period. (Hover on plot to get a bigger picture of values)
# 
# GNI helps in identifying inequality in a society. I will now try to infer any patterns in GNI index, Unemployment, School enrollment and Renewable energy consumption for top 5 GDP countries. 

# # Analysis of GNI

# In[31]:


# how gni effects, mortality female, education secondary female, unemployment
color = ['#440154', 'blue', '#208F8C', 'red', '#FDE724']


# GNI plot
df = subset.query('IndiCode == "NY.GNP.PCAP.CD"')
s1 = figure(plot_width=550, plot_height=350, title='GNI per capita, Atlas method (current US$)')
xs = [cols[4:15].tolist()]*5
ys = df[df.columns[4:15]].values.tolist()
data = {'xs': xs, 'ys' : ys, 'label': list(df['CountName'].values)}
data['colors'] = color
src = ColumnDataSource(data)
s1.multi_line(xs='xs', ys='ys', source= src,color = 'colors', legend= 'label', line_width=2)
s1.legend.location = "top_left"
s1.legend.click_policy="hide"
hover = HoverTool(
    tooltips=[
        ( 'Year',   '$x'),
        ( 'Value',  '$y' )
    ]
)
s1.add_tools(hover)
s1.x_range.start == '2007'

# Unemployment
df = subset.query('IndiCode == "SL.UEM.TOTL.ZS"')
s2 = figure(plot_width=550, plot_height=350, title='Unemployment, total (% of total labor force) (modeled ILO estimate)')
xs = [cols[4:15].tolist()]*5
ys = df[df.columns[4:15]].values.tolist()
data = {'xs': xs, 'ys' : ys, 'label': list(df['CountName'].values)}
data['colors'] = color
src = ColumnDataSource(data)
s2.multi_line(xs='xs', ys='ys', source= src,color = 'colors', legend= 'label', line_width=2)
s2.legend.location = "top_right"
s2.add_tools(hover)
s2.x_range.start == '2007'

# Renewable
df = subset.query('IndiCode == "EG.FEC.RNEW.ZS"')
s3 = figure(plot_width=550, plot_height=350, title='Renewable energy consumption (% of total final energy consumption)')
xs = [cols[4:15].tolist()]*5
ys = df[df.columns[4:15]].values.tolist()
data = {'xs': xs, 'ys' : ys, 'label': list(df['CountName'].values)}
data['colors'] = color
src = ColumnDataSource(data)
s3.multi_line(xs='xs', ys='ys', source= src,color = 'colors', legend= 'label', line_width=2)
s3.legend.location = "top_left"
s3.add_tools(hover)
s3.x_range.start == '2007'

# Education 
df = subset.query('IndiCode == "SE.SEC.ENRR.FE"')
s4 = figure(plot_width=550, plot_height=350, title='School enrollment, secondary, female (% gross)')
xs = [cols[4:14].tolist()]*5
ys = df[df.columns[4:15]].values.tolist()
data = {'xs': xs, 'ys' : ys, 'label': list(df['CountName'].values)}
data['colors'] = color
src = ColumnDataSource(data)
s4.multi_line(xs='xs', ys='ys', source= src,color = 'colors', legend= 'label', line_width=2)
s4.legend.location = "top_left"
s4.add_tools(hover)
s4.x_range.start == '2007'

show(row(s1,s2))
show(row(s3,s4))


# It can be observed from the above graphs-
# 1. a. Unemployment % rose to all time high in 2010 for United States, although its average income (GNI) increased during the period 2007-2017  
#    b. We can also study related economic markers in this dataset which may help us identify relationships with Unemployment.   
# 2. a. Altough India is the second fastest growing economy, its renewable energy consumption is quite high and decreasing. It also shows an opposite trend w.r.t developed countries.  
#    b. China's renewable consumption declined during 2007-2011.   
#    c. Germany is steadily shifting towards Renewable resources.   
# 3. Japan's GNI and unemployment has a valley trend during years 2009-2015.  
# 
# (Hover on plots to see values)
# 
# 
# I will further analyze any coorelation between GNI and other factors (although the sample size is quite small < 30 samples)-

# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
vals = {}
f, axes = plt.subplots(1, 5, figsize=(40,7))
c = 0
for i in selected_countries:
    df = subset[subset['CountName'].isin([i]) & subset['IndiCode'].isin(["NY.GNP.PCAP.CD","SL.UEM.TOTL.ZS","EG.FEC.RNEW.ZS","SE.SEC.ENRR.FE"])] 
    vals['Indicator'] = df['IndiCode'].values
    nwdf = df[df.columns[4:15]]
    nwdf = nwdf.T
    cordf = nwdf.corr()
    cordf.columns = vals['Indicator']
    cordf.index = vals['Indicator']
    vals[i] = cordf.iloc[0].values
    sns.heatmap(cordf, ax=axes[c]).set_title('Coorelation for {}'.format(i))
    c+=1
plt.subplots_adjust(left=0.15)
plt.show() 


# We can see their are high positive and newgative coorelation between GNI and other factors. This coorelation varies with different countries.

# In[33]:


cor_df = pd.DataFrame(vals)

cor_df


# In[ ]:




