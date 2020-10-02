#!/usr/bin/env python
# coding: utf-8

# # Data Import + some settings

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import collections as col

pd.set_option('max_columns', 50)
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 3

df = pd.read_csv("../input/database.csv")

# Scale costs to $1M for plot clarity
df2 = df.copy()
df2['All Costs'] = df2['All Costs']/1000000.0
df2['Property Damage Costs'] = df2['Property Damage Costs']/1000000.0
df2['Lost Commodity Costs'] = df2['Lost Commodity Costs']/1000000.0
df2['Public/Private Property Damage Costs'] = df2['Public/Private Property Damage Costs']/1000000.0
df2['Emergency Response Costs'] = df2['Emergency Response Costs']/1000000.0
df2['Environmental Remediation Costs'] = df2['Environmental Remediation Costs']/1000000.0
df2['Other Costs'] = df2['Other Costs']/1000000.0


# # Time Dependancy
# 
# We'll focus on a few things here:
# 
# * Course time-dependency (as a function of year)
# * Look into the details a bit, month-by-month
# * We are given the cost of each of the oil spills and a rough breakdown of where those costs came, so where is the majority of the cost?
# 
# 

# In[ ]:


# Some calculations to plot distributions by year

headers = ['Accident Year','All Costs','Property Damage Costs',
           'Lost Commodity Costs','Public/Private Property Damage Costs',
           'Emergency Response Costs','Environmental Remediation Costs', 'Other Costs']
df_forPlots = df2[headers]
df_forPlots = df_forPlots[df_forPlots['Accident Year'] != 2017]
by_year = df_forPlots.groupby('Accident Year')

xPt = by_year.sum()['All Costs'].index.values
yAll_sum = by_year.sum()['All Costs'].values
yPro_sum = by_year.sum()['Property Damage Costs'].values
yCom_sum = by_year.sum()['Lost Commodity Costs'].values
yPri_sum = by_year.sum()['Public/Private Property Damage Costs'].values
yEme_sum = by_year.sum()['Emergency Response Costs'].values
yEnv_sum = by_year.sum()['Environmental Remediation Costs'].values

fracProp = by_year.mean()['Property Damage Costs'].values / by_year.mean()['All Costs'].values
fracComm = by_year.mean()['Lost Commodity Costs'].values / by_year.mean()['All Costs'].values
fracPriP = by_year.mean()['Public/Private Property Damage Costs'].values / by_year.mean()['All Costs'].values
fracEmer = by_year.mean()['Emergency Response Costs'].values / by_year.mean()['All Costs'].values
fracEnvi = by_year.mean()['Environmental Remediation Costs'].values / by_year.mean()['All Costs'].values


# In[ ]:



# Year-by-year plot

f, a = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

a[0].plot(xPt, yAll_sum, label="All", color='black')
a[0].plot(xPt, yEme_sum,label="Emergency", color='orange')
a[0].plot(xPt, yEnv_sum,label="Environment", color='yellow')
a[0].plot(xPt, yPro_sum,label="Property", color='darkblue')
a[0].plot(xPt, yCom_sum,label="Commodity", color='darkred')
a[0].plot(xPt, yPri_sum,label="Public/Private Prop.", color='darkgreen')
a[0].fill_between(xPt, yAll_sum, 10E0, facecolor='black', alpha=0.1)
a[0].fill_between(xPt, yPro_sum, 10E0, facecolor='darkblue', alpha=0.1)
a[0].fill_between(xPt, yCom_sum, 10E0, facecolor='darkred', alpha=0.1)
a[0].fill_between(xPt, yPri_sum, 10E0, facecolor='darkgreen', alpha=0.1)
a[0].fill_between(xPt, yEme_sum, 10E0, facecolor='orange', alpha=0.1)
a[0].fill_between(xPt, yEnv_sum, 10E0, facecolor='yellow', alpha=0.1)
a[0].get_xaxis().get_major_formatter().set_useOffset(False)
a[0].set_xlabel('Year', fontsize=16)
a[0].set_ylabel('Sum Cost ($Million)', fontsize=16)
a[0].legend()


a[1].plot(xPt, fracEmer,label="Emergency", color='orange')
a[1].plot(xPt, fracEnvi,label="Environment", color='yellow')
a[1].plot(xPt, fracProp,label="Property", color='darkblue')
a[1].plot(xPt, fracComm,label="Commodity", color='darkred')
a[1].plot(xPt, fracPriP,label="Public/Private Prop.", color='darkgreen')
a[1].fill_between(xPt, fracProp, 0, facecolor='darkblue', alpha=0.1)
a[1].fill_between(xPt, fracComm, 0, facecolor='darkred', alpha=0.1)
a[1].fill_between(xPt, fracPriP, 0, facecolor='darkgreen', alpha=0.1)
a[1].fill_between(xPt, fracEmer, 0, facecolor='orange', alpha=0.1)
a[1].fill_between(xPt, fracEnvi, 0, facecolor='yellow', alpha=0.1)
a[1].get_xaxis().get_major_formatter().set_useOffset(False)
a[1].set_xlabel('Year', fontsize=16)
a[1].set_ylabel('Fraction of the Total Cost', fontsize=16)
a[1].set_ylim([0,1])
tmp = a[1].legend()


# The left plot shows the summed cost of all the pipeline spills year-by-year for the data given.  Summed cost is shown as the total ('All') spill costs as well as the major breakdowns given in the data.  
# 
# Some immediate takeaways:
# 
# * 2010 was a stand-out year with about 5 times the total expenses incurred
# 
# * The cost is largely dominated by environmental + emergency clean-up costs.  Property is the next highest factor hovering at around ~20% of the cost for any given year.  2014 is an outlier year where the total property costs incurred a highest singular fraction.
# 
# 
# ### Let's look into the details...

# In[ ]:



df3 = df2.copy()
df3['Month'] = df3['Accident Date/Time'].apply(lambda x: (str(x)[0:2]))
df3['Month'] = df3['Month'].apply(lambda x: int(''.join(c for c in x if c.isdigit())))
df3['Rel Month'] = (df3['Accident Year']-2010)*12. + df3['Month']

headers = headers = ['Accident Year','All Costs','Property Damage Costs',
           'Lost Commodity Costs','Public/Private Property Damage Costs',
           'Emergency Response Costs','Environmental Remediation Costs', 
            'Other Costs', 'Rel Month']
by_month = df3[headers].groupby(['Rel Month'])


xPt = by_month.sum()['All Costs'].index.values
yAll_sum = by_month.sum()['All Costs'].values
yPro_sum = by_month.sum()['Property Damage Costs'].values
yCom_sum = by_month.sum()['Lost Commodity Costs'].values
yPri_sum = by_month.sum()['Public/Private Property Damage Costs'].values
yEme_sum = by_month.sum()['Emergency Response Costs'].values
yEnv_sum = by_month.sum()['Environmental Remediation Costs'].values

fracProp = by_month.mean()['Property Damage Costs'].values / by_month.mean()['All Costs'].values
fracComm = by_month.mean()['Lost Commodity Costs'].values / by_month.mean()['All Costs'].values
fracPriP = by_month.mean()['Public/Private Property Damage Costs'].values / by_month.mean()['All Costs'].values
fracEmer = by_month.mean()['Emergency Response Costs'].values / by_month.mean()['All Costs'].values
fracEnvi = by_month.mean()['Environmental Remediation Costs'].values / by_month.mean()['All Costs'].values


# In[ ]:



fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

plt.plot(xPt, yAll_sum, label="All", color='black')
plt.plot(xPt, yEme_sum,label="Emergency", color='orange')
plt.plot(xPt, yEnv_sum,label="Environment", color='yellow')
plt.plot(xPt, yPro_sum,label="Property", color='darkblue')
plt.plot(xPt, yCom_sum,label="Commodity", color='darkred')
plt.plot(xPt, yPri_sum,label="Public/Private Prop.", color='darkgreen')
plt.fill_between(xPt, yAll_sum, 10E0, facecolor='black', alpha=0.1)
plt.fill_between(xPt, yPro_sum, 10E0, facecolor='darkblue', alpha=0.1)
plt.fill_between(xPt, yCom_sum, 10E0, facecolor='darkred', alpha=0.1)
plt.fill_between(xPt, yPri_sum, 10E0, facecolor='darkgreen', alpha=0.1)
plt.fill_between(xPt, yEme_sum, 10E0, facecolor='orange', alpha=0.1)
plt.fill_between(xPt, yEnv_sum, 10E0, facecolor='yellow', alpha=0.1)
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.xlabel('Relative Month (starting from Jan. 2010)', fontsize=19)
plt.ylabel('Sum Cost ($Millions)', fontsize=19)
tmp = plt.legend()
ax.text(9,600, "$840 Million Cost\nEnbridge Energy\nMarshall, MI",fontsize=11)
ax.text(13,160, "$135 Million Cost\nExxonMobil\nLaurel, MT",fontsize=11)
ax.text(35,160, "$91 Million Cost\nMobil\nMayflower, AR",fontsize=11)
ax.text(58,190, "$143 Million Cost\nPlains Pipeline Co\nGoleta, CA",fontsize=11)
ax.text(73,100, "$66 Million Cost\nColonial Pipeline\nHelena, AL",fontsize=11)


# Looking with a finer granularity (monthly) will give us an additional window into the data.  Here we see that not all oil-spills are created equally.  In fact, each of the yearly sums above were dominated by a single costly oil spill.  Here are the specifics of these top five costly spills...

# In[ ]:


df_byCost = df.sort_values(by=['All Costs'], ascending=False)
info = ['Accident Year','Operator Name','Accident City', 'Accident State', 'Cause Category'
       ,'Cause Subcategory','All Costs']
df_byCost[info].head()


# # Mapping the spills
# 
# 
# There are a couple of other important dimensions given in the data.  One of the big ones is that we are given the location of each of these spills.  So, let's map them and have a look where these barrels are spilling...
# 
# * Frequency state-by-state
# * Individually, weighted by the size of the spill in barrels
# 

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()


# In[ ]:


us_states = np.asarray(['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                        'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                        'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                        'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                        'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])

# Count the frequency state-by-state
AAA = col.Counter(df['Accident State'].values)
states = np.asarray(AAA.keys())
oil_perstate = np.asarray(AAA.values())

# Some states aren't in the data (good for them!), add them as zeros
addStates = np.asarray(['AZ','DC','DE','NH','RI','VT'])
addValues = np.asarray([0,0,0,0,0,0])
states = np.append(states,addStates)
oil_perstate = np.append(oil_perstate,addValues)


# In[ ]:


color_scale = [[0, 'rgb(225, 235, 255)'], [1, 'rgb(0, 81, 163)']]

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = color_scale,
        showscale = False,
        locations = states,
        locationmode = 'USA-states',
        z = oil_perstate,
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 1)
            )
        )]

layout = dict(
         title = 'Number of Oilspills by State (2010-2017)',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = data, layout = layout)
iplot(figure)


# Some take-aways:
# 
# * Texas wins!  By far.  I guess that we could've guessed that.  
# 
# * However, it is also interesting to note that nearly every state is represented in the data.  Spills are ubiquitious and not just in states like Texas, Oklahoma, or California that are known for oil resources.  Only five states (plus DC, our local 'non-state') aren't in there: Arizona, Delaware, New Hanpshire, Rhode Island, and Vermont.  

# In[ ]:


colors = ["black","black","darkblue","darkblue","darkgreen","darkgreen","darkred","darkred","orange"]
limits = [(0,100),(100,200),(200,300),(300,500),(500,1000),(1000,2000),(2000,5000),(5000,10000),(10000,36000)]
scale = 20
spills = []


for i in range(len(limits)):
    lim = limits[i]
    df_sub = df[((df['Net Loss (Barrels)']>lim[0]) & (df['Net Loss (Barrels)']<lim[1]))]
    spill = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['Accident Longitude'],
        lat = df_sub['Accident Latitude'],
        opacity=0.7,
        marker = dict(
            size = df_sub['Net Loss (Barrels)']/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'           
        ),
        name = '{0} - {1}'.format(lim[0],lim[1]) )
    spills.append(spill)

layout = dict(
        title = 'Net Barrel Loss of Individual Oil Spills: Jan/2010 - Jan/2017<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'silver',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)",
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'
        ),
    )

fig = dict( data=spills, layout=layout )
iplot(fig)


# This is interesting for several reasons.  
# 
# * You can see the individual pipelines snake across the country, particularly if you toggle the map to only show the 0-100 data.
# 
# * There are relatively sizable spills all over the country.  I am an avid news junkie, I find it kind of surprising that I haven't heard of many of these spills.  Spills are ubiquitious!
# 
# * The costliest spills we saw earlier don't seem to be the largest spills (by net barrel).  There must be some other factor at play that causes big cleanup costs.
#     - The states with the 5 highest costing spills: Michigan, California, Montana, Arkansas, Alabama

# # Which operators/causes are responsible for the most spills?

# In[ ]:


#### MOST COMMON OPERATORS
CCC = col.Counter(df['Operator ID'].values)
ids = [x[0] for x in CCC.most_common(20)]

hXvalue = []
hValues = []
hLabels = []
# Need to link the operator Id to the operator name, this isn't a 1-to-1 relationship
for j, i  in enumerate(ids):
    #print df['Operator Name'].loc[df['Operator ID'] == i].unique(), CCC[i]
    hXvalue.append(j+1)
    hValues.append(CCC[i])
    hLabels.append(" or ".join(df['Operator Name'].loc[df['Operator ID'] == i].unique()))

    
#### MOST COMMON SPILL CAUSES
DDD = col.Counter(df['Cause Category'].values)
ids2 = [x[0] for x in DDD.most_common(20)]

hXvalue2 = []
hValues2 = []
hLabels2 = []
for j, i  in enumerate(ids2):
    hXvalue2.append(j+1)
    hValues2.append(DDD[i])
    hLabels2.append(i)    


# In[ ]:



fig = plt.figure(figsize=(7, 7))
plt.barh(hXvalue[::-1], hValues, align='center', color='forestgreen')
plt.ylim([0,len(hValues)+1])
plt.title('Top 20 Most Frequent \'Spillers\' (Operator)', fontsize=20, y=1.04)
y = plt.yticks(hXvalue[::-1], hLabels, fontsize=10)
x = plt.xlabel('Number of Spills', fontsize=20)
x2 = plt.xticks(fontsize=15)


# In[ ]:



fig = plt.figure(figsize=(7, 7))
plt.barh(hXvalue2[::-1], hValues2, align='center', color='peru')
plt.ylim([0,len(hValues2)+1])
plt.title('Top 20 Most Frequent Spill Causes', fontsize=20, y=1.04)
y = plt.yticks(hXvalue2[::-1], hLabels2, fontsize=10)
x = plt.xlabel('Number of Spills', fontsize=20)
x2 = plt.xticks(fontsize=15)


# 

# # Correlation between cost and spill size, other factors...

# My naive guess would be that the bigger the spill size (in barrels) would lead to a most costly clean-up and damages.  However, we saw above that the most costly spills in the data aren't in fact the largest.  So there are other factors that come in to play.
# 
# This section is an exploration of this relationship and is definitely a work in progress.  

# In[ ]:


f, a = plt.subplots(nrows=3, ncols=1, figsize=(6,10), sharex=True)
f.subplots_adjust(hspace=0.)

a[0].scatter(df['Net Loss (Barrels)'].values,
              df['All Costs'].values, 
              label="All", color='darkred')
a[0].set_yscale('log')
a[0].set_ylim(0.1,4e9)
a[0].set_xscale('log')
a[0].set_xlim(0.001,100000)
a[0].set_ylabel('Cost', fontsize=13)
a[1].scatter(df['Net Loss (Barrels)'].values,
              df['Environmental Remediation Costs'].values, 
              label="All", color='darkblue')
a[1].set_yscale('log')
a[1].set_ylim(0.1,4e9)
a[1].set_xscale('log')
a[1].set_xlim(0.001,100000)
a[1].set_ylabel('Cost', fontsize=13)
a[2].scatter(df['Net Loss (Barrels)'].values,
              df['Emergency Response Costs'].values, 
              label="All", color='darkgreen')
a[2].set_yscale('log')
a[2].set_ylim(0.1,4e9)
a[2].set_xscale('log')
a[2].set_xlim(0.001,100000)
a[2].set_ylabel('Cost', fontsize=13)
a[2].set_xlabel('Net Loss (Barrels)', fontsize=13)
a[0].text(0.005, 1e8, "All Costs", fontsize=14, color='darkred')
a[1].text(0.005, 1e8, "Environmental Costs", fontsize=14, color='darkblue')
a[2].text(0.005, 1e8, "Emergency Costs", fontsize=14, color='darkgreen')


# The cost is certainly positively correlated with the net loss (less so the Emergency Costs).  I hesitate to fit these distributions because there is so much spread and because it definitely seems like there are several sub-populations in the data (for example, in the 'All Costs' plot in the top, there are outliers at low cost and at low loss.)

# This section is a work in progress...  more soon!

# In[ ]:




