#!/usr/bin/env python
# coding: utf-8

# As climate change becomes more of a reality, we are bound to face its grave repercussions world-wide. Climate change is most readily associated with the melting of artic/antartic ice-caps, but it also leads to more antithetical phenomena, such as droughts; an example of the latter can be seen in Chennai, historically kwown as Madras--a major metropolis in southern India. This exercise explores Chennai's 'water histoty' over nearly the last 20 years--including the recent crisis. 
# 
# Chennai has four major reservoirs that supply water for the populus. The combined capacity of these 4 reservoirs is ~11057 mcft. The four major reservoirs are: POONDI, CHOLAVARAM, RED-HILLS, CHEMBARAMBAKKAM. In this exercise, we evaulate the water-levels and the rainfall-levels in these four reservoirs to evaluate the severity of the crisis. 

# In[ ]:


'''Import Modules'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot
from plotly import subplots as t
import plotly.graph_objs as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison);


# In[ ]:


'''Trace Function'''
def trace (res, color, name):
    trace = go.Scatter(x = res.index[::-1], y = res.values[::-1], name = name, marker = dict(color = color));
    return trace


# In[ ]:


'''Import Data'''
levelsdata = pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv");
rfdata = pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv");
levelsdata.set_index("Date");
rfdata.set_index("Date");
levelsdata.head()


# In[ ]:


rfdata.head()


# In[ ]:


'''Convert to DateTime and then Sort By Date'''
levelsdata["Date"] = pd.to_datetime(levelsdata["Date"], format = '%d-%m-%Y');
rfdata["Date"] = pd.to_datetime(rfdata["Date"], format = '%d-%m-%Y');

# Sort by Date
levelsdata.sort_values(by = 'Date', inplace = True);
rfdata.sort_values(by = 'Date', inplace = True);


# In[ ]:


'''Extract Water Level Data for Each of the Reservoir'''
poondi = levelsdata["POONDI"];
poondi.index = levelsdata["Date"];
chola = levelsdata["CHOLAVARAM"];
chola.index = levelsdata["Date"];
red = levelsdata["REDHILLS"];
red.index = levelsdata["Date"];
chem = levelsdata["CHEMBARAMBAKKAM"];
chem.index = levelsdata["Date"];
'''Draw Trace'''
poondit = trace(poondi, 'blue', 'Poondi');
cholat = trace(chola, 'orange', 'Cholavaram');
redt = trace(red, 'red', 'Redhills');
chemt = trace(chem, 'purple', ' Chembarambakkam');


# In[ ]:


'''Calculate the Total Water Level (sum of levels from each reservoir) on a Given Day'''
total = [];
for i in range(0, len(poondi), 1):
    sum  = 0;
    sum = poondi[i] + chola[i] + red[i] + chem[i]
    total.append(sum)
total = pd.Series(total);
total.index = levelsdata["Date"];
totalt = trace(total, 'royalblue', 'Total');
tl ={'Date': total.index, 'total': total.values}
totall2 = pd.DataFrame(data = tl);
totall2['year'] = totall2['Date'].map(lambda x: x.strftime('%Y'));
totall2['modate'] = totall2['Date'].map(lambda x: x.strftime('%m-%d'));
yearwltot = totall2[totall2['modate']=='12-31']; # Will Use Later for calulation of Consumption/Water Loss


# In[ ]:


'''Water Levels in Each Reservoir'''
levels = t.make_subplots(rows=1, cols=1);
# Reservoir Levels
levels.append_trace(poondit, 1, 1);
levels.append_trace(cholat, 1, 1);
levels.append_trace(redt, 1, 1);
levels.append_trace(chemt, 1, 1);
levels['layout'].update(height = 800, width = 1200, title = "Water Levels (mcft) in Chennai's Four Main Reservoirs", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'))
levels.show();


# As is evident in the plot above, the water levels in the four reservoirs vary annualy in a cyclic manner. This is likely a seasonal phenomena, due to annual patterns in the weather--i.e. The Monsoon. We can also see that the Cholavaram reservoir historically has lower water levels than the other three; this reservoir possibly is a smaller  than the others, or it is located in a region that doesn't see as much rainfall. The other three reservoirs follow similar patterns to one another, but the from the plot it seems as if the Redhills Reservoir isn't susceptible extremities like Poondi and Chembarambakkam. This can be seen in the timespan from 2013 to 2016, where the water levels in all  reservoirs varied vastly, but the Redhill Reservoir didn't see as drastic of variations as the others. 
# 
# We also see two periods where the water levels dropped noticeably; between 2012 and 2016, and more recently since 2017. Decreased water levels for a persistent amount of time can be alaraming, and may foreshadow a looming water-crisis.  

# In[ ]:


'''Total Water Levels in all four Reservoir'''
# Total Water Level Only
totfig = go.Figure();
totfig.add_trace(totalt);
totfig['layout'].update(height = 800, width = 1200, title = "Total Water Level (mcft) in Chennai Combined", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'));
totfig.show();


# Since the focus of this exercise is to study the water crisis in Chennai as a whole, it can be useful to look at the total water level in the four reservoirs. The most recent plot above, sums the water levels in the four reservoirs on a given day. The trends seen in the earlier graph--water levels in individual reservoirs--are mimicked in this one. The water levels vary annually. We also see the same two periods where the water level has decreased. 
# 
# In June 2019, the water level was nearly 0 mcft. Prior to that, it was 0 mcft in July 2017, and prior to that it was 0 mcft back in October 2004. Other than the two peaks in early and late 2015, the total water level since 2012 is visibly lower than in the years past; the water crisis in Chennai might have started earlier than previosly thought.

# In[ ]:


'''Average Water Level'''
mltot = total.resample('Y').mean();
meanwl = go.Figure([go.Bar(x = mltot.index[::-1], y = mltot.values[::-1], marker_color = 'midnightblue', name = 'Average Annual Water Level')]);
meanwl['layout'].update(height = 800, width = 1200, title = "Average Daily Water Level (mcft))", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'));
meanwl.show();


# In the graph, we plot the daily  average water level. The average is of the total water level in all four water reservoirs combined. We see that the average water level visibly drops post 2012, and even further in 2017 and 2019. This provides further evidence for a looming water crisis in Chennai. Next, lets look at the recent water shortage more closely. 

# In[ ]:


# Post-2018
dip2018 = (levelsdata["Date"] > '2018-01-01');
dip2018levels = levelsdata.loc[dip2018];

poondi = dip2018levels["POONDI"];
poondi.index = dip2018levels["Date"];
chola = dip2018levels["CHOLAVARAM"];
chola.index = dip2018levels["Date"];
red = dip2018levels["REDHILLS"];
red.index = dip2018levels["Date"];
chem = dip2018levels["CHEMBARAMBAKKAM"];
chem.index = dip2018levels["Date"];

poondit = trace(poondi, 'blue', 'Poondi');
cholat = trace(chola, 'orange', 'Cholavaram');
redt = trace(red, 'red', 'Redhills');
chemt = trace(chem, 'purple', ' Chembarambakkam');

totalpo2018 = [];
for i in range(0, len(poondi), 1):
    sum  = 0;
    sum = poondi[i] + chola[i] + red[i] + chem[i]
    totalpo2018.append(sum)
totalpo2018 = pd.Series(totalpo2018);
totalpo2018.index = dip2018levels["Date"];
totalt = trace(totalpo2018, 'royalblue', 'Total');

levels = t.make_subplots(rows=1, cols=1);
levels.append_trace(poondit, 1, 1);
levels.append_trace(cholat, 1, 1);
levels.append_trace(redt, 1, 1);
levels.append_trace(chemt, 1, 1);
levels['layout'].update(height = 800, width = 1200, title = "Water Levels (mcft) in Chennai's Four Main Reservoirs Since 2018", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'))
levels.show();


# Total Water Level Only
totfig = go.Figure();
totfig.add_trace(totalt);
totfig['layout'].update(height = 800, width = 1200, title = "Total Water Level (mcft) in Chennai Combined Since 2018", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'));
totfig.show();


# Starting in mid-2018, the water levels in all four reservoirs started to drop. While Poondi and Redhills Water Level bounced back towards late-2018, levels in Cholavaram and Chembarambakkam did not. The severeity of the decrease in water level is more evident in the second plot. The total water level in Chennai in March 2018 was approximately 4500-5000 mcft; the total water level in Chennai in March 2019 was approximately 700-1000 mcft. That's nearly an 80% drop in water levels in a span of 1 year. At this rate, without a strong monsoon season in 2019, Chennai might run out of water in 2020.
# 
# We have looked at the water levels in individual reservoirs, as well as in total. Now, lets examine rainfall in Chennai to further assess the severity of the water crisis. 

# In[ ]:


# Rain Fall
poondirain = rfdata["POONDI"];
poondirain.index = rfdata["Date"];
cholarain = rfdata["CHOLAVARAM"];
cholarain.index = rfdata["Date"];
redrain = rfdata["REDHILLS"];
redrain.index = rfdata["Date"];
chemrain = rfdata["CHEMBARAMBAKKAM"];
chemrain.index = rfdata["Date"];

poondiraint = trace(poondirain, 'blue', 'Poondi');
cholaraint = trace(cholarain, 'orange', 'Cholavaram');
redraint = trace(redrain, 'red', 'Redhills');
chemraint = trace(chemrain, 'purple', ' Chembarambakkam');

totalrain = [];
for i in range(0, len(poondirain), 1):
    sum  = 0;
    sum = poondirain[i] + cholarain[i] + redrain[i] + chemrain[i]
    totalrain.append(sum)
totalrain = pd.Series(totalrain);
totalrain.index = rfdata["Date"];
tr ={'Date': totalrain.index, 'total': totalrain.values}
total2 = pd.DataFrame(data = tr);
total2['year'] = total2['Date'].map(lambda x: x.strftime('%Y'));
yearmax = totalrain.resample('Y').sum();
yearmean = totalrain.resample('Y').mean();
yrt = {'Date':yearmax.index, 'total': yearmax.values};
yearraintot = pd.DataFrame(data = yrt);
yearraintot['year'] = yearraintot['Date'].map(lambda x: x.strftime('%Y'));
yearraintot['modate'] = yearraintot['Date'].map(lambda x: x.strftime('%m-%d'));


# Total Rainfall
totfig = go.Figure();
totfig.add_trace(go.Scatter(x = totalrain.index[::-1], y = totalrain[::-1], mode = 'markers', name = 'Total Rainfall (mm)'));
totfig['layout'].update(height = 800, width = 1200, title = "Total Rainfall (mm) in Chennai Combined", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Rainfall (mm)'));
totfig.show();


# The scatter plot shows the daily rainfall in Chennai--in all four reservoirs--from 2004 to 2019. Clusters of elevated points follow a cyclic trend; these probably correspond to the annual monsoon season. From this plot, it doesn't seem if the rainfall in recent years is less than in years past. Next, lets take a look at the total and average annual rainfall. 

# In[ ]:


# Total Annual Rainfall
totalrain = go.Figure([go.Bar(x = yearmax.index[::-1], y = yearmax.values[::-1], marker_color = 'red', name = 'Total Annual Rain Fall')]);
totalrain['layout'].update(height = 800, width = 1200, title = "Total Annual Rainfall (mm)", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Rainfall (mm)'));
totalrain.show();


# The plot above presents the total annual rainfall from 2004 to 2019. Apart from 2015 and 2017, the total annual rainfall since 2012 is visibly less than in previous years; 2018 had the minimum total annual rainfall in the data set. This decrease in rainfall could possibly have contributed to the lower levels of water in the reservoirs during the same time period. The next graph is the total annual rainfall averaged over 365 days; it is essentially identical to the total annual rainfall plot. 

# In[ ]:


# Average Annual Rainfall
meanrain = go.Figure([go.Bar(x = yearmean.index[::-1], y = yearmean.values[::-1], marker_color = 'midnightblue', name = 'Average Annual Rainfall')]);
meanrain['layout'].update(height = 800, width = 1200, title = "Average Annual Rainfall (mm)", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Rainfall (mm)'));
meanrain.show();


# While looking at the total annual rainfall plot, we asserted that the visibly lower levels of rainfall could have contributed to the lower levels of water in the reservoirs in recent years. To test if this is accurate, we conduct a One-Way ANOVA and a Multi-Comparison; these tests will tell us if the total annual rainfall in s given year is significantly different than in other years. 

# In[ ]:


# 1-Way ANOVA to see if there is a difference in Average Annual Rain Fall over the time-course of the data
md = ols('total ~ year', data = total2).fit();
anocomp = sm.stats.anova_lm(md, typ =2);
print(anocomp);


# The One-Way ANOVA takes group values, and determines if at least one is significantly different from the others. A 'p-value' of 0.000002 is extremely small; this event, meaning the differences in group values, is very unlikely to happen solely by chance. The total/mean annual rainfall values differ siginificantly from others for at least one year.
# 
# Now, lets take a look at Multi-Comparison to see which years differ significantly. 

# In[ ]:


# Multi-Comparison to determine which year differs significantly
tuk = MultiComparison(total2['total'], total2['year']);
print(tuk.tukeyhsd())


# The Multi-Comparison table indicates that lower levels of water levels seen in 2012, 2013, 2014, 2016, 2018 do not differ significanly from most years, but only from 2005 and 2017, which in fact had unusally high levels of rainfall. This suggests that the lower levels of rainfall cannot be the only reason for the lower levels of water in Chennai's reservoirs. Though, I suspect the lower water levels might be a compounding effect of decreasing annual rainfall, as well as an increase in water loss/consumption. Next, lets define and study consumption/loss. 
# 
# Water loss/consumption is defined as the Water Level on (Dec-31 in Year 0 + Total Rainfall in Year 1 - Water Level on Dec-31 in Year 1)
# 

# In[ ]:


wl = yearwltot['total'].tolist();
rf = yearraintot['total'].tolist();
con = [];
year = [];
y1 = 2005;
for i in range(0, len(wl) - 1, 1):
    c = (wl[i] + rf[i+1]) - wl[i+1];
    con.append(c);
    year.append(y1);
    y1 += 1;

cr = {'year':year, 'consumption':con};
cr = pd.DataFrame(data = cr);
consumption = go.Figure([go.Bar(x = year, y = con, marker_color = 'red', name = 'Annual Water Consumtion')]);
consumption['layout'].update(height = 800, width = 1200, title = "Annual Water Consumption", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Consumption'));
consumption.show();


# Apart from 2016, it does not seem as if water loss/consumpion has increased; in fact, in 2017, the water loss/consumption was lower than in most years. Earlier we saw that in 2018, the total annual rainfall was lower in comparison to years past--in fact it had the lowest rainfall in the given dataset. In the consumption chart above, you can see that in 2018 the water loss/usage was nearly consistent with many years that had ample rainfall. Exhausting the water supply without proper replenishment in form of rainfall can have contributed to the recent crisis.

# In this exercise--mainly a visualization one--we have seen that the water levels in Chennai's major reservoirs have recently been dropping, annual rainfall has also been decreasing, but the water loss/consumption isn't following a similar trend. Though this exercise does not predict the scope of the 2019 monsoon season, it does help suggest to limit water usage in an effort to prevent exacerbating a water shortage crisis in a city of nearly 7 million.
