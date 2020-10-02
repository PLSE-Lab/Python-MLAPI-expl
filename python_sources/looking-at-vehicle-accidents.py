#!/usr/bin/env python
# coding: utf-8

# ## Vehicle Accident Patterns ##
# 
# This notebook looks at accidents in Montgomery County PA., and compares the data  to two neighboring townships: Cheltenham and Abington.  Both Cheltenham and Abington have SEPTA (local train transportation) stops within their borders.
# 
# It's possible that most accidents happen when there is a higher volume of vehicles on the road - morning and evening rush hour, plus possible lunch time travel.  Traffic volume would probably decrease during vacation summer months - July and August.  
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

# Read data 
d=pd.read_csv("../input/911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2016-01-01 00:00:00")]


# In[ ]:


d.head()


# In[ ]:


# Just take Traffic Vehicle Accidents...
v=d[(d.title == 'Traffic: VEHICLE ACCIDENT -')]
v.head()


# In[ ]:


p=pd.pivot_table(v, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling every week 'W'.  This is very powerful
pp=p.resample('W', how=[np.sum]).reset_index()
pp.sort_values(by='timeStamp',ascending=False,inplace=True)

# Let's flatten the columns 
pp.columns = pp.columns.get_level_values(0)

# Show values
# Note, last week might not be a full week
pp.tail(3)


# In[ ]:


# Drop the last week
pp=pp[(pp['timeStamp'] != pp['timeStamp'].max())]
pp.count()


# In[ ]:


# Plot this out
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12,rotation=45,ha='left')




ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'k')



ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nAll of Montco. PA. /week")
#fig.autofmt_xdate()
plt.show()


# Not really seeing much of a pattern.  There is a perceptible, linear, increase from April to late June.  But, then is bounces around.  There also could be an increase after the 2016 summer months.

# In[ ]:


p=pd.pivot_table(v, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling every month 'M'.  
pp=p.resample('M', how=[np.sum]).reset_index()
pp.sort_values(by='timeStamp',ascending=False,inplace=True)

# Let's flatten the columns 
pp.columns = pp.columns.get_level_values(0)

# Show a few values
# Note, last and first readings might not be a full capture
pp.head(-1)


# Above, obviously the last month isn't always going to contain a complete set of data.
# 

# In[ ]:


# Plot this out
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12,rotation=45,ha='left')




ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'k')



ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nAll of Montco. PA. /month")
#fig.autofmt_xdate()
plt.show()


# Interesting.  This smooths out the data and seems to show a slight increase.  

# In[ ]:





# In[ ]:


# We're taking out the last month
pp2 = pp[(pp['timeStamp'] != pp['timeStamp'].max())]
# Plot this out
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12,rotation=45,ha='left') 

ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'k')

ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nAll of Montco. PA. /month")

#fig.autofmt_xdate()
plt.show()


# In[ ]:


pp2.head(10)


# ## Cheltenham ##
# 
# Let's look at one Township

# In[ ]:


c=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'CHELTENHAM')]
c.head()


# In[ ]:


# Create pivot
p=pd.pivot_table(c, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling every month 'M'.  
pp=p.resample('M', how=[np.sum]).reset_index()
pp.sort_values(by='timeStamp',ascending=False,inplace=True)

# Let's flatten the columns 
pp.columns = pp.columns.get_level_values(0)


# Show a few values
pp.head(10)


# In[ ]:


# We're taking out the last month
pp2 = pp[(pp['timeStamp'] != pp['timeStamp'].max())]
# Plot this out
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12,rotation=45,ha='left') 

ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'k')
ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'ro')

ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nCheltenham /month")
#fig.autofmt_xdate()
plt.show()


# ## Abington ##

# In[ ]:


a=d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'ABINGTON')]
a.head()


# In[ ]:


# Create pivot
pa=pd.pivot_table(a, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling every month 'M'.  
ppA=pa.resample('M', how=[np.sum]).reset_index()
ppA.sort_values(by='timeStamp',ascending=False,inplace=True)

# Let's flatten the columns 
ppA.columns = ppA.columns.get_level_values(0)


# Show a few values
ppA.head(3)


# In[ ]:


# We're taking out the last month
ppA2 = ppA[(ppA['timeStamp'] != ppA['timeStamp'].max())]
# Plot this out
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12,rotation=45,ha='left')

# Abington
ax.plot_date(ppA2['timeStamp'], ppA2['Traffic: VEHICLE ACCIDENT -'],'g')
ax.plot_date(ppA2['timeStamp'], ppA2['Traffic: VEHICLE ACCIDENT -'],'go')

# Cheltenham
ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'r')
ax.plot_date(pp2['timeStamp'], pp2['Traffic: VEHICLE ACCIDENT -'],'ro')


ax.set_title("Traffic: VEHICLE ACCIDENT -"+"\nAbington /month (Green)"+
            "\nCheltenham /month (Red)")
#fig.autofmt_xdate()
plt.show()


# ## Seaborn Heat Map -- Look at times ##

# In[ ]:


# Vehicle Accident 
# Put this in a variable 'g'
g = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'CHELTENHAM')]
g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))
g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))
p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)

# Check data if you want
p.head()

cmap = sns.cubehelix_palette(light=2, as_cmap=True)
ax = sns.heatmap(p,cmap = cmap)
ax.set_title('Vehicle  Accidents - Cheltenham Townships ');


# In[ ]:


# Vehicle Accident 
# Put this in a variable 'g'
g = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'ABINGTON')]
g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))
g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))
p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)

# Check data if you want
p.head()

cmap = sns.cubehelix_palette(light=2, as_cmap=True)
ax = sns.heatmap(p,cmap = cmap)
ax.set_title('Vehicle  Accidents - Abington Townships ');


# Wow...odd that July 1700 hour is so high...  Is this unique to these two Townships?

# In[ ]:


# Vehicle Accident 
# Put this in a variable 'g'
g = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp != 'ABINGTON') & (d.twp != 'CHELTENHAM')         ]
g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))
g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))
p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)

# Check data if you want
p.head()

cmap = sns.cubehelix_palette(light=2, as_cmap=True)
ax = sns.heatmap(p,cmap = cmap)
ax.set_title('Vehicle  Accidents - Montco Townships\n (Except Abington + Cheltenham) ' );


# So why would July be so bad for Abington and Cheltenham.  What goes on at 1700?  Work related rush-hour.  Also, these two Townships rely on SEPTA.
# 
# 
# Reference:
# SEPTA media release
# http://www.septa.org/media/releases/2016/7-3-16.html
# 
# 
# 
# [SEPTA - OTP on Kaggle][1]
# 
# 
# 
#   [1]: https://www.kaggle.com/septa/on-time-performance

# ## Cheltenham - July ##
# 
# Let's take a closer look at the data for Cheltenham

# In[ ]:


c = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'CHELTENHAM') ]
c['Month'] = c['timeStamp'].apply(lambda x: x.strftime('%m %B'))
c['Hour'] = c['timeStamp'].apply(lambda x: x.strftime('%H'))


c = c[(c.Month == '07 July') & (c.Hour == '17')]
c.head()


# In[ ]:


c['zip'].value_counts()
# Interesting... 19027 is Elkins Park. A SEPTA train stop


# You can interact (zoom, pan) with a map of these accidents [here][1].
# 
# ![Map of Accidents][2]
# 
# 
#   [1]: https://www.kaggle.com/mchirico/d/mchirico/montcoalert/cheltenham-vehicle-accidents-july-2016
#   [2]: https://raw.githubusercontent.com/mchirico/mchirico.github.io/master/p/images/accidentCheltenhamJuly2016.png

# The map above could be showing traffic choke points.  There doesn't appear to be any way to figure out whether this is traffic influenced by SEPTA changes.  

# ## Abington - July ##

# In[ ]:


a = d[(d.title == 'Traffic: VEHICLE ACCIDENT -') & (d.twp == 'ABINGTON') ]
a['Month'] = a['timeStamp'].apply(lambda x: x.strftime('%m %B'))
a['Hour'] = a['timeStamp'].apply(lambda x: x.strftime('%H'))


a = a[(a.Month == '07 July') & (a.Hour == '17')]
a.head()


# In[ ]:


a['zip'].value_counts()


# You can pan and zoom on Google maps [here][1].  
# 
# 
# ![Abington Accidents][2]
# 
# 
#   [1]: https://www.kaggle.com/mchirico/d/mchirico/montcoalert/abington-vehicle-accidents-july-2016
#   [2]: https://raw.githubusercontent.com/mchirico/mchirico.github.io/master/p/images/accidentAbingtonJuly2016.png

# In[ ]:




