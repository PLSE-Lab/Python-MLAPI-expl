#!/usr/bin/env python
# coding: utf-8

# Data Analysis and visualizing 911 data!

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')


# In[ ]:


edf = pd.read_csv("../input/911.csv", header = 0,
                  names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
                  dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
                  parse_dates=['timeStamp'],date_parser=dateparse)
print(edf.shape)
edf.head()


# In[ ]:


def MakePregMap(df):
    from collections import defaultdict
    d = defaultdict(list)
    for index, title in df.title.iteritems():
        d[title].append(index)
    return d
title_index = MakePregMap(edf)
indexes = title_index['EMS: ELECTRICAL FIRE OUTSIDE']
print(edf.title[indexes])


# In[ ]:


#Some different method to group and sort the dataframe based on the number of occurence of title. 
print(edf.columns)
print(edf.title.value_counts(sort = True).head())
print("second")
import operator
hist = {}
for x in edf.title:
    hist[x] = hist.get(x,0) + 1
sorted_title_by_count = sorted(hist.items(), key=operator.itemgetter(1), reverse = True)
print(sorted_title_by_count[:5])
print("third")
from collections import Counter
sorted_title_by_count2 = Counter(edf.title).most_common()
print(sorted_title_by_count2[:5])
print("fourth")
sorted_grp = edf['title'].groupby(edf['title']).count().order(ascending = False)
print(sorted_grp[:5])


# In[ ]:


def itemFrequency(item, field):
    try:
        grp
    except NameError:
        grp = edf[field].groupby(edf[field]).count()
        return grp[item] 
    else:
        return grp[item]  
itemFrequency("Traffic: VEHICLE ACCIDENT -", 'title')     


# In[ ]:


sorted_grp.plot(kind = 'pie', figsize = (10,10),autopct = '%1.1f%%')


#  - Traffic accident is outlier as compare to other reason for 911 call. 
#  - Maximum call to 911 was made with title - Traffic: VEHICLE ACCIDENT
# 

# In[ ]:


def parseStringDtae(a):
    #from dateutil.parser import parse
    return a.date()

edf['call_date'] = edf['timeStamp'].apply(parseStringDtae)
No_call_to_911_by_date = edf['call_date'].groupby(edf.call_date).count().order(ascending = False)
print(No_call_to_911_by_date.head())
call_detail_on_012316 = edf[edf['call_date'] == datetime.date(2016,1, 23)]['title'].groupby(edf.title).count().order(ascending = False)


# In[ ]:


call_detail_on_012316.head()


# In[ ]:


def getCriteriaFromTitle(title):
    return title.split(':')[0]
edf['title_criteria'] = edf.title.apply(getCriteriaFromTitle)
title_criteria_grp = edf['title_criteria'].groupby(edf['title_criteria']).count()


# In[ ]:


title_criteria_grp.plot(kind = 'bar', color = 'red', title = "911 calls ditribution across various criteria:")


#  **So, the maximum number of call for EMS not Traffic related.**

# In[ ]:


Call_Traffic = edf[edf['title_criteria'] == 'Traffic']
edf.index = pd.DatetimeIndex(edf.timeStamp)


# In[ ]:


# Taken reference from Mike Chirico notebook
p = pd.pivot_table(Call_Traffic, values = 'e', index = ['timeStamp'], columns = ['title'], aggfunc = np.sum)

pp=p.resample('W', how=[np.sum]).reset_index()
pp.head()


# In[ ]:


# Red dot with Line

get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (12, 6)
fig, ax = plt.subplots()
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'k')
ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'ro')


ax.set_title("Traffic: VEHICLE ACCIDENT")
fig.autofmt_xdate()
plt.show()

# Note, you'll get a drop at the ends...not a complete week


# In[ ]:


# Remove the first and last row
pp = pp[pp['timeStamp'] < pp['timeStamp'].max()]
pp = pp[pp['timeStamp'] > pp['timeStamp'].min()]


# In[ ]:


pp['timeStamp'].max()


# In[ ]:


# Get the best fitting line

# Need to import for legend
import matplotlib.lines as mlines

# For best fit line
from sklearn import linear_model

# Red dot with Line
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



# Build Linear Fit
Y = pp['Traffic: VEHICLE ACCIDENT -'].values.reshape(-1,1)
X=np.arange(Y.shape[0]).reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(X,Y)
m = model.coef_[0][0]
c = model.intercept_[0]
ax.plot(pp['timeStamp'],model.predict(X), color='blue',
         linewidth=2)
blue_line = mlines.Line2D([], [], color='blue', label='Linear Fit: y = %2.2fx + %2.2f' % (m,c))
ax.legend(handles=[blue_line], loc='best')


ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'k')
ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'ro')


ax.set_title('Traffic: VEHICLE ACCIDENT')
fig.autofmt_xdate()
plt.show()


# In[ ]:


# Need to import for legend
import matplotlib.lines as mlines

fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  


ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 


ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'k')
ax.plot_date(pp['timeStamp'], pp['Traffic: VEHICLE ACCIDENT -'],'ro')


ax.plot_date(pp['timeStamp'], pp['Traffic: DISABLED VEHICLE -'],'g')
ax.plot_date(pp['timeStamp'], pp['Traffic: DISABLED VEHICLE -'],'bo')


ax.set_title("Traffic: VEHICLE ACCIDENT vs  Traffic: DISABLED VEHICLE")


# Legend Stuff
green_line = mlines.Line2D([], [], color='green', marker='o',markerfacecolor='blue',
                          markersize=7, label='Traffic: VEHICLE ACCIDENT')
black_line = mlines.Line2D([], [], color='black', marker='o',markerfacecolor='darkred',
                          markersize=7, label='Traffic: DISABLED VEHICLE')

ax.legend(handles=[green_line,black_line], loc='best')


fig.autofmt_xdate()
plt.show()

# Note scale hides the assault increase 


# In[ ]:




