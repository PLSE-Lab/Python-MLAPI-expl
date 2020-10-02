#!/usr/bin/env python
# coding: utf-8

# # trainView.csv
# 
# Most GPS coordinates are based on track telemetry.  GPS coordinates are subject to change 
# as track/signal work is being done on SEPTA lines.
# 
# 
# Columns:
# 
# * train_id
# * status
# * next_station
# * service
# * dest
# * lon
# * lat
# * source
# * track_change
# * track
# * date
# * timeStamp0  (timeStamp when entered area)
# * timeStamp1  (timeStamp when left area)
# 
# 
# Example of tracking a single train, **319**
# 
# 
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# You may want to define dtypes and parse_dates for timeStamps
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

d=pd.read_csv("../input/trainView.csv",
    header=0,names=['train_id','status','next_station','service','dest','lon',
                    'lat','source','track_change','track','date','timeStamp0',
                    'timeStamp1','seconds'],
    dtype={'train_id':str,'status':str,'next_station':str,'service':str,'dest':str,
    'lon':str,'lat':str,'source':str,'track_change':str,'track':str,'date':str,
    'timeStamp0':datetime.datetime,'timeStamp1':datetime.datetime,'seconds':str}, 
     parse_dates=['timeStamp0','timeStamp1'],date_parser=dateparse)







# In[ ]:


def getDeltaTime(x):
    r=(x[1] - x[0]).total_seconds() 
    return r

# It might make sense to add delta_s to the next version
d['delta_s']=d[['timeStamp0','timeStamp1']].apply(getDeltaTime, axis=1)


# In[ ]:


d.head()


# # Train 319
# 
# Train schedules can be found [here](http://www.septa.org/schedules/rail/).  Train 
# numbers are listed at the top of the schedule.  
# 
# Most of the time schedules match
# what happens in the system; however, it might be interesting to
# look for and identify exceptions.
# 
# > [SEPTA riders sound off on Regional Rail problems.](http://articles.philly.com/2016-01-15/news/69768873_1_septa-officials-ron-hopkins-regional-rail)
# 
# > [April 3, 2016. Train derailment impacts SEPTA.](https://en.wikipedia.org/wiki/2016_Chester,_Pennsylvania,_train_derailment)
# 
# > [April 8, 2016. Philadelphia region commemorates the 2016 NCAA Men's Basketball Champions.](http://www.septa.org/events/2016-villanova-parade.html) 
# 
# ![Image from github](https://raw.githubusercontent.com/mchirico/mchirico.github.io/master/p/images/septaScheduleSs.png)
# 
# 

# In[ ]:


# Train: 319
# Day:  2016-05-23

d[(d['train_id']=='319') & (d['date']=='2016-05-23')].sort_values(by='timeStamp0',ascending=True)

