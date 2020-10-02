#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Diagnosis Codes, Procedure Codes, BB USC codes, Oh my
# ================================================
# 
# This is a fairly rich data set.  In addition to the data leaks discussed in the forums, 
# there are scads of different things to look at if you can get sql to disgorge them in time.
# Our goal is to find groups that satisfy the following criterion:
# 
# 1. Relatively large group size.
# 1. Relatively low or high screener rate for the group, albeit preferably low.
# 1. Groups that are easily identified by only a few features
# 
# The first criterion is important for two reasons.  The first reason is number of people who benefit 
# from an increase in the screening rate of a group is directly proportional to the size of the 
# group. In turn this translates into more lives saved.  The second reason is that screener rates 
# estimated from larger groups are more likely to approximate the screener rate of the general 
# population meeting that criterion.  
# 
# The second criterion relates to cost and annoyance. The cost of reaching a specific group is 
# proportional to the size of the group whereas the benefit is proportional to the size of the 
# subset of non-screeners.  If a group has a high screening rate, then we are actually considering
# the portion of the population that doesn't fall into the group as our target.  In this case 
# recall that the screening rate for the target is not 1 - the screening rate of the group and in 
# fact unless the group is relatively large (think greater than 1% of the population), then the 
# screening rate of the target is close to that of the general population. As a final note for 
# this criterion we'd rather not potentially annoy screeners into changing their future behavior.
# 
# The third criterion comes from the fact that simpler models easier to implement and are 
# more likely to generalize.  In the particular case though we are looking into statistics and 
# not specific models.
# 
# CAVEAT:  The constraints of having to limit queries may have unintential side effects.

# In[ ]:


import numpy as np 
import pandas as pd 
import pylab as plt
import matplotlib as mpl
import sqlite3
from scipy import stats


# In[ ]:


# Any results you write to the current directory are saved as output.
con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# In[ ]:


db = sqlite3.connect('../input/database.sqlite')
dcode_rates = pd.read_sql_query("SELECT diagnosis_code, count(diagnosis_code) as cnt, avg(is_screener) as screener_rate FROM (patients_train as pt INNER JOIN (SELECT DISTINCT patient_id, diagnosis_code from diagnosis LIMIT 10000000) as ds ON pt.patient_id = ds.patient_id) GROUP BY diagnosis_code ORDER BY cnt;",db)


# In[ ]:


print(dcode_rates.shape)
print(dcode_rates.head(30))
print(dcode_rates.tail(30))


# In[ ]:


bones = dcode_rates.iloc[-800:].sort_values('screener_rate').copy()


# In[ ]:


print(bones.min(), "\n",bones.max())


# In[ ]:


grid = np.meshgrid(range(32),range(25))
bones['y'] = grid[1].flatten()
bones['x'] = grid[0].flatten()


# In[ ]:


ann_list = bones[(bones.cnt > 4500) & (bones.screener_rate > .68)]
print(ann_list)
ann_list2 = bones[(bones.cnt > 4500) & (bones.screener_rate < .44)]
print(ann_list2)


# In[ ]:


#fig = plt.figure(figsize=(17,12))
#ax = fig.add_subplot(111)
p1 = bones.plot(kind = 'scatter', x = 'x', y = 'y', c = 'screener_rate', s = bones['cnt']/50.0, cmap = 'RdBu', figsize=(17,12))
t1 = plt.ylabel(' ')
t1 = p1.yaxis.set_ticklabels([])
for dc,x,y in zip(ann_list['diagnosis_code'],ann_list['x'],ann_list['y']):
    #plt.annotate(dc, xy = (x, y), xytext = (-20 + 5*(27-y), 20 +20*(24-y) ),textcoords = 'offset points', ha = 'right', va = 'bottom',
    plt.annotate(dc, xy = (x, y), xytext = (-20 + 5*(27-y), 20 +20*(24-y) + 20*((x-1)%3)),textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
for dc,x,y in zip(ann_list2['diagnosis_code'],ann_list2['x'],ann_list2['y']):
    #plt.annotate(dc, xy = (x, y), xytext = (-20 + 5*y, -20 -20*(y+1)),textcoords = 'offset points', ha = 'right', va = 'bottom',
    plt.annotate(dc, xy = (x, y), xytext = (-20 + 5*y, -20 -20*(y+1) - 10*(x%3)),textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


# In[ ]:


ca_dcodes = pd.read_sql_query("SELECT ds.diagnosis_code, pt.patient_id FROM (diagnosis as ds INNER JOIN (SELECT * FROM patients_train WHERE patient_state = 'CA') as pt ON ds.patient_id = pt.patient_id);",db)


# In[ ]:


print(ca_dcodes.shape)
print(ca_dcodes.head())
parket = ca_dcodes['pt.patient_id'].value_counts()
print(parket)
print(parket.shape)


# In[ ]:


descript = pd.read_sql_query("SELECT ds.diagnosis_code, dc.diagnosis_description from (diagnosis as ds INNER JOIN diagnosis_code as dc ON ds.diagnosis_code = dc.diagnosis_code) LIMIT 100;",db)


# In[ ]:


print(descript)


# In[ ]:




