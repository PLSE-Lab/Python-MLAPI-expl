#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
import operator
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/breakdownlist.csv', delimiter=';', index_col='id')


# In this article data obtained from various CNC machines' down times will be studied. The data set contains Machine name, person operating the machine, start/end of the down time period and total duration of machine down time in minutes. Multiple aspects of data will is included in this study.

# In[ ]:


df1= df.dropna(axis=0)


# In[ ]:


df.head()


# There are some entries with missing critical values therefore they are removed from the dataset. There are 299 entries prior to dropping missing values and 254 entries after removing them.

# In[ ]:


nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


nRow, nCol = df1.shape
print(f'After dropping rows with missing data, there are {nRow} rows and {nCol} columns')


# In[ ]:


cleaning_sum = df1.loc[df1['cause'] == 'Cleaning' , 'total'].sum()
breakdown_sum = df1.loc[df1['cause'] == 'breakdown' , 'total'].sum()
trouble_sum = df1.loc[df1['cause'] == 'trouble' , 'total'].sum()
other_sum = df1.loc[df1['cause'] == 'other' , 'total'].sum()


# In[ ]:


x1 = ['Cleaning','breakdown','trouble','other']
y1 = [cleaning_sum, breakdown_sum, trouble_sum, other_sum]


# ## Down time vs. Cause

# In the chart below machine down time is broken down into four categories and the amount of time for each category shown.

# In[ ]:


plt.bar(x1,y1, color='green')
plt.ylabel('down time (min)')
plt.xlabel('cause')
plt.title('down time vs. cause')


# Next, we are going to look at the how long each machine was down. We will look at top 10 values.
# Adding up the down time for each machine to get a total amount of time the machine was down. We are interested in breakdown as the cause only.

# In[ ]:


machine_log_breakdown={}
for index, row in df1.iterrows():  
    if row['cause'] == 'breakdown':
        if row['cncmachine'] in machine_log_breakdown:
            machine_log_breakdown[row['cncmachine']]=machine_log_breakdown[row['cncmachine']]+row['total']
        else:
            machine_log_breakdown[row['cncmachine']]=row['total']


# Next the values will be sorted from largest to smallest down time.

# In[ ]:


sorted_log_breakdown = sorted(machine_log_breakdown.items(), key=operator.itemgetter(1),reverse=True)


# In[ ]:


x2=[]
y2=[]
for i in range(0,10,1):
    x2.append(sorted_log_breakdown[i][0])
    y2.append(sorted_log_breakdown[i][1])


# ## breakdown time vs. machine name

# The graph below shows down time (minute) per machine. (10 highest down time values shown)

# In[ ]:


plt.bar(x2,y2, color='r')
plt.ylabel('breakdown time (min)')
plt.xlabel('machine name')
plt.title('down time vs. machine')


# In[ ]:


personnel_log={}
for index, row in df1.iterrows():  
    if row['personnelid'] in personnel_log:
        if row['cause'] == 'Cleaning':
            personnel_log[row['personnelid']][0]=personnel_log[row['personnelid']][0]+row['total']
        elif row['cause'] == 'breakdown':
            personnel_log[row['personnelid']][1]=personnel_log[row['personnelid']][1]+row['total']
        else:
            personnel_log[row['personnelid']][2]=personnel_log[row['personnelid']][2]+row['total']
    else:
        if row['cause'] == 'Cleaning':
            personnel_log[row['personnelid']]=[row['total'],0,0]
        elif row['cause'] == 'breakdown':
            personnel_log[row['personnelid']]=[0,row['total'],0]
        else:
            personnel_log[row['personnelid']]=[0,0,row['total']]


# In[ ]:


hour_sum={}
for key,value in personnel_log.items():
    print(key,':', value)
    hour_sum[key] = np.sum(value)


# ## down time vs. employee

# 10 employeers with the highest machine down time.

# In[ ]:


sorted_sum_hours = sorted(hour_sum.items(), key=operator.itemgetter(1),reverse=True)


# In[ ]:


for i in range(0,10,1):
    print(sorted_sum_hours[i][0],':', sorted_sum_hours[i][1])


# In[ ]:


x3=[]

y3_1=[]
y3_2=[]
y3_3=[]

for i in range(0,10,1):
    employee_id = sorted_sum_hours[i][0]
    
    # retrieving total Cleaning time for employee i
    y3_1.append(personnel_log[employee_id][0]) 
    # retrieving total breakdown time for employee i
    y3_2.append(personnel_log[employee_id][1])
    # retrieving total other time for employee i
    y3_3.append(personnel_log[employee_id][2])
    
    x3.append(str(sorted_sum_hours[i][0]))


# In[ ]:


plt.bar(x3,y3_1, color='b', label='Cleaning')
plt.bar(x3,y3_2, color='r', label='breakdown')
plt.bar(x3,y3_3, color='g',label='other')
plt.ylabel('breakdown time (min)')
plt.xlabel('personnel id')
plt.title('down time vs. employee id')
plt.legend()
plt.show()

