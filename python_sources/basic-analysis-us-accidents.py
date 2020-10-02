#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <font size=5>Welcome to my quick analysis of the US Accidents data!</font>
# <br><br>First, let's read the data using Pandas. Then we can check what exactly we're to be working with.

# In[ ]:


data = pd.read_csv(r'/kaggle/input/us-accidents/US_Accidents_May19.csv')


# In[ ]:


data.describe()


# In[ ]:


data[:10]


# In[ ]:


data.columns


# <font size=3>Now we can start looking for relations between the data. For example, let's take a look at the amount of accidents per state.</font><br><br> To do that (I think) we have to prepare our data, which we can do with a simple **for** loop.

# In[ ]:


states = data.State.unique()


# In[ ]:


count_by_state=[]
for i in data.State.unique():
    count_by_state.append(data[data['State']==i].count()['ID'])
    


# <font size=3>In the loop above we also used the **count** function to get the amount.

# In[ ]:


plt.figure(figsize=(16,10))
sns.barplot(states, count_by_state)


# We can clearly see that California had had the biggest amount of accidents.
# <br>Next, let's check if severity affects the way the amounts align.

# In[ ]:


severity_1_by_state = []
severity_2_by_state = []
severity_3_by_state = []
severity_4_by_state = []
for i in states:
    severity_1_by_state.append(data[(data['Severity']==1)&(data['State']==i)].count()['ID'])
    severity_2_by_state.append(data[(data['Severity']==2)&(data['State']==i)].count()['ID'])
    severity_3_by_state.append(data[(data['Severity']==3)&(data['State']==i)].count()['ID'])
    severity_4_by_state.append(data[(data['Severity']==4)&(data['State']==i)].count()['ID'])


# The graph below shows us how different severity levels compare.

# In[ ]:


plt.figure(figsize=(20,15))

plt.bar(states, severity_2_by_state, label='Severity 2')
plt.bar(states, severity_3_by_state, label='Severity 3')
plt.bar(states, severity_4_by_state, label='Severity 4')
plt.bar(states, severity_1_by_state, label='Severity 1')


plt.legend()


# <font size=3>Although we plotted four different features, in most cases only three of them appear in our graph.<font>
#     <br>This shows us, that accidents of severity 1 barely ever occur. Or are barely ever reported.

# In[ ]:


data.TMC.unique()


# In[ ]:


TMC_counts=data.TMC.value_counts()
plt.figure(figsize=(16, 10))

ax=sns.barplot(TMC_counts.index, TMC_counts)
ax.set(xlabel='TMC type', ylabel='Amount')


# The graph above shows that the most common type of TMC is 201.0, which according to the event code list translates to "(Q) accident(s)".

# In[ ]:


Temperature = data['Temperature(F)']
Severity_1_data = data[data['Severity']==1]['Temperature(F)'].mean()
Severity_2_data = data[data['Severity']==2]['Temperature(F)'].mean()
Severity_3_data = data[data['Severity']==3]['Temperature(F)'].mean()
Severity_4_data = data[data['Severity']==4]['Temperature(F)'].mean()
Severity_labels = ['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4']

Mean_temp_by_severity = [Severity_1_data, Severity_2_data, Severity_3_data, Severity_4_data]


# Is temperature a key factor increasing the severity of an accident?

# In[ ]:


plt.figure(figsize=(16, 6))
sns.barplot(Severity_labels, Mean_temp_by_severity)
plt.grid(color='black', linestyle='-', linewidth=1, alpha=0.3)


# While the differences are small, there is **some** corelation between temperature and severity.
# <br>What about the weather conditions?

# In[ ]:


Weather = data.Weather_Condition.value_counts()


# In[ ]:


plt.figure(figsize=(16, 16))
sns.barplot(Weather.values, Weather.index)


# In[ ]:


severity_1_by_Weather = []
severity_2_by_Weather = []
severity_3_by_Weather = []
severity_4_by_Weather = []
for i in Weather.index:
    severity_1_by_Weather.append(data[(data['Severity']==1)&(data['Weather_Condition']==i)].count()['ID'])
    severity_2_by_Weather.append(data[(data['Severity']==2)&(data['Weather_Condition']==i)].count()['ID'])
    severity_3_by_Weather.append(data[(data['Severity']==3)&(data['Weather_Condition']==i)].count()['ID'])
    severity_4_by_Weather.append(data[(data['Severity']==4)&(data['Weather_Condition']==i)].count()['ID'])


# In[ ]:


plt.figure(figsize=(80, 20))

plt.bar(Weather.index, severity_2_by_Weather, label='Severity 2')
plt.bar(Weather.index, severity_3_by_Weather, label='Severity 3')
plt.bar(Weather.index, severity_4_by_Weather, label='Severity 4')
plt.bar(Weather.index, severity_1_by_Weather, label='Severity 1')
plt.legend()


# We can observe, that specific weather conditions have slight impact on the severity of an accident.
# <br>Let's mute the first rows that are clearly the most frequent conditions and have one more look at less frequent weather conditions.

# In[ ]:


plt.figure(figsize=(80, 20))

plt.bar(Weather.index[9:], severity_2_by_Weather[9:], label='Severity 2')
plt.bar(Weather.index[9:], severity_3_by_Weather[9:], label='Severity 3')
plt.bar(Weather.index[9:], severity_4_by_Weather[9:], label='Severity 4')
plt.bar(Weather.index[9:], severity_1_by_Weather[9:], label='Severity 1')
plt.legend()


# In[ ]:


plt.figure(figsize=(80, 20))

plt.bar(Weather.index[27:], severity_2_by_Weather[27:], label='Severity 2')
plt.bar(Weather.index[27:], severity_3_by_Weather[27:], label='Severity 3')
plt.bar(Weather.index[27:], severity_4_by_Weather[27:], label='Severity 4')
plt.bar(Weather.index[27:], severity_1_by_Weather[27:], label='Severity 1')
plt.legend()


# What were the weather conditions of the accidents where severity is the highest?

# In[ ]:


percentage_severity_1 = []
percentage_severity_2 = []
percentage_severity_3 = []
percentage_severity_4 = []
for i in range(len(severity_1_by_Weather)):
    percentage_severity_1.append((severity_1_by_Weather[i]/Weather[i])*100)
    percentage_severity_2.append((severity_2_by_Weather[i]/Weather[i])*100)
    percentage_severity_3.append((severity_3_by_Weather[i]/Weather[i])*100)
    percentage_severity_4.append((severity_4_by_Weather[i]/Weather[i])*100)


# In[ ]:


percentage_severity_3[1]+percentage_severity_2[1]+percentage_severity_1[1]+percentage_severity_4[1]


# In[ ]:


plt.figure(figsize=(80, 20))

plt.bar(Weather.index, percentage_severity_2, label='Severity 2')
plt.bar(Weather.index, percentage_severity_3, label='Severity 3')
plt.bar(Weather.index, percentage_severity_4, label='Severity 4')
plt.bar(Weather.index, percentage_severity_1, label='Severity 1')
plt.legend()


# It seems that accidents of highest severity occur during weather conditions such as **thunderstorm, heavy rain, snow (in general) and blowing sand**, the worst conditions being **light blowing snow** and **heavy thunderstorms and snow**.
