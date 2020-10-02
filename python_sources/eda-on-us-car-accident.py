#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# importing the librabries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
import missingno as msno # check missing value

# geographic visualization 
import chart_studio.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected= True)


# In[ ]:


df = pd.read_csv('../input/us-accidents/US_Accidents_May19.csv')


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


# checking null values 
def chk_null(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.shape[0]*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(10))


# In[ ]:


chk_null(df)


# In[ ]:


msno.matrix(df)


# here we can see that 'Precipitation', 'Wind_Chill' , 'End_Lat' , 'End_Lng' have higher missing values hence dropping those columns 

# In[ ]:


df.drop(['Precipitation(in)','Wind_Chill(F)','End_Lat','End_Lng'] ,axis =1,inplace = True)
df.shape


# In[ ]:


plt.figure(figsize=(10,7))
by_cat = df.groupby(["Source"]).size().sort_values(ascending = False)
sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")
plt.title("Data colleted by different surces")
plt.xlabel("Collection Count")


# most of datas are collected by MapQuest

# In[ ]:


plt.figure(figsize=(10,7))
by_cat = df.groupby(["Timezone"]).size().sort_values(ascending = False)
sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")
plt.title("accident count for different timezone")
plt.xlabel("Numer of acidents")


# from above plot we found that US/Eastern timezone has highest acccident count and US/Mountain has lowest accident count. 

# In[ ]:


plt.figure(figsize=(10,7))
by_cat = df.groupby(["Side"]).size().sort_values(ascending = False)
sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")
plt.title("accident count for different timezone")
plt.xlabel("Numer of acidents")


# means most of accident occured from right side. 

# **US state wise accident **

# In[ ]:


plt.figure(figsize=(7,10))
by_cat = df.groupby(['State']).size().sort_values(ascending = False)
sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")
plt.title("accident count for different State")
plt.xlabel("Numer of acidents")


# **Geographic accident Visualization using plotly **

# In[ ]:


# total number of accident grouped by US state 
acc_count = df.groupby('State')['State'].size()


# Now we need to begin to build our data dictionary. Easiest way to do this is to use the dict() function of the general form:
# 
# type = 'choropleth',
# 
# locations = list of states
# 
# locationmode = 'USA-states'
# 
# colorscale=
# 
# **Either a predefined string:**
# 
# 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
# 
# or create a custom colorscale
# 
# text= list or array of text to display per point
# 
# z= array of values on z axis (color of state)
# 
# colorbar = {'title':'Colorbar Title'})

# In[ ]:


data = dict(type = 'choropleth',
            locations = ["AL","AR","AZ","CA","CO","	CT","DC","DE","FL","GA","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI	","WV","WY"],
            locationmode = 'USA-states',
            colorscale= 'Electric',
            text= ["AL","AR","AZ","CA","CO","	CT","DC","DE","FL","GA","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI	","WV","WY"],
            z=acc_count,
            colorbar = {'title':'Accident_count'})
layout = dict(geo = {'scope':'usa'})
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)


# From above plot we found that CA state has highest number of accident.

# In[ ]:


#chekcing corelation etween various features 
plt.figure(figsize=(8,8))
corr = df.corr()
sns.heatmap(corr)


# In[ ]:


# Number of unique classes in each 'object' column
# Number of each type of column
df.dtypes.value_counts()


# In[ ]:


df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[ ]:


df['Start_Time'] = pd.to_datetime(df['Start_Time'], format="%Y/%m/%d %H:%M:%S")
df['DayOfWeekNum'] = df['Start_Time'].dt.dayofweek
df['DayOfWeek'] = df['Start_Time'].dt.weekday_name
df['MonthDayNum'] = df['Start_Time'].dt.day
df['HourOfDay'] = df['Start_Time'].dt.hour


# Severity Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay).

# In[ ]:


sev_count = df.groupby('Severity').size()


# **Severity vs Fraction of total accidents **

# In[ ]:


df.Severity.value_counts(normalize=True).sort_index().plot.bar()
plt.grid()
plt.title('Severity')
plt.xlabel('Severity')
plt.ylabel('Fraction');


# In[ ]:


sns.set_style('whitegrid')
ax = sns.pointplot(x="HourOfDay", y="TMC", hue="DayOfWeek", data=df)
ax.set_title('hoursoffday vs TMC(Traffic Message Channel) of accident')
plt.show()


# from above plot we can see that between 6 to 18  hours of the day below 212 TMC are generating for any accident. 
# 

# **weekdays vs fraction of accident **
# 
# from below plot we find that on weekend less amount of  accident occured 

# In[ ]:


weekday = df.groupby('DayOfWeek').ID.count()
weekday = weekday/weekday.sum()
dayOfWeek=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday[dayOfWeek].plot.bar()
plt.title('Acccidents by Weekday')
plt.xlabel('Weekday')
plt.ylabel('fraction of total accident');


# In[ ]:


st = pd.to_datetime(df.Start_Time, format='%Y-%m-%d %H:%M:%S')
end = pd.to_datetime(df.End_Time, format='%Y-%m-%d %H:%M:%S')


# In[ ]:


diff = (end-st)
top20 = diff.astype('timedelta64[m]').value_counts().nlargest(20)
print('top 20 accident durations correspond to {:.1f}% of the data'.format(top20.sum()*100/len(diff)))
(top20/top20.sum()).plot.bar(figsize=(7,5))
plt.title('Accident Duration [Minutes]')
plt.xlabel('Duration [minutes]')
plt.ylabel('Fraction');

