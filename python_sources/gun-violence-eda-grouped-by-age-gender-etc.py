#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True) #do not miss this line
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from wordcloud import WordCloud, STOPWORDS 

import cufflinks as cf
cf.go_offline()

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv("..//input/gun-violence-data_01-2013_03-2018.csv")


# In[3]:


df.info()


# In[4]:


df.head(5)


# In[5]:


df.drop(['incident_id', 'incident_url', 'source_url', 'incident_url_fields_missing', 'sources','participant_name','address'], axis=1, inplace=True)


# Create Year, Month, Day and Day of Week from Date

# In[6]:


df = pd.concat([pd.DataFrame([each for each in df['date'].str.split('-').values.tolist()],
                             columns=['year', 'month', 'day']),df],axis=1)
df['day_of_week'] = df.date.apply(lambda x: pd.to_datetime(x).weekday())


# Number of incidents per year is plotted below. It is evident that the number of incidents reported over the years are growing. We will ignore 2013 and 2018,

# In[7]:


df.groupby(['year']).size().iplot(kind='bar',title = 'Number of Incidents', xTitle = 'Year', yTitle='Count')


# In[8]:


df_mod = df[(df.year == '2014') | (df.year == '2015') | (df.year == '2016') | (df.year == '2017')]


# Box plot solidifies our previous observation. In 2014, median incidents per month was around 4.35K while in 2017, it has increased to 5.18K. Similar increase in observable across all quantiles.

# In[9]:


pd.crosstab(df_mod.month, df_mod.year).iplot(kind='box', title = 'Incidents', xTitle ='Year',
                                                                    yTitle = 'Number')


# [](http://)We will visualize the incident  trend for individual months by plotting the quantiles.

# 2014
# * Quarter 3 is the busiest
# * In September, 75% of the days had at least 155 or more incidents, 25% had 182 or more
# * February seems to be the quietest with 75% days reporting 94 or below incidents
# 
# 2015
# * 75% days in May had incidents more than 151, while sametime previous year only 25% days had incidents more than 151.
# 
# 2016
# * August is busiest with 75% of days reporting 165 or more incidents
# 
# 2017
# * 75% of the days in April had more than 172 incidents

# In[10]:


pd.merge((pd.crosstab([df_mod.year,df_mod.month], df_mod.day).T.quantile(.25).loc['2014',:].reset_index().iloc[:,1:3].set_index('month')),
         (pd.crosstab([df_mod.year,df_mod.month], df_mod.day).T.quantile(.75).loc['2014',:].reset_index().iloc[:,1:3].set_index('month')),
         left_index=True, right_index=True).iplot(title = 'Incidents -2014', xTitle ='Month',
                                                                    yTitle = 'Number')


# In[11]:


pd.merge((pd.crosstab([df_mod.year,df_mod.month], df_mod.day).T.quantile(.25).loc['2015',:].reset_index().iloc[:,1:3].set_index('month')),
         (pd.crosstab([df_mod.year,df_mod.month], df_mod.day).T.quantile(.75).loc['2015',:].reset_index().iloc[:,1:3].set_index('month')),
         left_index=True, right_index=True).iplot(title = 'Incidents -2015', xTitle ='Month',
                                                                    yTitle = 'Number')


# In[12]:


pd.merge((pd.crosstab([df_mod.year,df_mod.month], df_mod.day).T.quantile(.25).loc['2016',:].reset_index().iloc[:,1:3].set_index('month')),
         (pd.crosstab([df_mod.year,df_mod.month], df_mod.day).T.quantile(.75).loc['2016',:].reset_index().iloc[:,1:3].set_index('month')),
         left_index=True, right_index=True).iplot(title = 'Incidents -2016', xTitle ='Month',
                                                                    yTitle = 'Number')


# In[13]:


pd.merge((pd.crosstab([df_mod.year,df_mod.month], df_mod.day).T.quantile(.25).loc['2017',:].reset_index().iloc[:,1:3].set_index('month')),
         (pd.crosstab([df_mod.year,df_mod.month], df_mod.day).T.quantile(.75).loc['2017',:].reset_index().iloc[:,1:3].set_index('month')),
         left_index=True, right_index=True).iplot(title = 'Incidents -2017', xTitle ='Month',
                                                                    yTitle = 'Number')


# In[14]:


pd.crosstab(df_mod.month, df_mod.day).T.sum().iplot(kind='bar', xTitle= 'Month', yTitle= "Number", 
                                                    title = 'Total incidents grouped by month')


# January seems to be reporting more incidents which we didn't notice in previous analysis, these jumps in count must have been caused by outlier. We will check this a bit more closely

# In[15]:


months_df = df_mod.groupby(['year', 'month', 'day']).size().reset_index()
months_df.drop('year', axis=1, inplace=True)


# The outliers in January are clearly visible. On a particular day in January there were almost 350 incidents! While the median is far below that 

# In[16]:


ax1 = months_df[months_df['month'] == '01'][['day', 0]].plot(kind='box', title = "Incid January")
ax1.set_ylabel("count")
ax1.set_xlabel("January")


# In[17]:


months_df.iloc[months_df[months_df['month'] == '01'][[0]].idxmax()]


# It is evident that weekends reports more number on incidents

# In[18]:


ax1 = pd.crosstab([df_mod.year,df_mod.month], df_mod.day_of_week).reset_index().drop('year',axis=1).plot(kind='box')
ax1.set_xticklabels(['Mon', 'Tue','Wed', 'Thu', 'Fri', 'Sat', 'Sun'])


# Now We will create a helper function to read and tabularize the fields which has a pattern ?::X||. Once this table is created, we would be able to use it to match the fields in the same row and create meaningful observations from it. For example among the victims, who belongs to which age group?

# In[19]:


#Helper function to extract and tabulate text which is of pattern ?::X||
def df_return(lst):
    dic = {}
    ls = []
    for each in lst:
         ls.append(re.findall(r"[^:|]+", str(each)))
    
    for i, each in enumerate(ls):
        if each == ['nan']:
            dic[i] = {0: 'nan'}
        else:
            dic[i] = dict(key_val for key_val in zip(*([iter(each)] * 2)))
    return pd.DataFrame.from_dict({(i,j): dic[i][j] for i in dic.keys() for j in dic[i].keys()},orient='index')    


# In[20]:


df_gun_typ = df_return(df['gun_type']) 
df_gun_typ.index = pd.MultiIndex.from_tuples(df_gun_typ.index)


# Most frequently used gun, excluding NaN and Unknowns, is handgun

# In[21]:


df_gun_typ[0].value_counts()[2:].iplot(kind='bar', xTitle= 'Type of Gun', yTitle= "Number", title = 'Type of Gun Used ordered by Frequency') # nan and unknow omitted


# In[22]:


df_gun_stolen = df_return(df['gun_stolen']) 
df_gun_stolen.index = pd.MultiIndex.from_tuples(df_gun_stolen.index)
df_gun = pd.merge(df_gun_typ, df_gun_stolen, left_index=True, right_index=True, how='left')
df_gun.columns = ['type', 'status']
gun_pd = pd.crosstab(df_gun.type, df_gun.status)
gun_pd.drop('nan', inplace=True)
gun_pd.drop('nan', axis= 1, inplace=True)
gun_pd.drop('Unknown', inplace=True)
gun_pd.drop('Unknown',axis=1, inplace=True)


# Since handgun is most frequently used, obviously it is the most stolen,, but the intersting fact is rifles are stolen more times than 9mms.

# In[23]:


layout = dict(title = "Type of Guns Used by Stolen Status", xaxis = dict(title = 'Type'), yaxis = dict(title = 'Count'))
trace= []
for i, each in enumerate(gun_pd):
    trace.append(go.Bar(x = gun_pd.index, y =gun_pd[each], name= each ))
data = go.Data(trace)
fig = go.Figure(data= data, layout=layout)
py.offline.iplot(fig)


# In[24]:


df_part_age = df_return(df['participant_age']) 
df_part_age.index = pd.MultiIndex.from_tuples(df_part_age.index)
df_part_typ = df_return(df['participant_type']) 
df_part_typ.index = pd.MultiIndex.from_tuples(df_part_typ.index)
df_part = pd.merge(df_part_typ, df_part_age, left_index=True, right_index=True, how='left')
df_part.columns = ['type', 'age']
df_part.fillna(999,inplace=True)
df_part.age.replace('nan', 999, inplace=True)
df_part.age = df_part.age.apply(lambda x : int(x))


# There are many suspects who are below 2 years of age. Did we mess something up during the feature engineering or were they really below 2 years? Looks like they were, parental negligence caused these incidents, which were totally avoidable.

# In[42]:


l = df.iloc[df_part[(df_part.age <= 2 ) & (df_part.type == 'Subject-Suspect')].reset_index()['level_0']]['notes'].values


# In[41]:


wordcloud = WordCloud(stopwords=STOPWORDS,
                      ).generate(' '.join(str(each) for each in l))

plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[26]:


part_pd = pd.crosstab(df_part.age, df_part.type)
part_pd.drop(999, inplace=True)
part_pd.drop('nan', axis= 1, inplace=True)
part_pd = part_pd[part_pd.index <= 105] # There is some data entry errors it seems, so we will limit the age to 105 or below
part_pd_bin = part_pd.groupby(pd.cut(part_pd.index, np.arange(0, 102, 2), include_lowest=True, precision=0)).sum()
part_pd_bin.index.name = 'age'
part_pd_bin_sum = part_pd_bin.cumsum(axis=0)


# * Almost 40% of total reported crimes are committed by 15 to 26 age group
# * Upto 16 years, majority are victims, but it changes afterwards, as you can see from the CDF curves, the suspect curve was way below the victim curve but it quickly catches up and then takes over, This trend is noticeable from 15-16 group onwards
# * Gun violence committed by 50+ are less than 10%

# In[ ]:


ax1 = part_pd_bin.plot(kind='bar',title = 'Suspect and Victims by Age Group')
ax1.set_ylabel("Count")
ax2 = (part_pd_bin_sum/part_pd_bin_sum.max()).plot(drawstyle='steps',ax= ax1, rot = 90,secondary_y = True,
                                              alpha=0.5,legend=False, figsize=(15,7))
ax2.set_ylabel("%")
plt.tight_layout()


# In[ ]:


df_part_stat = df_return(df['participant_status']) 
df_part_stat.index = pd.MultiIndex.from_tuples(df_part_stat.index)
df_part_2 = pd.merge(df_part_typ, df_part_stat, left_index=True, right_index=True, how='left')
df_part_2.columns = ['type', 'status']
part2_pd = pd.crosstab(df_part_2.status, df_part_2.type)
part2_pd.drop('nan', inplace=True)
part2_pd.drop('nan', axis= 1, inplace=True)


# * Majority of the victims are injured or killed, this is obvious though
# * Majority of the suspects are arrested but unharmed, comparatively very less suspects are injured or killed

# In[ ]:


layout = dict(title = "Suspects and Victims Final Status", xaxis = dict(title = 'Status'), yaxis = dict(title = 'Count'))
trace= []
for i, each in enumerate(part2_pd):
    trace.append(go.Bar(x = part2_pd.index, y =part2_pd[each], name= each ))
data = go.Data(trace)
fig = go.Figure(data= data, layout=layout)
py.offline.iplot(fig)


# In[ ]:


df_part_gend = df_return(df['participant_gender']) 
df_part_gend.index = pd.MultiIndex.from_tuples(df_part_gend.index)
df_part_3 = pd.merge(df_part_typ, df_part_gend, left_index=True, right_index=True, how='left')
df_part_3.columns = ['type', 'Gender']
part3_pd = pd.crosstab(df_part_3.Gender, df_part_3.type)
part3_pd.drop('nan', inplace=True)
part3_pd.drop('nan', axis= 1, inplace=True)


# 1. There are more female victims than female suspects and more male suspects than male victims

# In[ ]:


layout = dict(title = "Suspects and Victims by Gender", xaxis = dict(title = 'Gender'), yaxis = dict(title = 'Count'))
trace= []
for i, each in enumerate(part3_pd):
    trace.append(go.Bar(x = part3_pd.index, y =part3_pd[each], name= each ))
data = go.Data(trace)
fig = go.Figure(data= data, layout=layout)
py.offline.iplot(fig)


# In[ ]:


df_part_4 = pd.merge(df_part_3, df_part_stat, left_index=True, right_index=True, how='left')
df_part_4.columns = ['type', 'gender', 'status']
part4_pd = pd.crosstab(df_part_4[df_part_4['type'] == 'Subject-Suspect']['status'], df_part_4[df_part_4['type'] == 'Subject-Suspect']['gender'])
#part4_pd.drop('nan', inplace=True)
#part4_pd.drop('nan', axis= 1, inplace=True)
part4_pd = (part4_pd.div(part4_pd.sum(axis=0), axis=1)*100)


# * Regardless of gender, majority of the suspects are arrested unharmed
# * Around 5 percent of the male suspects are killed while only 3.6% of the female suspects are killed

# In[ ]:


layout = dict(title = "Suspects Final Status by Gender", xaxis = dict(title = 'Status'), yaxis = dict(title = '%'))
trace= []
for i, each in enumerate(part4_pd):
    trace.append(go.Bar(x = part4_pd.index, y =part4_pd[each], name= each ))
data = go.Data(trace)
fig = go.Figure(data= data, layout=layout)
py.offline.iplot(fig)


# In[ ]:


part4_pd = pd.crosstab(df_part_4[df_part_4['type'] == 'Victim']['status'], df_part_4[df_part_4['type'] == 'Victim']['gender'])
#part4_pd.drop('nan', inplace=True)
#part4_pd.drop('nan', axis= 1, inplace=True)
part4_pd = (part4_pd.div(part4_pd.sum(axis=0), axis=1)*100)


# * As per the data, chances of escaping unharmed is more for female victims than males
# * Male victims are injured or killed more than females

# In[ ]:


layout = dict(title = "Victim Final Status by Gender", xaxis = dict(title = 'Status'), yaxis = dict(title = '%'))
trace= []
for i, each in enumerate(part4_pd):
    trace.append(go.Bar(x = part4_pd.index, y =part4_pd[each], name= each ))
data = go.Data(trace)
fig = go.Figure(data= data, layout=layout)
py.offline.iplot(fig)


# In[ ]:


df_part_5 = pd.merge(df_part, df_part_stat, left_index=True, right_index=True, how='left')
df_part_5.columns = ['type','age','status']
part5_pd = pd.crosstab(df_part_5[df_part_5['type'] == 'Subject-Suspect']['age'], df_part_5[df_part_5['type'] == 'Subject-Suspect']['status'])
part5_pd.drop(999, inplace=True)
#part5_pd.drop('nan', axis= 1, inplace=True)
part5_pd = (part5_pd.div(part5_pd.sum(axis=1), axis=0)*100)
part5_pd.index = part5_pd.index.astype(int)
part5_pd = part5_pd[(part5_pd.index <= 100) & (part5_pd.index >= 1)]


# In[ ]:


layout = dict(title = "Suspects Final Status by Age (1-100)", xaxis = dict(title = 'Status'), yaxis = dict(title = '%'))
trace= []
for i, each in enumerate(part5_pd):
    trace.append(go.Bar(x = part5_pd.index, y =part5_pd[each], name= each ))
data = go.Data(trace)
fig = go.Figure(data= data, layout=layout)
py.offline.iplot(fig)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




