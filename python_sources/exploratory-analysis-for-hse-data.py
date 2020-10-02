#!/usr/bin/env python
# coding: utf-8

# Incidents on industrial sites are a major source of concern for workers, managers, and even policy makers. As someone who has been on site many times, I have first hand experience with safety measures and incident reports in oil and mining industry.  In this notebook, I will explore this health and safety dataset and will try to find insights into what the major trends are for incidents in each sector, and potentially, what policies can be put in place to reduce the number and severity of incidents. Let's dig in! 

# In[ ]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
get_ipython().run_line_magic('matplotlib', 'inline')
import calendar


# Loading the data:

# In[ ]:


acc=pd.read_csv('../input/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')


# In[ ]:


acc.head(3)


# In[ ]:


acc.size


# In[ ]:


acc.isnull().sum()


# There are no null values but we need to do some clean up:

# In[ ]:


del acc['Unnamed: 0']


# In[ ]:


acc.head(3)


# In[ ]:


acc.rename(columns={'Data': 'Date', 'Genre': 'Gender', 'Employee or Third Party':'Employee type'}, inplace=True)


# **1- what is the trend of accidents with time?**

# some changes to the "Date" column is needed: 

# In[ ]:


acc.Date.max()


# In[ ]:


acc.Date.min()


# In[ ]:


acc['Date'] = pd.to_datetime(acc['Date'])


# In[ ]:


month_order={
    'January':1,
    'February':2,
    'March':3,
    'April':4,
    'May':5,
    'June':6,
    'July':7,
    'August':8,
    'September':9,
    'October':10,
    'November':11,
    'December':12
}


# In[ ]:


acc.groupby('Date').count()['Local'].plot(figsize=(15,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# I want to see the incidents each month:

# In[ ]:


def month_n(a):
    b=calendar.month_name[a]
    return b


# In[ ]:


x=[]
for i in range (0, len(acc.Date)):
                x.append(month_n(acc.Date.loc[i].month))


# In[ ]:


y=[]
for i in range (0, len(acc.Date)):
                y.append(acc.Date.loc[i].year)


# In[ ]:


acc['month']=x
acc['year']=y


# In[ ]:


acc.head(3)


# In[ ]:


acc_trend=acc.pivot_table(index='month', columns=[ 'year','Accident Level'], aggfunc='count')['Countries']


# In[ ]:


n=np.nan
acc_trend.replace(n,0,inplace=True)


# In[ ]:


acc_trend.head(2)


# In[ ]:


acc_trend[2016]


# In[ ]:


acc_trend[2016].loc[month_order].plot(kind='bar', figsize=(15,4), width=0.9, cmap='cool', title='2016')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
acc_trend[2017].loc[month_order].plot(kind='bar', figsize=(15,4), width=0.9, cmap='hot', title='2017')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Ok, first observations: 
# * accidents with the level I are most common. these are your small negligences, like when people forgot their PPE, or when they drop a tool, etc.
# * by looking at 2016 data, you can see that number of incidents in the first half of the year seem to be higher than in the second. The overview for this dataset mentioned that this data is from manufacturing plants in South America, so the first half of the year is cold and the second is warmer. So the number of incidents is higher in colder months. 

# Let's take a look at the role of operator gender: 

# In[ ]:


order={'I':1, 'II':2, 'III':3, 'IV':4, 'V':5}


# In[ ]:


fig=sns.FacetGrid(acc,aspect=1.2,palette="winter", hue='Gender',col='Industry Sector', legend_out=True)
fig.map(sns.countplot, 'Accident Level', order=order)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Looks like there are fewer female operators in the incidents reports and it seems to bo consistent in every sector in the dataset. 

# In[ ]:


sns.countplot('Employee type',data=acc,palette='cool' )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#  There is not much of a difference between Third Party workers and Employees in the incident report. This information would be valueable if we had a better idea of the composition of the workforce. 

# Now let's dive a little deeper into our incidents reports. The factor of "potential incident level" is extremely important here. Something as minor as droping a tool can be a level I incident, but if this is happening when the worker is working on an instrument with a lot of moving parts, it can potentially lead to an explosion. To take this effect into account, I defined a new paramter: "Accident Impact" which is defined as accident level multiplied by potential accident level.  

# In[ ]:


acc['Potential Accident Level'].unique()


# In[ ]:


order2={'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}


# In[ ]:


q=Series()
for i in range(0, len(acc)):
    q=q.append(Series(order2[acc.loc[i]['Accident Level']]*order2[acc.loc[i]['Potential Accident Level']]), ignore_index=True)


# In[ ]:


acc['Accident Impact']=q


# In[ ]:


acc.head(5)


# In[ ]:


acc.plot(x='Date', y='Accident Impact', figsize=(15,4), kind='line')


# There are a number of high peaks on this graph that were not visible in our first graph which was just based on the number of incidents. For example, July 2016 doesn't show a particularly higher number of incidents, while we know from this graph that July 2016 was a ver high risk month and the incidents during that time probably need to be investigated further. The same goes for March of 2017.

# In[ ]:


acc.plot(x='Date', y='Accident Impact', figsize=(15,4), kind='line')
plt.text(x='2016-6-15', y=28.5, s='July 2016', color='red', fontsize=12)
plt.vlines(x='2016-7-1', ymin=25, ymax=28, color='red', linestyles=':', linewidth=3)
plt.text(x='2017-3-5', y=28.5, s='March 2017', color='red', fontsize=12)
plt.vlines(x='2017-3-15', ymin=25, ymax=28, color='red', linestyles=':', linewidth=3)


# In[ ]:


plt.figure(figsize=(15,4))
sns.boxplot(x='month', y='Accident Impact', data=acc, palette='Set3', saturation=1)


# October seems particularly high risk, specially considering the fact that the data from October is from year of 2016 alone.

# Now, I'm curious to see which day of the week has more incidents.

# In[ ]:


wd=[]
for i in range(len(acc)):
    wd.append(acc['Date'].loc[i].weekday())


# In[ ]:


weekday={'0':'Monday',
        '1':'Tuesday', 
        '2':'Wednesday',
        '3':'Thursday',
        '4':'Friday',
        '5':'Saturday',
        '6':'Sunday'}


# In[ ]:


wd_order={'Monday':1,
        'Tuesday':2, 
        'Wednesday':3,
        'Thursday':4,
        'Friday':5,
        'Saturday':6,
        'Sunday':7}


# In[ ]:


wwd=[]
for i in wd:
    wwd.append(weekday[str(i)])


# In[ ]:


acc['weekday']=wwd


# In[ ]:


week_d=acc.pivot_table(index='weekday', columns='Industry Sector', aggfunc='count')['Accident Level']


# In[ ]:


week_d.loc[wd_order].plot(figsize=(10,4), xticks=range(7), cmap='Dark2', kind='line')
plt.ylabel('number of accidents')


# Saturdays are high risk in mining sector, while there is a drop in number of incidents on Saturday for metals sector. this could be due to number of operations running on Saturday, number of people present on site on Saturday or the reporting procedures and how well they are followed on Saturdays.  
# overall, Thursdays seem to be consistently high incident days in all sectors.

# In[ ]:


acc.groupby('weekday').count()['month'].loc[wd_order].plot(kind='line', figsize=(9,4), xticks=range(0,8), color='#FF6A6A')


# In[ ]:


sns.factorplot(x='year', y='Accident Impact', data=acc, hue='Industry Sector', aspect=2, size=4)


#  Operations in both mining and metals sectors are becoming more risky with time. 

# In[ ]:


acc_ind=acc.groupby('Industry Sector').count()['Date']


# In[ ]:


acc_ind_imp=acc.groupby('Industry Sector')['Accident Impact'].mean()


# In[ ]:


acc_ind_imp.plot(kind='pie', figsize=(5,5), cmap='Set1', autopct='%.2f', title='Mean Accident Impact')


# 

# In[ ]:


acc_ind.plot(kind='pie', figsize=(5,5), cmap='Set2', autopct='%.2f', title='Number of Incidents')


# Incidents happen more frequently in Mining sector. Almost twice as Metals.
# 

# Now let's take a look at the critical risk category. This column contains information on the type of incident.

# In[ ]:


acc_cr=acc.pivot_table(index='Critical Risk', columns='Accident Level', aggfunc='count')['month']
acc_cr.replace(n, 0, inplace=True)
acc_cr['total']=acc_cr.sum(axis=1)


# In[ ]:


acc_cr.style.background_gradient(cmap='Blues')


# The obvious problem here is that the row of "Others" has the highest number of entries. This means the reporting and classification process of critical risk is not optimum. The majority of the incidents that happen on site can not be classified using the categories assigned here.
# For now, let's get rid of the "Others" row:

# In[ ]:


acc_cr.drop('Others', axis=0, inplace=True)


# In[ ]:


acc_cr.total.sort_values().plot(kind='barh', figsize=(8,20), xticks=range(0,25), grid=False, width=0.65)
plt.xlabel('total number of accidents')


# In[ ]:


acc_cr.nlargest(6, 'total').style.background_gradient(cmap='winter')


# In[ ]:


acc_ind_risk=acc.pivot_table(index='Critical Risk', columns='Industry Sector', aggfunc='count')['Accident Level']
acc_ind_risk.drop('Others', axis=0, inplace=True)
acc_ind_risk.replace(n, 0, inplace=True)
acc_ind_risk['total']=acc_ind_risk.sum(axis=1)
acc_ind_risk


# In[ ]:


acc_ind_risk.nlargest(6,'total').plot(kind='bar', xticks=range(30), figsize=(15,5), cmap='summer')
plt.xticks(rotation=90)


# In[ ]:


acc_ind_risk_nt=acc_ind_risk.drop('total', axis=1)


# In[ ]:


fig=acc_ind_risk_nt.plot(kind='bar', xticks=range(30),yticks=range(0,21), figsize=(15,6), cmap='Paired', width=0.9)
fig.set_facecolor('#2B2B2B')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.text(x=0.9, y=11, s='V', color='#CD661D', fontsize=25)
plt.text(x=3.3, y=16, s='V', color='#ADD8E6', fontsize=25)
plt.text(x=5.4, y=11, s='V', color='#ADD8E6', fontsize=25)
plt.text(x=14.4, y=15, s='V', color='#ADD8E6', fontsize=25)
plt.text(x=18.4, y=17.5, s='V', color='#ADD8E6', fontsize=25)
plt.text(x=21.6, y=10, s='V', color='#FFD39B', fontsize=25)
plt.text(x=28.6, y=9, s='V', color='#FFD39B', fontsize=25)
plt.text(x=30, y=14, s='V', color='#CD661D', fontsize=25)


# 
# *  The "Others" sector has the highest number of incidents related to bees and venomous animals. Maybe it's mostly related to agriculture?
# *  In Metals, chemical substances, manual tools, cuts, and pressed are the most frequent incident types.
# *  In Mining, projections and mobile equipement are the most frequent type of incidents.
# 

# In[ ]:


acc_risk_other=acc.loc[acc['Critical Risk']=='Others']


# Now I am intersted to know more about the employee composition in the reports that categorized the risk as "Others". My hypothesis is that the third part employees may not be as familiar with risk categories as the company employees and that's why they report most incidents as "Others". 

# In[ ]:


sns.countplot('Employee type', data=acc_risk_other)


# Well, this graph proves that the hypothesis is not right because the compostion of employee type in this group is similar to the one we had for the whole data set.

# In[ ]:


sns.distplot(acc_risk_other['Accident Impact'])


# The good thing is that the majority of incidents in this category have an accident impact of less than five.

# In[ ]:




