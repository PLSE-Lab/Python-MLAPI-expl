#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

mass_shooting = pd.read_csv("../input/Mass Shootings Dataset.csv", encoding = "ISO-8859-1", 
                            parse_dates=["Date"])
mass_shooting.head(50) # shows top 5 entries of the file


# In[ ]:


mass_shooting.info()


# In[ ]:


mass_shooting['Year'] = mass_shooting['Date'].dt.year     # seperates years from date 
year = mass_shooting['Year'].value_counts()           # counts values according to specific year
plt.figure(figsize=(12,6))                                # decides size of plot
sns.barplot(year.index, year.values, alpha=0.7, color=color[2])
"""
Color
0 Blue
1 Orange
2 Green
3 Red
"""
plt.xticks(rotation='vertical')  # makes x axis values vrertical
plt.xlabel('Year of Shooting', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks per Year', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Year', 'Injured', 'Fatalities']].groupby('Year').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries per Year', fontsize=18)


# Attacks have seen a dramatic increase in past few years. 
# Although attacks were less in 2017, it has almost double number of fatalities than 2016.

# In[ ]:


mass_shooting['Quarter'] = mass_shooting['Date'].dt.quarter
quarter = mass_shooting['Quarter'].value_counts()   # counts values according to specific year
plt.figure(figsize=(12,6))                                # decides size of plot
sns.barplot(quarter.index, quarter.values, alpha=0.7, color=color[2])

plt.xlabel('Quarter', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks per Quarter', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Quarter', 'Injured', 'Fatalities']].groupby('Quarter').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries per Quarter', fontsize=18)


# The start and end of the year sees high number of attacks than other quarters.
# If combined, it has 60% increase than other quarters.

# In[ ]:


mass_shooting['Month'] = mass_shooting['Date'].dt.month
month = mass_shooting['Month'].value_counts()           # counts values according to specific year
plt.figure(figsize=(15,8))                                # decides size of plot
month = (month.sort_index())
month.index=['January','February','March','April','May','June','July','August','September','October','November','December']
sns.barplot(month.index, month.values, alpha=0.7, color=color[2])

plt.xlabel('Month', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks per Month', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Month', 'Injured', 'Fatalities']].groupby('Month').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries per Month', fontsize=18)


# In[ ]:


mass_shooting['Week'] = mass_shooting['Date'].dt.weekofyear
week = mass_shooting['Week'].value_counts()  
plt.figure(figsize=(18,8))                                # decides size of plot
sns.barplot(week.index, week.values, alpha=0.7, color=color[2])

plt.xlabel('Week', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks per Week', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Week', 'Injured', 'Fatalities']].groupby('Week').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries per Week', fontsize=18)


# In[ ]:


mass_shooting['Day'] = mass_shooting['Date'].dt.day
day = mass_shooting['Day'].value_counts()  
plt.figure(figsize=(15,8))                                # decides size of plot
sns.barplot(day.index, day.values, alpha=0.7, color=color[2])

plt.xlabel('Day', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks per Day', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Day', 'Injured', 'Fatalities']].groupby('Day').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries per Day', fontsize=18)


# In[ ]:


mass_shooting['Day_Of_Week'] = mass_shooting['Date'].dt.dayofweek
day_of_week = mass_shooting['Day_Of_Week'].value_counts()  
plt.figure(figsize=(15,8))  
day_of_week = (day_of_week.sort_index())
day_of_week.index=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
sns.barplot(day_of_week.index, day_of_week.values, alpha=0.7, color=color[2])

plt.xlabel('Day Of Week', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks per Day', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Day_Of_Week', 'Injured', 'Fatalities']].groupby('Day_Of_Week').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries per DoW', fontsize=18)


# In[ ]:


week = pd.DataFrame({ '0' : sum(day_of_week[:5]), '1' : sum(day_of_week[5:]) }, index=[0,1])
week = pd.DataFrame(week.values.T, columns = week.columns)['0']
week.index = ['Weekdays','Weekends']
sns.barplot(week.index, week.values, alpha=0.7, color=color[2])

plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('Weekdays vs Weekends', fontsize=18) # fontsize and label
plt.show()


# Weekdays have more attack compared to Weekends
# But there are **5 days in Weekdays** and on normalization 
# we can see **no of attacks** on average on a **Weekday is same as that on Weekends**

# In[ ]:


mass_shooting['Race'].unique()


# In[ ]:


mass_shooting['Race'].fillna('0', inplace=True)
mass_shooting.Race.replace(['0','white', 'black', 'Some other race', 'unclear','White ','Black American or African American/Unknown','White American or European American/Some other Race','Asian American/Some other race','Native American','White American or European American','Black American or African American','Asian American','Native American or Alaska Native','Two or more races'], 
                           ['Unknown','White', 'Black', 'Other','Unknown','White','Black American or African American','White American or European American','Asian American','Native American or Alaska Native','White','Black','Asian','Natives','Other'], inplace=True)
mass_shooting['Race'].unique()


# In[ ]:


day_of_week = mass_shooting['Race'].value_counts()  
plt.figure(figsize=(15,8))  
sns.barplot(day_of_week.index, day_of_week.values, alpha=0.7, color=color[2])

plt.xlabel('Year of Shooting', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks per Year', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Race', 'Injured', 'Fatalities']].groupby('Race').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries per Race', fontsize=18)


# In[ ]:


mass_shooting['Gender'].unique()


# In[ ]:


mass_shooting['Gender'].fillna('0', inplace=True)
mass_shooting['Gender'].replace(['0','M','M/F'],
                        ['Unknown','Male','Male/Female'],inplace=True)
mass_shooting['Gender'].unique()


# In[ ]:


day_of_week = mass_shooting['Gender'].value_counts()  
plt.figure(figsize=(15,8))  
sns.barplot(day_of_week.index, day_of_week.values, alpha=0.7, color=color[2])

plt.xlabel('Gender', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks by gender', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Gender', 'Injured', 'Fatalities','Total victims']].groupby('Gender').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries by Gender', fontsize=18)


# **Males ** are responsible for most attacks and number of victims category.

# In[ ]:


mass_shooting['City'] = mass_shooting['Location'].str.rpartition(',')[0]
mass_shooting['State'] = mass_shooting['Location'].str.rpartition(', ')[2]
g = mass_shooting['State'].unique()
"""mass_shooting['State'].replace(['','NV','CA','PA','WA','D.C.','LA'],
                        ['Unknown','Nevada','California','Pennysylvania','Washington','Washington D.C.','Louisiana'],inplace=True)
mass_shooting['State'].unique()"""

f=['Maryland','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY',]
g = np.append(g,f)
mass_shooting['State'].value_counts('')
g = np.delete(g,10)
print(g)
ll = mass_shooting['State']
print(ll)


# In[ ]:


#print (mass_shooting['State'] )
for i in range(397):
    h = []
    #print (mass_shooting['State'][i])
    if mass_shooting['State'][i] == '':
        s1 = str(mass_shooting['Summary'][i])
        s2 = str(mass_shooting['Title'][i])
        s = s1+s2
        s = s.replace(',','')
        s = s.replace('.',' ')
        #print(s)
        h =  ([x.strip() for x in s.split(' ')])
        for j in range(len(h)):
            #print(i,h,)
            if h[j] in g and mass_shooting['State'][j]=='':
                print(i,h,)
                #print (h[j])
                mass_shooting['State'][i] = h[j]
                break
#mass_shooting.head(100)


# In[ ]:


day_of_week = mass_shooting['State'].value_counts()  
plt.figure(figsize=(15,8))  
sns.barplot(day_of_week.index[1:], day_of_week.values[1:], alpha=0.7, color=color[2])

plt.xticks(rotation='vertical')  # makes x axis values vrertical
plt.xlabel('State', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks by State', fontsize=18) # fontsize and label
plt.show()


# **California** has highest no of attacks, almost an **increase of 80%** from second placed** Florida.**

# In[ ]:


tot = mass_shooting[['State', 'Injured', 'Fatalities']].groupby('State').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries by State', fontsize=18)


# Most **injuries** are recorded in **Nevada higher than next 2 combined**, followed by** California , Colorada and Texas**
# Highest **fatalities** are seen in **California **foloowed by **Florida and Texas**

# In[ ]:


day_of_week = mass_shooting['City'].value_counts()  
plt.figure(figsize=(15,8))  
sns.barplot(day_of_week.index[:50], day_of_week.values[:50], alpha=0.7, color=color[2])

plt.xticks(rotation='vertical')  # makes x axis values vrertical
plt.xlabel('City', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks by City', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['City', 'Fatalities', 'Injured','Total victims']].groupby('City').sum()
tot = tot.sort_values(['Total victims'], ascending=[False])[:30]
tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries by City', fontsize=18)


# In[ ]:


mass_shooting['Mental Health Issues'].unique()


# In[ ]:


mass_shooting['Mental Health Issues'].replace(['Unknown','unknown','Unclear '],
                        ['Unclear','Unclear','Unclear'],inplace=True)
mass_shooting['Mental Health Issues'].unique()


# In[ ]:


day_of_week = mass_shooting['Mental Health Issues'].value_counts()  
plt.figure(figsize=(15,8))  
sns.barplot(day_of_week.index, day_of_week.values, alpha=0.7, color=color[2])

plt.xlabel('Mental State', fontsize=12)  # fontsize and label
plt.ylabel('Number of Attacks', fontsize=12)  # fontsize and label
plt.title('No. of Attacks by Mental State', fontsize=18) # fontsize and label
plt.show()


# In[ ]:


tot = mass_shooting[['Mental Health Issues', 'Fatalities', 'Injured','Total victims']].groupby('Mental Health Issues').sum()

tot.plot.bar(figsize=(15,8))
plt.ylabel('Number of Victims', fontsize=12)
plt.title('Fatalities vs Injuries by Mental Health Issues', fontsize=18)


# A very important observation here is 
# 
# **Fatalities drastically increase if the attacker has mental health issues.**

# This is my first kernel, feel free to give suggestions on how to make better analysis and graphs.
# Thank you.

# In[ ]:




