#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.listdir()
data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
# Any results you write to the current directory are saved as output.


# In[ ]:


data.columns
data.columns.values
data.info()


# In[ ]:


#  1. Number of incidents taken place ? - count incidents ID  
data.isnull().sum()


# In[ ]:


data[['incident_id']] # Ans: 239677


# In[ ]:


#How many where injured ?
injured = data[['n_injured']].sum() 
injured #Ans: 118402


# In[ ]:


#How many were killed ? 
killed = data[['n_killed']].sum() 
killed #Ans: 69468


# In[ ]:


#Whats the total number of victims 
tv = float(injured) + float(killed)
tv #NS: 178870


# In[ ]:


#Plotting of killed & injured 

from matplotlib import pyplot as plt


# In[ ]:


plt.plot(data.incident_id,data.n_injured, 'ro' ,data.incident_id,data.n_killed, 'g+')
plt.title('Test Plot')
plt.xlabel('Incident_Id')
plt.ylabel('killed & Injured')
plt.legend(["killed", "injured"])
plt.show()


# In[ ]:


#  2. in which year or month or on date maximum incidents occoured? - play with dates 

df =data
df['date'] = pd.to_datetime(df.date)
df.dtypes


# In[ ]:


#Yearly Total Number of Cases Registered 
df['years'] =df.date.dt.year
df.years
df.head()


# In[ ]:


df.years.value_counts().sort_index().plot(kind = 'pie', figsize = (15,15))
plt.legend('years')
plt.title('Number of Incidents Yearly')
plt.xlabel('Years')
plt.ylabel('Number of cases')


# In[ ]:


#Monthly Total number of cases resgistered 
df['months'] = df.date.dt.month
df.months
df.head


# In[ ]:


df.months.value_counts().sort_index().plot(kind = 'bar', figsize = (15,15))
plt.legend('months')
plt.title('Collective Monthwise Distribution of Crime')
plt.xlabel('Months')
plt.ylabel('Collective number of cases')

#df.drop(['months'], axis = 1, inplace = True)
#df.drop(['years'], axis = 1, inplace = True)


# In[ ]:


#  3. In which state maximum incudent occoured? - state

df.info()


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.state.unique()


# In[ ]:


#Bar Chart
df.state.value_counts().sort_index().plot(kind = 'barh', figsize = (20,15))
plt.legend('States')
plt.title('Statewise distribution of incidents')
plt.xlabel('Number of incidents')
plt.ylabel('States')


# In[ ]:


#  4. Statewise show dates with maximum incidents?- 

#Pie Chart
df.state.value_counts().head().plot(kind = 'pie', figsize = (15,15))
plt.legend('States')
plt.title('Statewise distribution of incidents')
plt.xlabel('Number of incidents')
plt.ylabel('States')


# In[ ]:


#State with minimum incidents recorded 
df.state.value_counts().tail(10).plot(kind = 'barh', figsize = (15,15))
plt.legend('States')
plt.title('Safest States in USA')
plt.xlabel('Number if incidents')
plt.ylabel('States')


# In[ ]:


#  8. Number of states ?
x = df.state.value_counts()
x.count() #Ans: 51 States 


# In[ ]:


#or 
df.state.value_counts().count() #Ans: 51 States 


# In[ ]:


#  9.  Number of city or county ?

df.columns


# In[ ]:


df.dtypes


# In[ ]:


df.city_or_county.value_counts()
#Ans: There are in Total of 12898 cities or counties listed 


# In[ ]:


#Top 10 Cities with maximum occurences 
df.city_or_county.value_counts().head(10).plot(kind = 'pie', figsize = (15, 15))
plt.legend('Cities')
plt.title('Top 10 Cities or Counties with maximum incidents from 2013 to 2018')
plt.xlabel('Number of incidents')
plt.ylabel('Top 10 Cities or Counties')


# In[ ]:


#Top 10 Safest Cities or Counties 
#Top 10 Cities with least number of incidents

df.city_or_county.value_counts().tail(10).plot(kind = 'pie', figsize = (15,15))
plt.legend('Cities')
plt.title('Top 10 Cities or Counties with minimum incidents from 2013 to 2018')
plt.xlabel('Number of incidents')
plt.ylabel('Safest Cities or Counties')


# In[ ]:


# 10. Number of killings :- Total & statewise  ? 

df.dtypes


# In[ ]:


import seaborn as sns


# In[ ]:



sns.jointplot("years","n_killed", df, kind ='resid')


# In[ ]:


#Number of People killed monthly 
sns.jointplot("months", "n_killed", df,kind ='reg')


# In[ ]:


# 11. Number of injured ? : total & statewise 
sns.jointplot("years","n_injured",df, kind = 'scatter')


# In[ ]:


# Number of people injured month wise 
sns.jointplot("months","n_injured", df, kind = 'hex',  color = "green")


# In[ ]:


###############################################################################
from matplotlib import rcParams #for increasing image size
rcParams['figure.figsize'] = 20,15
sns.violinplot("months", "n_injured", data=df, fill = 'state' );     
sns.violinplot( "n_killed", "months", data=df, color = 'blue');    
plt.legend('Incidents')


###############################################################################


# In[ ]:


varsforgrid = ['years', 'months', 'n_injured', 'n_killed']


# In[ ]:


g = sns.PairGrid(data,vars=varsforgrid,hue='state', size = 4.5)
g = g.map_diag(plt.hist)           
g.map_offdiag(plt.scatter)


# In[ ]:


#A Few Observations :

#1. Data recorded from 01-2013. to 03-2018, giving details related gun violence occourences in USA.
#2. Total occourence from 2013 to 2018 are 2,39,677.
#3. Majorly divided into 2 parts Injured or Kill 
#4. Total of Injured are 1,18,402.
#5. Total of killed 69,468.
#6. Number of incidents in 2014, 2015, 2016 & 2017 are very high and more or less same. 
#7. Number of incidents in 2013 are very less and in 2018 have reduced drastically.
#8. Maximum number of cases are in Janurary or February
#9. Numbers reduces significantly in November 
#10. Teaxas, Illinois, Florida & california has the maximum number of incidents. 
#11. Safest state is Hawaii 
#12. Records mention details for 51 states &  12,898. cities


# In[ ]:




