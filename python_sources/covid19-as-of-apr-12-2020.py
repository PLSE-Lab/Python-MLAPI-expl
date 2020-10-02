#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


world = pd.read_csv("../input/httpsourworldindataorgcoronavirussourcedata/new version WHO.csv")
world.head(2)


# In[ ]:


plt.figure(figsize=(21,8)) # Figure size
world.groupby("location")['total_cases'].max().plot(kind='bar', color='brown')


# In[ ]:


USA = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
USA = USA.drop(['fips'], axis = 1) 
USA.tail(3)


# In[ ]:


#Regression plot of cases-to-deaths in the US
sns.regplot(x='cases', y='deaths', data=USA, color='cadetblue')


# In[ ]:


#correlation
USA.corr().style.background_gradient(cmap='magma')


# In[ ]:


#US Overall cases
plt.figure(figsize=(16,7)) # Figure size
sns.lineplot(x='date', y='cases', data=USA, marker='o', color='palevioletred') 
plt.title('Cases per day across the US') # Title
plt.xticks(USA.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# In[ ]:


#US Overall cases
plt.figure(figsize=(16,7)) # Figure size
sns.lineplot(x='date', y='deaths', data=USA, marker='o', color='gray') 
plt.title('Deaths per day across the US') # Title
plt.xticks(USA.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# In[ ]:


#Days since 1st US case reported
a=USA['date'].nunique()
a


# ### As of April 11, 2020, 80 days since the first US reported case.

# In[ ]:


#In the beginning...
USA.sort_values(by=['date'], ascending=True).head(20)


# In[ ]:


#Most recently...
USA.sort_values(by=['date'], ascending=False).head(10)


# In[ ]:


#COVID19 cases across states
plt.figure(figsize=(19,7)) # Figure size
USA.groupby("state")['cases'].max().plot(kind='bar', color='darkblue')


# In[ ]:


##For ease of visualization
NY=USA.loc[USA['state']== 'New York']
WA=USA.loc[USA['state']== 'Washington']
IL=USA.loc[USA['state']== 'Illinois']
Penn=USA.loc[USA['state']== 'Pennsylvania']
PUR=USA.loc[USA['state']== 'Puerto Rico']


# In[ ]:


# Concatenate dataframes above
States=pd.concat([NY,WA,IL,PUR,Penn]) 

States=States.sort_values(by=['date'], ascending=True)
States.head(2)


# In[ ]:


plt.figure(figsize=(15,9))
plt.title('COVID-19 cases comparison of WA, IL, NY, PR, and Pennsylvania') # Title
sns.lineplot(x="date", y="cases", hue="state",data=States, palette="Set2")
plt.xticks(States.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


##Washington state
plt.figure(figsize=(16,11))
plt.title('COVID-19 cases in WA') # Title
sns.lineplot(x="date", y="cases", hue="county",data=WA)
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


cens = pd.read_csv('../input/uncover/UNCOVER/us_cdc/us_cdc/500-cities-census-tract-level-data-gis-friendly-format-2019-release.csv')


# In[ ]:


cens=cens[['stateabbr', 'placename','stroke_crudeprev', 'obesity_crudeprev', 
           'diabetes_crudeprev', 'copd_crudeprev', "casthma_crudeprev", "cancer_crudeprev",
          'arthritis_crudeprev']]
cens= cens.rename(columns={'placename': 'city','stroke_crudeprev': 'stroke', 'copd_crudeprev':'copd',
                           'obesity_crudeprev': 'obesity','diabetes_crudeprev': 'diabetes',
                           'casthma_crudeprev':'asthma', 'cancer_crudeprev': 'cancer', 'arthritis_crudeprev':'arthritis'})
cens.head()


# In[ ]:


#Cancer incidence per city
plt.figure(figsize=(21,8)) # Figure size
cens.groupby("stateabbr")['cancer'].max().plot(kind='bar', color='darkorange')


# In[ ]:


#COPD incidence per city
plt.figure(figsize=(21,8)) # Figure size
cens.groupby("stateabbr")['copd'].max().plot(kind='bar', color='mediumaquamarine')


# In[ ]:


#obesity incidence per city
plt.figure(figsize=(21,8)) # Figure size
cens.groupby("stateabbr")['obesity'].max().plot(kind='bar', color='peru')


# In[ ]:


#asthma incidence per city
plt.figure(figsize=(21,8)) # Figure size
cens.groupby("stateabbr")['asthma'].max().plot(kind='bar', color='orchid')


# In[ ]:


cens.corr().style.background_gradient(cmap='cubehelix')


# In[ ]:


vuln= pd.read_csv('../input/uncover/UNCOVER/esri_covid-19/esri_covid-19/cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level.csv')


# In[ ]:


# iterating the columns to find column names
for col in vuln.columns: 
    print(col)


# In[ ]:


vuln=vuln[['state', 'county', 'e_totpop', 'e_pov',
           'e_age65','e_age17', 'e_disabl', 'e_minrty']]
vuln= vuln.rename(columns={'e_totpop': 'est population','e_pov': 'est. poverty',
                           'e_age65': 'over 65','e_age17': 'under 17',
                           'e_disabl':'disabled', 'e_minrty': 'minority'})
vuln.head(2)


# In[ ]:


v = vuln.drop(['county'], axis = 1) 
v.head()


# In[ ]:


v.corr().style.background_gradient(cmap='cividis')


# In[ ]:


plt.figure(figsize=(19,9)) # Figure size
v.groupby("state")['est. poverty'].max().plot(kind='bar', color='forestgreen')


# In[ ]:


#elderly
plt.figure(figsize=(19,9)) # Figure size
v.groupby("state")['over 65'].max().plot(kind='bar', color='rebeccapurple')

