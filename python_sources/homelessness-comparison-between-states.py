#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

homeless = pd.read_csv("../input/2007-2016-Homelessnewss-USA.csv")
population = pd.read_csv("Population-by-state.csv")


# In[ ]:


"""Clean population dataset"""
pop = population.copy()
pop.columns = pop.iloc[0]
pop.drop(0,axis=0, inplace=True)
pop.drop(['Id', 'Id2','April 1, 2010 - Census', 'April 1, 2010 - Estimates Base'], axis=1, inplace=True)
pop.columns = ['State','pop10','pop11','pop12','pop13','pop14','pop15','pop16']

"""Clean homelss dataset"""
hless = homeless.copy()
hless['Count'] = hless['Count'].str.replace(',', '').astype(np.int64) #turn count number into int
hless.drop(['CoC Number','CoC Name'], axis=1, inplace=True)
hless['Year'] = pd.to_datetime(hless['Year'])
hless['Year'] = hless['Year'].dt.year
hless.head()


# In[ ]:


pop.head()


# In[ ]:


"""Difference in homeless between California and NY"""

g = hless[hless['Measures']=='Total Homeless'].groupby(['State', 'Year'])[['Count']].sum()
ttl_homeless1 = g.reset_index()

ax = ttl_homeless1[ttl_homeless1['State']=='CA'].plot.area('Year', 'Count', alpha=0.5, color='red', label='CA')
ttl_homeless1[ttl_homeless1['State']=='NY'].plot.area('Year', 'Count', color='green', 
                                                   ax=ax, label='NY')
plt.legend(loc='upper right')
plt.title('Total number of homeless(California, New York)')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()


# In[ ]:


"""Difference between 2010 and 2016 (unstandardized, absolute numbers)"""

df_hless = ttl_homeless1[(ttl_homeless1['Year']==2016) | (ttl_homeless1['Year']==2010)]
counting = df_hless.groupby('State')[['Count']].diff().fillna(0)
counting.columns=['Difference']
dd = counting.join(df_hless)
final = dd.groupby('State')['Difference'].sum()
final.plot.bar(figsize=(14,6))
plt.title('Difference in homeless poplution between 2010-2016 (unstandardized, absolute numbers)')
plt.show()


# In[ ]:


"""Standardized (merge homelessness with poplution datasets)"""

st_hless = hless.copy()
st_pop = pop.copy()

st_pop['State'] = st_pop['State'].replace({'Alaska':'AK', 'Alabama':'AL', 'Arkansas':'AR', 'Arizona':'AZ', 
                      'California':'CA', 'Colorado':'CO', 'Connecticut':'CT', 
                      'District of Columbia':'DC', 'Delaware':'DE', 'Florida':'FL', 
                      'Georgia':'GA', 'Hawaii':'HI', 'Iowa':'IA', 
                      'Idaho':'ID', 'Illinois':'IL', 'Indiana':'IN', 'Kansas':'KS', 
                      'Kentucky':'KY', 'Louisiana':'LA', 'Massachusetts':'MA', 'Maryland':'MD', 
                      'Maine':'ME', 'Michigan':'MI', 'Minnesota':'MN', 'Missouri':'MO', 
                      'Mississippi':'MS', 'Montana':'MT', 'North Carolina':'NC', 
                      'North Dakota':'ND', 'Nebraska':'NE', 'New Hampshire':'NH', 
                      'New Jersey':'NJ', 'New Mexico':'NM', 'Nevada':'NV', 'New York':'NY', 
                      'Ohio':'OH', 'Oklahoma':'OK', 'Oregon':'OR', 'Pennsylvania':'PA', 
                      'Puerto Rico':'PR', 'Rhode Island':'RI', 'South Carolina':'SC', 
                      'South Dakota':'SD', 'Tennessee':'TN', 'Texas':'TX', 'Utah':'UT', 
                      'Virginia':'VA', 'Vermont':'VT', 'Washington':'WA', 
                      'Wisconsin':'WI', 'West Virginia':'WV', 'Wyoming':'WY'})

"""Take out Virging Islands and Guam because we have no population for them. Start from 2010"""

st_hless = st_hless[(st_hless['State']!= 'GU') & (st_hless['State']!= 'VI')] 
st_hless =  st_hless[(st_hless['Year']!= 2007) & (st_hless['Year']!= 2008) &
                    (st_hless['Year']!= 2009)]
homelessness = st_hless.merge(st_pop, on='State')
homelessness.head()


# In[ ]:


"""Homelessness (standardized)"""
hlessness = homelessness.copy()
hlessness = hlessness[hlessness['Measures']=='Total Homeless']
hlessness = hlessness[(hlessness['Year']==2010) | (hlessness['Year']==2016)]
hlessness = hlessness.groupby(['State','Year',])[['Count']].sum()
hlessness.reset_index(inplace=True)

hlessness = hlessness.merge(st_pop, on='State')
hlessness['pop10'] = hlessness['pop10'].astype(int)
hlessness['pop16'] = hlessness['pop16'].astype(int)

p10 = hlessness.where(hlessness['Year']==2010)
p16 = hlessness.where(hlessness['Year']==2016)
p10['pop_percnt10'] = p10['Count']/p10['pop10']
p16['pop_percnt16'] = p16['Count']/p16['pop16']

#Below, I am filling p10 with data belonging to p16. But before I do that, I need to rename column 
#'pop_percnt16' in data p16 because I want all data to go under column 'pop_percnt10' in p10
hlessness = p10.combine_first(p16.rename(columns={'pop_percnt16':'pop_percnt10'}))
hlessness = hlessness.rename(columns={'pop_percnt10':'%_of_pop'})

percent = hlessness[hlessness['Year']==2016]
print ("1.2 out of every 100 people in DC is homeless!!!!\n\nMissourri has lowest % of homeless, with less than 1 out of every 1000\n\nIn New York, almost 1 out of every 200 is homeless")
plt.figure(figsize=(18,6))
sns.barplot(y='%_of_pop', x='State', data=percent)
plt.ylabel('Percentage')
plt.title('Percentage of homeless from state population')

plt.show()


# In[ ]:


"""Difference between 2010 and 2016 (standardized)"""
difference = hlessness.groupby('State')[['%_of_pop']].pct_change().fillna(0)
difference = difference.join(hlessness, lsuffix='_change2010_2016')
difference = difference[difference['%_of_pop_change2010_2016']!=0.0]

print("Louisiana's percentage of homeless people dropped by almost 70%\n\nNew York's increased by almost 30%")
plt.figure(figsize=(18,6))
plt.title('Difference in percentage of homeless from 2010 to 2016')
sns.barplot(y='%_of_pop_change2010_2016', x='State', data=difference)
plt.ylabel('Percentage')
plt.show()


# In[ ]:


"""(Parenting Youth (Under 25)) and elderly people (Homeless Veterans)"""

print ('To end on a good note: Homeless vets has decreased by tens of thousands')
vets = homelessness.copy()
vets = vets[vets['Measures']=='Homeless Veterans']
homeless_vets = vets.groupby('Year')[['Count']].sum()
homeless_vets.reset_index(inplace=True)
plt.plot(homeless_vets['Year'], homeless_vets['Count'])
plt.title("Homeless Veterans in all 50 US states + Puerto Rico")
plt.ylabel('Count')
plt.show()


# In[ ]:


"""Practice get.dummies()"""
homeless.Measures.unique()
homeless_cols = homeless['Measures'].str.get_dummies()
homeless.join(homeless_cols[['Homeless Individuals','Homeless People in Families', 'Unsheltered Homeless']])

