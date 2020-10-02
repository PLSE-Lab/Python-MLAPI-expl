#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename[-4:] == '.csv':
            filepath = os.path.join(dirname, filename)
            #print(filepath)
            try:
                df = pd.read_csv(filepath)
                if 'county' in df.columns:
                    print(filepath)
            except:
                print('The CSV made a boo boo', filepath)

# Any results you write to the current directory are saved as output.


# # Which populations of clinicians and patients require protective equipment?
# This notebook is designed to help answer this question. The first chapter is focusing on the state of New York. It has been strongly affected by the pandemic, but more important it has rich statistics about cases and testing. More chapters will follow.
# 
# ## COVID-19 cases and social vulnerability in the state of New York
# The following lines of code will join 
# 
# **/kaggle/input/uncover/UNCOVER/ny_dept_of_health/new-york-state-statewide-covid-19-testing.csv** 
# 
# with 
# 
# **/kaggle/input/uncover/UNCOVER/esri_covid-19/esri_covid-19/cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level.csv**
#  
# 
# The focus will be on the estimate percentage columns from the social vulnerabilty data for now. (Description of the columns can be found [here](https://svi.cdc.gov/Documents/Data/2016_SVI_Data/SVI2016Documentation.pdf) shoutout to [William](https://www.kaggle.com/whegedusich))

# In[ ]:


soc = pd.read_csv('/kaggle/input/uncover/UNCOVER/esri_covid-19/esri_covid-19/cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level.csv')
nytests = pd.read_csv('/kaggle/input/uncover/UNCOVER/ny_dept_of_health/new-york-state-statewide-covid-19-testing.csv')
# County plus total population and estimate percentage cols
cols = ['county', 'e_totpop'] + [c for c in soc.columns if c[:3] == 'ep_']
tsoc = soc[soc['st_abbr'] == 'NY'][cols]

# Only interested in data of performed tests
tnytests = nytests[nytests['cumulative_number_of_tests_performed']>0]

combo = pd.merge(tnytests, tsoc, how='left')
#combo.head()


# In[ ]:


mapping = { 'new_positives': 'p_newpos',
            'cumulative_number_of_positives' : 'p_cpos', 
            'total_number_of_tests_performed' : 'p_tests',
            'cumulative_number_of_tests_performed' : 'p_ctests'}

for k in mapping:
    combo[mapping[k]] = (combo[k] / combo['e_totpop']) * 100
    
#combo.head()


# In the next step the measures of NY-Tests will be divided by the population, to enable more comparability between the counties. The following picture shows a correlation matrix of the COVID-19 cases and the social vulnerability.

# In[ ]:


cols = ['p_cpos'] + [c for c in combo.columns if 'ep_' in c]
corr = combo[combo.test_date==combo.test_date.max()][cols].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
#sns.heatmap(corr, mask= mask, cmap=cmap, vmax=.3, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})

temp = corr['p_cpos']
ax = temp[temp<1].sort_values().plot(kind='barh', 
                                     figsize=(15,10),
                                     title='Correlation of COVID-19 cases in NY with social vulnerability measures')


# In[ ]:


temp = corr['p_cpos']
top_corrs = temp[(abs(temp)>0.4) & (temp<1)].sort_values(ascending=False).index.values
print('Vulnerability measures with absolute correlation value above 0.4:')
for i in top_corrs:
    print('\t-', i)


# The measures that correlate the most (absolute correlation value > 0.4) will be shown as horizontal bar charts in separate columns. This should show if the distribution of these measures are somehow matching the distribution of cumulated COVID-19-Cases per population.

# In[ ]:


cols = ['p_cpos'] + [c for c in top_corrs]
tdf = combo[combo.test_date==combo.test_date.max()].sort_values(by='p_cpos')
f, axes = plt.subplots(1,11, figsize=(30,30))
#f.suptitle('Banana')
for i,col in enumerate(cols):
    axt = axes[i].set_title(col)
    ax = axes[i].barh(tdf['county'], tdf[col])
    yl = axes[i].set_ylim(0,62)
    xg = axes[i].grid('both')
    #xt = axes[i].set_xticklabels([])
    if i>0:
        yt = axes[i].set_yticklabels([])


# In my intepration of this visualization it seems that the following social vulnerability measures follow the distribution of the COVID-19 cases per population.
# * ep_limeng: Percentage of persons (age 5+) who speak English "less than well"
# * ep_minrty: Percentage minority (all persons except white, nonHispanic) 
# * ep_crowd: Percentage of occupied housing units with more people than rooms
# * ep_munit: Percentage of housing in structures with 10  or more unit
# * ep_noveh: Percentage of households with no vehicle available estimate
# 
# ## COVID-19 cases and social vulnerability for the states and counties of USA
# The following lines will join the previously mentioned social vulnerability data with 
# 
# **'/kaggle/input/uncover/UNCOVER/New_York_Times/covid-19-county-level-data.csv'**
# 
# and apply the same visualization. Social factors like percentage of minorites, housing units and so on do not seem to follow the pattern like seen in NY. My first guess is, that some states have been impacted by the pandemic more and some less. So there can be very crowded counties with many housing units, but the COVID-19 cases can still be quite low. I will invest more time and brain cells into this.

# In[ ]:


cc = pd.read_csv('/kaggle/input/uncover/UNCOVER/New_York_Times/covid-19-county-level-data.csv')
tcc = cc[cc.date==cc.date.max()].copy()
tsoc = soc[['state', 'county', 'e_totpop']+[c for c in top_corrs]].copy()
for col in ['state', 'county']:
    tcc[col] = tcc[col].str.lower()
    tsoc[col] = tsoc[col].str.lower()

ccsoc = pd.merge(tcc, tsoc, on=['state', 'county'], how='inner')

mapping = { 'cases' : 'p_cpos', 
            'deaths' : 'p_cdeaths'}

for k in mapping:
    ccsoc[mapping[k]] = (ccsoc[k] / ccsoc['e_totpop']) * 100
#ccsoc.head()


cols = ['p_cpos'] + [c for c in top_corrs]
#tdf = ccsoc[ccsoc.test_date==ccsoc.test_date.max()].sort_values(by='p_cpos')
tdf = ccsoc[~ccsoc['p_cpos'].isnull()].sort_values(by='p_cpos').reset_index(drop=True)
f2, axes2 = plt.subplots(1,11, figsize=(30,90))
#f.suptitle('Banana')
for i,col in enumerate(cols):
    axt = axes2[i].set_title(col)
    ax = axes2[i].barh(tdf['state'] + ', ' + tdf['county'], tdf[col])
    yl = axes2[i].set_ylim(0,len(tdf))
    #xg = axes2[i].grid('both')
    xt = axes[i].set_xticklabels([])
    if i>0:
        yt = axes2[i].set_yticklabels([])


# ## Potential hospital beds per county compared to population
# The following analysis is aming to find out which states are in danger to have too less hospital beds. The simple idea is to divide the potential free hospital beds by the total population. The following visualization shows the 20 states with the least ratio.

# In[ ]:


hp= pd.read_csv('/kaggle/input/uncover/UNCOVER/esri_covid-19/esri_covid-19/definitive-healthcare-usa-hospital-beds.csv')
hppot = pd.DataFrame(hp.groupby(['state_name', 'county_nam']).potential.sum())
hppot['state'] = [str(i).lower() for i in hppot.index.get_level_values(0).values ]
hppot['county'] = [str(i).lower() for i in hppot.index.get_level_values(1).values ]
hpsoc = pd.merge(hppot, tsoc, on=['state', 'county'], how='inner')
hpsoc['p_hppotpop'] = (hpsoc['potential'] / hpsoc['e_totpop']) * 100


temp = hpsoc.groupby('state').p_hppotpop.sum().sort_values(ascending=False)
ax = temp.tail(20).plot(kind='barh',
                        figsize=(15,10),
                        title='States with the least "Free Hospital Beds to Population Ratio"')


# ## Feedback
# I plan to expand this notebook, but your constructive feedback is already very welcome. 
# 
# Is this helpful in any way? What information is missing? Is there an bias/error in my calculations? Do you have tips on bringing in new data or methods, on my coding? Feel free to leave a comment or contact me through Kaggle.
#     
# 
# ## Roadmap
# Planned steps to improve/ehance this analysis
# * Compare NY with other states that have been strongly affected by the pandemic.
# * Bring in the dynamics of the pandemic. Which counties seem to need less/more equipment based on the growth of the COVID-19 cases?
# * Make use of the geographic information. Can we see the pandemic moving to the neighbor counties/states? Do some neighbor counties/states have enough ressources (hospital beds, icu beds) to help out others? In other words: County XY has 0 hospital bed potential but the neighbours could help, county ZZ has a bigger potential but all the neighbours are at their limits. 
# 
# 
# 
