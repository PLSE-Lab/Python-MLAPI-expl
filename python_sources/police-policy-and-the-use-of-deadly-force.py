#!/usr/bin/env python
# coding: utf-8

# <img src="https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2020/06/1862/1048/camden-police-solidarity1.jpg" width="800">
# 
# 
# Racism and police violence are long-standing problems in the United States. The recent killing of George Floyd at the hands of the Minneapolis PD is the latest example. It has received the entire world's attention and brought the issue front and center once again. 
# 
# This notebook examines to what degree the policies of a police department relate to its officers' use of deadly force. The data comes from [this dataset](https://www.kaggle.com/jpmiller/police-violence-in-the-us) which contains data from the [Police Use of Force](http://useofforceproject.org/) project, [Mapping Police Violence](https://mappingpoliceviolence.org/), and the [Washington Post](https://github.com/washingtonpost/data-police-shootings).
# 
# Based on this informal study, it is my opinion that:
#  - Policy adoption varies widely across departments, and there is no standard set of policies common to most of the group.
#   
#  - Two policies show association with lower rates of deadly force for departments that have the policy in place:
#     1. Require officers to report each time they use force or threaten to use force against civilians
#     2. Require officers to de-escalate situations, where possible, by communicating with subjects, maintaining distance, and otherwise eliminating the need to use force
#  
#   
#  - Departments with 5-8 policies in effect may have a lower rate of deadly force than departments with 1-2 policies. Having the two policies above as part of the 5-8 could be a contributor to the difference vs just the number of policies. 
#  
#  - There is no evidence of a "magic recipe" of policy combinations associated with a significant decrease in police shootings across the cities studied.
# 

# ## Introduction
# 
# A 2016 study found that police departments with clear restrictions on when and how officers use force had significantly fewer killings than those that did not have these restrictions in place. The study was sponsored by the [Police Use of Force Project](http://useofforceproject.org). The Project maintains data on the 100 largest departments in the US and whether they have the following policy types in place:
# 
#  * Require officers to de-escalate situations, where possible, by communicating with subjects, maintaining distance, and otherwise eliminating the need to use force
# 
#  * Prohibit officers to choke or strangle civilians, in many cases where less lethal force could be used instead, resulting in the unnecessary death or serious injury of civilians
# 
#  * Require officers to intervene and stop excessive force used by other officers and report these incidents immediately to a supervisor 
# 
#  * Restrict officers from shooting at moving vehicles, which is regarded as a particularly dangerous and ineffective tactic
# 
#  * Apply a Force Continuum that limits the types of force and/or weapons that can be used to respond to specific types of resistance
# 
#  * Require officers to exhaust all other reasonable means before resorting to deadly force
# 
#  * Require officers to give a verbal warning, when possible, before shooting at a civilian
# 
#  * Require officers to report each time they use force or threaten to use force against civilians
#  
# 
# The study concluded that each of the policies had a positive reduction on the use of deadly force by police.
# 
# For this analysis I used the same policy data as used by the study. The deaths used here represent total shooting deaths from Jan 2018 to Jun 2020.
# 

# In[ ]:


conda install -y hvplot


# In[ ]:


import numpy as np
import pandas as pd
import hvplot.pandas
import colorcet as cc
from scipy import stats
from statsmodels.stats import multitest


# ## Policy Adoption
# 
# This section looks at how widely the eight policies have been adopted by the 100 largest police departments. Policy deployment varies widely among the eight types.

# In[ ]:



policies = pd.read_csv('../input/police-violence-in-the-us/police_policies.csv', 
                       index_col=['City'])
policies = policies.dropna(axis=0, how='all')                    .fillna(0).astype(int)                    .rename(index={'Washington DC': 'Washington'})

chart_opts = dict(width=600, height=400)

policies.sum().sort_values().to_frame().hvplot.bar(invert=True, title='Policy Deployment', 
        xlabel='', ylabel='Department Count', **chart_opts)


# Police departments vary in the number of policies they have adopted. Nearly half of the departments have adopted 3-4 policies. All departments have adopted at least one policy.

# In[ ]:


policies.sum(axis=1).value_counts().sort_index().hvplot.bar(title='Department Adoption', 
        xlabel='Number of Policies in Use', ylabel='Department Count', **chart_opts)


# There is little correlation between which of the eight policy types are adopted by any one department. In other words, these eight departments have different sets of policies in place for the most part. Even when restricting the analysis to the eight most populated cities, correlations are not strong. The exception is for departments of all sizes that have adopted 6-8 of the 8 policies. 
# 
# There is a negative correlation between comprehensive reporting and restrictions on shooting at moving vehicles. This may be due to the small sample size and the relative adoption rates of the policies.

# In[ ]:


policies.corr().hvplot.heatmap(rot=45, width=800, height=600, cmap=cc.coolwarm,
                                clim=(-1,1), title="Policy Correlations, All Cities")


# In[ ]:


big_metros = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Las Vegas',
                 'Phoenix', 'Philadelphia', 'Dallas']
policies.loc[big_metros].corr().hvplot.heatmap(rot=45,  width=800, height=600, cmap=cc.coolwarm,
                                clim=(-1,1), title="Policy Correlations, 8 Large Metro Areas")


# Overall it is clear that policy adoption varies widely across departments.

# ## Policy Effectiveness
# 
# The data used here doesn't include shootings before and after a policy was enacted by a city. As such it is hard to absolutely answer whether or not a policy caused a difference. We can however look across cities and compare those with a given policy to those without. Of course that can also be complicated because cities are different and there are many factors at play.
# 
# 
# ### Single policy comparison
# I first compared the average number of shootings between departments without a policy and departments with it. The output below shows results from statistical tests for each of the eight policies. I used the Mann-Whitney U test (one-sided) to compare samples and adjusted for multiple tests using the Benjamini-Hochberg method. 
# 

# In[ ]:


deaths = pd.read_csv('../input/police-violence-in-the-us/fatal_encounters_dot_org.csv', 
            usecols=['Location of death (city)', 'Date (Year)'],
            na_values=['#VALUE!', '#REF!'])
deaths.columns = ['City', 'Year']
deaths = deaths.dropna()                .assign(Year = lambda x: x.Year.astype(int))                .query('Year>2017')                .groupby('City')['Year'].size()                .rename('Deaths')

pops = pd.read_csv('../input/police-violence-in-the-us/deaths_arrests_race.csv', 
                       usecols=['City', 'Total'], index_col='City')
pops['Total'] = pops.Total.str.replace(",", "").astype(int)

policies_all = policies.join([deaths, pops], how='inner')                .assign(DeathsPer100K = lambda x: x.Deaths/x.Total*100_000,
                       PolicyCount = policies.sum(axis=1))


# In[ ]:


dist = policies_all.hvplot.kde('DeathsPer100K')
z, p_norm = stats.normaltest(policies_all.DeathsPer100K)

welch_array = np.zeros((8, 3), dtype='float')
for i, policy in enumerate(policies.columns):

    cities_without = policies[policies.loc[:, policy] == 0].index
    without = policies_all.loc[policies_all.index.isin(cities_without), 'DeathsPer100K']
    with_ = policies_all.loc[~policies_all.index.isin(cities_without), 'DeathsPer100K']
    display(policies_all.hvplot.kde('DeathsPer100K', by=policy))
    
    u_stat, pvalue = stats.mannwhitneyu(without, with_, alternative='greater')

    welch_array[i,0] = round(without.median(), 2)
    welch_array[i,1] = round(with_.median(), 2)
    welch_array[i,2] = round(pvalue, 4)

print("Similarity of shapes checked.")


# In[ ]:


colnames = ['Median Deaths/100K without', 'Median Deaths/100K with', 'p_value']
df = pd.DataFrame(welch_array, columns=colnames, index = policies.columns)    
df['Adjusted p'] =  multitest.multipletests(df.p_value, method='fdr_bh')[1].round(4)
df


# When adjusted for multiple tests, none of the policies meet the usually accepted requirement of having adjusted p-values of less than 0.05. However, two of the policies, Requires Comprehensive Reporting and Requires De-Escalation, show a p-value of 0.09. This indicates that there is a 9% chance that the difference in deaths is due to random chance and not associated with the policy.
# 
# The presence of these policies might still be meaningful. To give this number some perspective, think of a weather person telling you it will rain today. She thinks it will rain and admits there's a 10% probability her conclusion is based on random chance. You might be reluctant to bet a day's pay on her being right about the rain, and yet still take an umbrella to work. There's still value in her estimate, just as with this one.
# 
# 
# 
# ### Comparison for the number of policies
# The output below shows the difference bewtween cities with 5-8 policies in effect and those with 1-2. A p-value of 0.055 indicates a 5.5% chance that the difference is due to randomness, whcih isn't terribly different thana p-value of 0.05. On the other hand, it is possible that the difference is also a matter of the 5-8 policies including the effective policies mentioned above. Finally, there is no such thing as 2.2 or 1.74 people. Both averages generally equate to 2 fatal shootings per 100K people.

# In[ ]:


few = policies_all.loc[policies_all.PolicyCount < 3, 'DeathsPer100K']  # n=22
many = policies_all.loc[policies_all.PolicyCount >4, 'DeathsPer100K']  # n=31
whit, p_val = stats.mannwhitneyu(few, many, alternative='greater')


print(f"Median deaths per 100K for cities with 1-2 policies: {round(few.median(), 2)} \n"
     f"Median deaths per 100K for cities with 5 or more policies: {round(many.median(), 2)} \n"
     f"ROC AUC of Mann-Whitney: {whit/len(few)/len(many)} \n"
     f"p value: {p_val}")


# ### Multi-policy comparison
# 
# Machine learning methods are very good at discovering interactions among factors (policy adoption in this case). I used a method known as [UMAP](https://umap-learn.readthedocs.io/en/latest/) to compare the 100 cities based on each city's combination of policies. The method compares each city to the others and basically groups like with like. A separate group of cities with noticably lower shooting rates could indicate a best practice for policy adoption. The below chart is a spatial map of policy choices made by the 100 cities. Cities with similar sets of policies appear closer to one another on the map. After plotting the cities I overalaid the fatality rates (indicated by dot color) and the number of policies in effect (indicated by dot size.)

# In[ ]:


import umap


reducer = umap.UMAP(n_neighbors=6, metric='hamming', 
                    min_dist=0.1, random_state=20)
embedding = reducer.fit_transform(policies_all.iloc[:, :8])

policies_all = policies_all.assign(V1 = embedding[:,0], 
                                   V2 = embedding[:,1])

policies_all.hvplot.scatter('V1', 'V2', c='DeathsPer100K', s='PolicyCount', scale=4,
                      alpha=0.6, width=600, height=500,
                      hover_cols=['City', 'PolicyCount'])


# There are three general groups with the left group being most cohesive. This group appears to be the cities with 6 or so policies in place and have a good amount of overlap in policy types. There does not appear to be a visible difference in the death rates as indicated by dot colors specific to each group. There is no evidence here of a "magic recipe" of policies associated with fewer fatalities. Keep in mind though that is not a vigorous statistical test and other methods may find something different.

# 
# 
# 
# ## Closing
# 
# The chart below shows which departments have put in place the two policies associated with lower rates of fatal police shootings.

# In[ ]:


policies.iloc[:,[0, 7]].sum(axis=1)     .sort_values()     .hvplot.bar(invert=True, height=1000, line_alpha=0,                     
                xaxis=None, title='Number of Helpful Policies in Effect  (2, 1, 0)')


# City officials without these policies may want to explore adopting them.
