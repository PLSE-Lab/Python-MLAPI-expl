#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Age and Screeners
# ==================
# 
# We explore whether or not the age group a patient falls into influences how 
# likely they are to be screened for cervical cancer.

# In[ ]:


import numpy as np 
import pandas as pd 
import pylab as plt
import matplotlib as mpl
import sqlite3
from scipy import stats


# In[ ]:


#Collect the patient demographic data
db = sqlite3.connect('../input/database.sqlite')
sdf = pd.read_sql_query("SELECT patient_age_group, patient_state, ethinicity, household_income, education_level, is_screener FROM patients_train;",db)


# In[ ]:


#Number of patients in each age group and screener rate for each age group
age_group_totals = sdf.groupby('patient_age_group').size()
screener_rates = sdf.groupby('patient_age_group').apply(lambda x: x.is_screener.sum()/x.shape[0])

#Test data age_group_totals
tdf = pd.read_sql_query("SELECT patient_age_group FROM patients_test;",db)
test_age_group_totals = tdf.groupby('patient_age_group').size()


# In[ ]:


#95% confidence intervals for screener rates
conf_ints = [stats.binom.interval(.95, age_group_totals[x],screener_rates[x])/age_group_totals[x] for x in screener_rates.index]


# In[ ]:


#Bar colors ratio of num_patients_in_age_group/max_num_patients_in_an_age_group
cm = plt.get_cmap('Blues')
colors = [cm(age_group_totals.loc[x]/age_group_totals.max()) for x in screener_rates.index]

#Plot the training data 
fig = plt.figure(figsize=(17,12))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)
p1 = screener_rates.plot(kind = 'barh', color = colors,xerr = [x[1]-x[0] for x in conf_ints])
t1 = plt.title('Screener Rates by Age Group',fontsize = 20)
t2 = plt.ylabel('Age Group')


# **The training data indicates that screening rates drop as a function of age.**  The 
# next question then is how consistent is this drop if we look at subcategories based 
# on other variables.  
# 
# Is the drop in screening rates similar in all states?
# ------------------------------------------------
# 
# ###Answer:  no 
# 
# To visualize this we plot the screening rates by state for the 10 states with the most patients 
# including in each case a shaded area representing the 95% confidence interval.

# In[ ]:


#Screening rate by age and state
sr_age_state = sdf.groupby(['patient_age_group','patient_state',]).apply(lambda x: x.is_screener.sum()/x.shape[0])

#Number of patients by age and state
age_state_totals = sdf.groupby(['patient_age_group','patient_state',]).size()

#Confidence intervals for screening rates by age and state
age_state_conf_ints = pd.DataFrame(np.array(stats.binom.interval(.95, age_state_totals,sr_age_state)).T,index = sr_age_state.index)
age_state_conf_ints.columns = ['lower','upper']
age_state_conf_ints = age_state_conf_ints.apply(lambda x: x/age_state_totals,axis = 0)

#Length of confidence interval
age_state_sr_err = age_state_conf_ints.upper - age_state_conf_ints.lower

#States with the most patients sorted
use_states = age_state_totals.unstack().sum(axis = 0).sort_values(ascending=False).index[:10]


# In[ ]:


#cm2 = plt.get_cmap('Blues')
cm2 = plt.get_cmap('gist_ncar')
color_list = [cm2(i/len(use_states)) for i in range(len(use_states))]
#Using http://colorbrewer2.org  First Qualitative list with 10 data classes
color_list = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']

fig = plt.figure(figsize=(17,12))
ax = fig.add_subplot(111, axisbg='w')

p1 = screener_rates.plot(color = 'black', label = 'All', lw = 2)
plt.fill_between(range(len(screener_rates.index)),[x[0] for x in conf_ints],[x[1] for x in conf_ints], color = 'green', alpha = .1)

for st_code, st_color in zip(use_states,color_list):
    sr_age_state.unstack()[st_code].plot(color = st_color,label = st_code, lw = 2 )
    plt.fill_between(range(len(screener_rates.index)), age_state_conf_ints.lower.unstack()[st_code], age_state_conf_ints.upper.unstack()[st_code], color = st_color, alpha = .1)

l1 = plt.legend()
t1 = plt.title('Screener Rates by Age Group for Ten States',fontsize = 20)
t2 = plt.xlabel('Age Group')
t3 = plt.ylabel('Screener Rate')


# Starting at the top California is unusual because the screening rate is 
# not strictly decreasing and is in fact *lower* for lowest age groups than 
# for the middle age groups.  There are no states where the screening rates are
# good for the highest age groups but in some cases the drop is less precipitous.
# Florida and Maryland both have rather low overall screening rates, however in 
# the case of Florida the screening rates are much more equal across all age groups 
# whereas in Maryland the screening rate for the youngest age group is relatively 
# good but then the rates drop sharply and stay below those of other states shown 
# until the very end.
# 
# Does the drop in screening rates vary with ethnicity, household income or education level?
# ---------------------------------------------------
# 
# ###Answer:  Yes, but not by a lot.

# In[ ]:


col_names = ['ethinicity','household_income', 'education_level']

sr_age_feat = {}
age_feat_totals = {}
age_feat_conf_ints = {}

sr_feat = {}
feat_totals = {}

for feat in col_names:
    #Screening rate by age and feature
    sr_age_feat[feat] = sdf.groupby(['patient_age_group',feat]).apply(lambda x: x.is_screener.sum()/x.shape[0])
    #Number of patients by age and feature
    age_feat_totals[feat] = sdf.groupby(['patient_age_group',feat]).size()
    #Confidence intervals for screening rates by age and feature
    age_feat_conf_ints[feat] = pd.DataFrame(np.array(stats.binom.interval(.95, age_feat_totals[feat],sr_age_feat[feat])).T,index = sr_age_feat[feat].index)
    age_feat_conf_ints[feat].columns = ['lower','upper']
    age_feat_conf_ints[feat] = age_feat_conf_ints[feat].apply(lambda x: x/age_feat_totals[feat],axis = 0)
    #Screening rates and totals by ethinicity
    sr_feat[feat] = sdf.groupby([feat]).apply(lambda x: x.is_screener.sum()/x.shape[0])
    feat_totals[feat] = sdf.groupby([feat]).size()


# In[ ]:


cm2 = plt.get_cmap('gist_ncar')
color_list = [cm2(i/len(use_states)) for i in range(len(use_states))]
#Using http://colorbrewer2.org  First Qualitative list with 10 data classes
color_list = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']

fig = plt.figure(figsize=(17,18))
for feat, num in zip(col_names,range(len(col_names))):
    ax = fig.add_subplot('31%d'%(num+1), axisbg='w')
    p1 = screener_rates.plot(color = 'black', label = 'All', lw = 2)
    plt.fill_between(range(len(screener_rates.index)),[x[0] for x in conf_ints],[x[1] for x in conf_ints], color = 'green', alpha = .1)
    if len(feat_totals[feat].index) < 10:
        codes = feat_totals[feat].index
        color_list = color_list[:len(feat_totals[feat].index)]
    else:
        codes = feat_totals[feat].index[:10]
    for st_code, st_color in zip(codes,color_list):
        sr_age_feat[feat].unstack()[st_code].plot(color = st_color,label = st_code, lw = 2 )
        plt.fill_between(range(len(screener_rates.index)), age_feat_conf_ints[feat].lower.unstack()[st_code], age_feat_conf_ints[feat].upper.unstack()[st_code], color = st_color, alpha = .1)
    l1 = plt.legend()
    t1 = plt.title('Screener Rates by Age Group and %s'%feat,fontsize = 20)
    t2 = plt.xlabel('Age Group')
    t3 = plt.ylabel('Screener Rate')


# Unlike with patient state, all of these curves are at least concave down and fairly close 
# to the overall population curve. From these we can infer that people with a higher household 
# income are more likely to be screened.  There are a few surprises though.  In general people 
# identified as African American are less likely to be screen than Caucasion or Hispanic except 
# in the youngest age groups where they are perhaps more likely to be screened.  Also while people 
# reporting education are more likely to be screened and in order of the amount of education 
# reported, this stratification is not distinct for the lowest age group.
# 
# 

# In[ ]:




