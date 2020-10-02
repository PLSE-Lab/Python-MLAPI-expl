#!/usr/bin/env python
# coding: utf-8

# ## U.S. Educational Finance Data Exploration (Idaho)

# A brief scan of U.S. financial data, broken down by state and year. Source is the [U.S. Census Bureau](https://www.census.gov/programs-surveys/school-finances/data/tables.html), [programatically organized by me](https://www.kaggle.com/noriuk/us-educational-finances). The emphasis for this exploration is the state of Idaho (my home). It's financial patterns are compared to the national median across a span of 23 years (1992 to 2015). 
# 
# Data is organized by state and year. Entries include revenue by source, expenditure by source, and enrollment. 

# In[1]:


# Import data analysis tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Matplotlib Style
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

# Matplotlib axis formats
import matplotlib.ticker as mtick
dfmt = '${x:,.0f}'
dtick = mtick.StrMethodFormatter(dfmt)
fmt = '{x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)


# In[5]:


# Import the data
st_summaries = pd.read_csv('../input/elsect_summary.csv')


# In[4]:


# Preview the data
# (Some NaN's expected for 1992)
st_summaries.tail()


# In[5]:


# Idaho Data
idaho_data = st_summaries.loc[st_summaries['STATE'] == 'Idaho']
idaho_data.tail()


# In[6]:


# National Median Data

# Determine median values for all columns, for a given year
def mdn_by_year(year):
    year_data = st_summaries.loc[st_summaries['YEAR'] == year]
    year_mdn = {}
    for key in year_data.columns.values:
        if 'STATE' != key:
            year_mdn[key] = year_data[key].median()
    year_mdn['STATE'] = 'Median'
    return year_mdn

# Build the "median state"
years = range(1992,2016)
mdn_state = []
for year in years:
    year_mdn = mdn_by_year(year)
    mdn_state.append(year_mdn)

mdn_data = pd.DataFrame(mdn_state, columns=idaho_data.columns.values)
mdn_data.tail()


# ### Financial Data Comparisions
# 
# Idaho is a small, rural state with less than two million residents. We should expect it to have lower values than the national median for most things. 

# In[8]:


# Big picture finances; revenue vs. expenditure
def plot_rev_and_expend(data):
    plt.plot(data['YEAR'], data['TOTAL_REVENUE'], color='k')
    plt.plot(data['YEAR'], data['TOTAL_EXPENDITURE'], color='r')
    plt.gca().yaxis.set_major_formatter(dtick)
    
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Revenue vs. Expenditure')
plot_rev_and_expend(idaho_data)
plt.subplot(1, 2, 2)
plot_rev_and_expend(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()


# Revenue and expenditure seem to keep pace with each other over the years for both Idaho and the median state. The magnitude is considerably larger for the median (~\$5,000,000 median increase vs. ~\$1,400,000 Idaho increase) though the percentage increase is comprable (167% vs. 175%).
# 
# A point in Idaho's favor is that it manages to keep revenue above expenses from 2005 onwards.

# In[9]:


# Revenue breakdown
def plot_revenue_breakdown(data):
    plt.plot(data['YEAR'], data['TOTAL_REVENUE'], color='k')
    plt.plot(data['YEAR'], data['STATE_REVENUE'])
    plt.plot(data['YEAR'], data['FEDERAL_REVENUE'])
    plt.plot(data['YEAR'], data['LOCAL_REVENUE'])
    plt.gca().yaxis.set_major_formatter(dtick)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Revenue Breakdown')
plot_revenue_breakdown(idaho_data)
plt.subplot(1, 2, 2)
plot_revenue_breakdown(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()


# In[10]:


# Revenue Percentages
def plot_revenue_percent(data):
    data['STATE_PERCENTAGE'] = data['STATE_REVENUE'] / data['TOTAL_REVENUE']
    data['FEDERAL_PERCENTAGE'] = data['FEDERAL_REVENUE'] / data['TOTAL_REVENUE']
    data['LOCAL_PERCENTAGE'] = data['LOCAL_REVENUE'] / data['TOTAL_REVENUE']
    plt.plot(data['YEAR'], data['STATE_PERCENTAGE'])
    plt.plot(data['YEAR'], data['FEDERAL_PERCENTAGE'])
    plt.plot(data['YEAR'], data['LOCAL_PERCENTAGE'])

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Revenue Percentage Breakdown')
plot_revenue_percent(idaho_data)
plt.subplot(1, 2, 2)
plot_revenue_percent(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()


# Idaho's revenue stems primarily from the state, followed by local sources, then federal sources. This pattern is mirrored in the median data but to very different extents. Idaho trends demonstrate a primacy in state funding, more than 10% higher on average than the median (~60% Idaho vs. ~50% median). Local funding is likewise 10% lower (~28% Idaho vs. ~40% median), while federal funding is roughly the same (10%).

# In[15]:


# Expenditure breakdown
def plot_expenditure_breakdown(data):
    plt.plot(data['YEAR'], data['TOTAL_EXPENDITURE'], color='r')
    plt.plot(data['YEAR'], data['INSTRUCTION_EXPENDITURE'])
    plt.plot(data['YEAR'], data['SUPPORT_SERVICES_EXPENDITURE'])
    plt.plot(data['YEAR'], data['CAPITAL_OUTLAY_EXPENDITURE'])
    plt.plot(data['YEAR'], data['OTHER_EXPENDITURE'])
    plt.gca().yaxis.set_major_formatter(dtick)


plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Expenditure Breakdown')
plot_expenditure_breakdown(idaho_data)
plt.subplot(1, 2, 2)
plot_expenditure_breakdown(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()


# In[17]:


# Expenditure Percentages
def plot_expenditure_percent(data):
    data['INSTRUCTION_PERCENTAGE'] = data['INSTRUCTION_EXPENDITURE'] / data['TOTAL_EXPENDITURE']
    data['SUPPORT_PERCENTAGE'] = data['SUPPORT_SERVICES_EXPENDITURE'] / data['TOTAL_EXPENDITURE']
    data['OUTLAY_PERCENTAGE'] = data['CAPITAL_OUTLAY_EXPENDITURE'] / data['TOTAL_EXPENDITURE']
    data['OTHER_PERCENTAGE'] = data['OTHER_EXPENDITURE'] / data['TOTAL_EXPENDITURE']
    plt.plot(data['YEAR'], data['INSTRUCTION_PERCENTAGE'])
    plt.plot(data['YEAR'], data['SUPPORT_PERCENTAGE'])
    plt.plot(data['YEAR'], data['OUTLAY_PERCENTAGE'])
    plt.plot(data['YEAR'], data['OTHER_PERCENTAGE'])

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Expenditure Percentage Breakdown')
plot_expenditure_percent(idaho_data)
plt.subplot(1, 2, 2)
plot_expenditure_percent(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()


# The Idaho data and median data are in line for expenditures. Instruction costs represent the majority of expenditure (between 52% and 57%). Support services are next in line, constituting adminstrators, transportation, and other similar functionaries (~30%). Capital Outlay (fixed-asset expenses like building renovations) comprise 10%, while the remaining Other expenses produce a minor 5%.
# 
# Idaho's Capital Outlay begins a noticable drop in 2005 for unknown reasons.

# In[13]:


# State Enrollment
def plot_enroll(data):
    plt.plot(data['YEAR'], data['ENROLL'])
    plt.gca().yaxis.set_major_formatter(tick)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Enrollment')
plot_enroll(idaho_data)
plt.subplot(1, 2, 2)
plot_enroll(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()


# In[14]:


# Expenditure per Student
def plot_expenditure_enroll(data):
    data['EXPENDITURE_PER_STUDENT'] = data['TOTAL_EXPENDITURE'] / data['ENROLL']
    plt.plot(data['YEAR'], data['EXPENDITURE_PER_STUDENT'])
    
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Expenditure per Student')
plot_expenditure_enroll(idaho_data)
plt.subplot(1, 2, 2)
plot_expenditure_enroll(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()


# Enrollment data is one area where median values may not have been the best choice. There is a 100,000 student spike in 1995 for the median state that is difficult to explain. It could be a sign that there is a rigid stratification between the states (populous vs. sparsely populated). Or, it may hint at a policy change of some sort, perhaps one that alters the way student enrollment is counted. Whatever the case, it is not present in the Idaho data.
# 
# Remarkably, expenditure per student for Idaho outpaces the increase in enrollment. Idaho goes from 230,000 students to 280,000 students and manages to raise it's expenditure per student from \$4 a student to \$7.50 a student. *(Admittedly, my calculation for "expenditure per student" is pretty barebones).* Median per student expenditure  effectively doubles from 1992 (as does Idaho's). The median dollar amount is somewhat higher than Idaho's (\$6-12 vs. \$4-8).

# ### Investigating State Stratification
# 
# It would've behooved me to do this before the previous analysis, but some of the findings from this survey suggest that the data may not be conveniently spread. It's possible that the states cluster around certain areas. Maybe states with a high population are *very high*, while states with a low population are *very low*. This wouldn't be reflected in the median state.

# In[35]:


# State enrollment by year
plt.figure(figsize=(10,5))
plt.title('State Enrollment by Year')
plt.scatter(st_summaries['YEAR'], st_summaries['ENROLL'], alpha=0.5)
plt.show()


# In[36]:


# Zoom and enchance
plt.figure(figsize=(10,5))
plt.title('State Enrollment by Year (Zoomed-In)')
plt.scatter(st_summaries['YEAR'], st_summaries['ENROLL'], alpha=0.5)
plt.ylim((0,2000000))
plt.show()


# In[43]:


data_1995 = st_summaries.loc[st_summaries['YEAR'] == 1995]
data_2005 = st_summaries.loc[st_summaries['YEAR'] == 2005]
data_2015 = st_summaries.loc[st_summaries['YEAR'] == 2015]

plt.figure(figsize=(10,5))
plt.subplot(1, 3, 1)
plt.violinplot(data_1995['ENROLL'])
plt.ylim((0,5000000))
plt.subplot(1, 3, 2)
plt.violinplot(data_2005['ENROLL'])
plt.ylim((0,5000000))
plt.gca().yaxis.set_ticks([])
plt.subplot(1, 3, 3)
plt.violinplot(data_2015['ENROLL'])
plt.ylim((0,5000000))
plt.gca().yaxis.set_ticks([])
plt.show()


# The distribution of data does not appear to be bimodal as feared. It seems to exist on a reasonable gradient with most states having an enrollment size between 500,000 and 2,000,000. Enrollment grows steadily across the time frame. 

# ## Conclusions
# 
# Idaho more-or-less follows the national trends for education finances. It's revenue, expenditure, and enrollment all see comparable increases from 1992 to 2015. Predictably, Idaho's values are lower than those of the national median though it's percentage changes are roughly the same.

# ### Code Citations
# 
# * [Matplotlib dollar format](https://stackoverflow.com/questions/38152356/matplotlib-dollar-sign-with-thousands-comma-tick-labels) courtesy of StackOverFlow user Alberto Garcia-Raboso
# * [Seaborn Correlation Heatmap](https://seaborn.pydata.org/examples/many_pairwise_correlations.html) from the documentation

# In[ ]:




