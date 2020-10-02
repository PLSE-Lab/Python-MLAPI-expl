#!/usr/bin/env python
# coding: utf-8

# <img src="https://nleomf.org/wp-content/uploads/2018/11/cropped-memorial-medalion-w-gradiant.jpg" width="650">

# In[ ]:


get_ipython().system(' conda install -y hvplot')
import pandas as pd
import hvplot.pandas


# ## Intro
# 
# This notebook is a basic analysis of police officers killed in the United States. I focus on officers killed in the line of duty by gunfire, stabbing, assault, and vehicular assault. The data comes from FiveThirtyEight's [Database of Police Deaths](https://github.com/fivethirtyeight/data/tree/master/police-deaths) and the [Officer Down Memorial Page](https://www.odmp.org/).

# In[ ]:


deaths = pd.read_csv('../input/police-violence-in-the-us/police_deaths_538.csv')
deaths.cause_short.unique()


# ## Historical context
# 
# The chart below was quite a surprise to me. Police officer deaths in recent times are relatively low compared to times past.
# 
# The 1920s were the most dangerous years on record. This was soon after the end of World War 1 and during the era of [Prohibition](https://en.wikipedia.org/wiki/Prohibition_in_the_United_States). From what I understand it was a period of general lawlessness and open opposition against law enforcement. The population of the US was only 1/3 of what it is now which makes the numbers even more noteworthy.

# In[ ]:


causes = ['Gunfire', 'Stabbed', 'Assault', 'Vehicular assault']
killed = deaths.loc[deaths.cause_short.isin(causes) & (deaths.canine==False)]
killed.groupby('year')['person'].size()         .hvplot.line(width=600, title="Police Officers Killed")


# The other peak period was during the 70's. Like the 1920's, there was a lot of trouble - high crime, racial and social tension, and the Vietnam war.
# 
# 
# 
# 

# ## The 21st Century
# 
# The chart below shows police deaths from shootings between 2001 and 2019. I restricted this to shootings only because the data used so far stops at mid-2016. More current data from the [Officer Down Memorial Page](http://odmp.org) uses a different breakdown of causes. Both sources have shootings as a separate cause making this extension the most consistent.
# 
# 

# In[ ]:


added = pd.Series([64, 45, 52, 48], index=range(2016,2020), name='person')

recent = killed.loc[killed.year.between(2001,2015) &                     (killed.cause_short=="Gunfire")]                 .groupby('year')['person'].size()                 .append(added)

print(f"Average killed: {round(recent.mean(), 0)} \n"
        f"Standard Deviation: {round(recent.std(),0)}")
pk = recent.hvplot.line(width=600, title="Police Officers Killed",
                        ylim=(0,100))
pk


# Given the amount of variation here and the numbers, I see no evidence that things have gotten any better or worse for police over the last 20 years. Note that these numbers are not adjusted for the number of police on duty, number of arrests, crime rate, etc.

# ## Areas
# 
# The chart below shows the departments suffering the biggest losses. Puerto Rico PD seems like the outlier here. The department is slightly smaller than the NYPD and has 3 times the deaths.

# In[ ]:


killed.loc[killed.year.between(2006,2016) & killed.dept.str.contains("Police"), 'dept']                 .value_counts().head(10).hvplot.bar(invert=True, flip_yaxis=True,
                                             title="Officer Deaths 2006-2016", width=600)


# ## Closing
# There is a lot more analysis that could be done by using this data along with other sources. Possibilities include:
# - comparing officer deaths with citizen deaths over time
# - comparing officer deaths with police budgets
