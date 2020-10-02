#!/usr/bin/env python
# coding: utf-8

# Full disclosure: I'm not familiar enough with the health marketplace data to draw any conclusions from my analysis. I am interested in healthcare data in general and I decided to use this dataset as a way to learn more about using Python (I'm more comfortable in SAS and R). And away we go!
# First we'll import the appropriate libraries and import some data. I'm going to take a look at the rates file and will specifically be looking at just a few columns (State, IndividualRate, and PrimarySubscriberAndOneDependent).
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
from  matplotlib import pyplot

fields = ['StateCode','IndividualRate','PrimarySubscriberAndOneDependent','BusinessYear']

csv_chunks = pd.read_csv("../input/Rate.csv",iterator=True,chunksize = 1000,usecols=fields)
rates = pd.concat(chunk for chunk in csv_chunks)


# Now that our data is loaded, I'll pair it down to 2016 and clear out the weird data (with rates over 9,000). Thankfully I've seen some notes from Ben and others pointing out this data, so I didn't have to find it myself. Also, I'm going to be comparing the individual rates vs. those with one dependent, so I'll filter out those with a null in the one dependent column. Then we'll take a quick look at the data.

# In[ ]:


rates = rates[np.isfinite(rates['PrimarySubscriberAndOneDependent'])]
rates = rates[rates.IndividualRate <9000]
rates = rates[rates.BusinessYear == 2016]

rates.head(n=5)


# In[ ]:


print(rates.describe())


# In[ ]:


import matplotlib.pyplot as plt

##Individual histogram
plt.hist(rates.IndividualRate.values)


# In[ ]:


##Remove records with 0 as PrimarySubscriberAndOneDependent
rates = rates[rates.PrimarySubscriberAndOneDependent > 0]

##OneDependent Histogram
plt.hist(rates.PrimarySubscriberAndOneDependent.values)


# Finally, I want to look at the data aggregated (I'll use median). Then I'd like to look at the ratio between the expense for One dependent vs. Individual.

# In[ ]:


## Group data by state (using Median)
rateMed = rates.groupby('StateCode', as_index=False).median()
del rateMed['BusinessYear']



## JointPlot of grouped data

plt = sns.jointplot(x="IndividualRate", y="PrimarySubscriberAndOneDependent", data=rateMed)
sns.plt.show()


# In[ ]:


## Calculate the ratio
rateMed['ratio'] = rateMed['PrimarySubscriberAndOneDependent']/rateMed['IndividualRate']
rateMed.sort(['ratio'], ascending=[0])


# In[ ]:


plt = sns.barplot(rateMed.sort(['ratio'], ascending=[0]).StateCode, rateMed.ratio,palette="Blues")
sns.plt.show()


# As the table and chart show, 7 states (UT, WV, TN, AK, IL, VA, and IN) appear charge more than 2X the individual rate for those with the subscriber and one dependent.
