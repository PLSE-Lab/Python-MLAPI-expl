#!/usr/bin/env python
# coding: utf-8

# ## Purpose
# My attempt to find yet another way to estimate the true number of COVID infections in your area, this time by scaling the infections according to the testing rate.
# 
# ## Introduction
# That the real number of people infected with COVID is likely higher than the reported number of infections has been explained in many excellent articles (for example: [here](https://fivethirtyeight.com/features/coronavirus-case-counts-are-meaningless/). In lieu of good testing, those skilled in disease modelling are able to come up with some very impressive models. Those that are not pandemic experts but are curious anyways have come up with some data-driven ways of estimating the real number of infections. Some of these include mortality (which is about 1% of infections if the healthcare system is not overwhelemed) and ICU (which is about 5%). 
# 
# Here I present another way of estimating based on the reported number of tests being conducted.
# 
# # Summary
# Estimating real infection numbers based on tests could be about as accurate as estimations based on deaths, with the additional benefit of having less of a time-delay than deaths-based estimations.
# 
# According to my estimates, in Ontario there could be close to 50,000 infections by April 19, almost 5 times the reported number by the province.
# 
# Similar to the reported results, the daily new cases seemed to have plateaued, at around 2000 new cases per day, around 4 times more than the 500-600 new cases per day reported by the province.**
# 
# You are welcome to change this code to analyze another region that you are concerned about.
# 
# # Procedure
# Import relevant modules and read in data

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import matplotlib.ticker as plticker  #ticker control
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df1 = pd.read_csv('/kaggle/input/covid19testing/tested_worldwide.csv', delimiter=',') 
df1.dataframeName = 'tested_worldwide.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.head(5)


# Narrow it down to Canada:

# In[ ]:


df1[df1.Country_Region=='Canada']


# The ideal place to try-out this new method is in a place where not enough testing is being done.  [Ontario seems a good place to start](https://www.theglobeandmail.com/canada/article-doug-ford-calls-ontarios-low-testing-rate-unacceptable/). As an added bonus, despite the low tests, its healthcare system appears to be [in good shape](https://nationalpost.com/news/canada/that-is-a-surprise-doctors-still-waiting-for-feared-surge-of-covid-19-patients-in-canadian-icus) which allows me to cross-check with the estimate assuming mortality rate is 1% of all infected persons.
# 
# Let's examine the testing data from Ontario.
# 
# I will define the rate of daily positive tests. It is simply 
# \begin{equation}
# daily\:positive\:test\:rate = \frac{number\:of\:daily\:new\:infections}{number\:of\:daily\:new\:tests}
# \end{equation}

# In[ ]:


df1_on=df1[df1.Province_State=='Ontario'].dropna(subset=['daily_tested'])
df1_on=df1_on.assign(daily_rate=df1_on.daily_positive/df1_on.daily_tested*100)
#df1_on['daily_pos_rate']=df1_on.daily_positive/df1_on.daily_tested*100
df1_on.daily_rate.replace(np.inf, 0, inplace=True)
df1_on.daily_rate.fillna(0, inplace=True)
df1_on.reset_index(drop=True, inplace=True)
df1_on.head()


# Let's see first how testing has changed over time:

# In[ ]:


fig, ax = plt.subplots()
ax.plot(df1_on.Date, df1_on.daily_tested,'x-')
plt.title('Daily Number of Tests')
plt.xticks(rotation=90)
loc = plticker.MultipleLocator(base=5.0) 
ax.xaxis.set_major_locator(loc)
plt.show()


# The testing isn't consistent, but at least the overall trend line is up.
# 
# But the question is: is testing keep up with the increase in infections? If it is, then the rate of daily positive tests should stay flat. However, from the results below, the answer seems to be no. Towards the end of March, the percentage of tests that returned positive skyrocketed

# In[ ]:


fig, ax = plt.subplots()
ax.plot(df1_on.Date, df1_on.daily_rate,'x-')
plt.title('Daily Positive Test Rate')
plt.xticks(rotation=90)
loc = plticker.MultipleLocator(base=5.0) 
ax.xaxis.set_major_locator(loc)
plt.show()


# Recently, South Korea was often mentioned as an example of effective pandemic control through mass testing. Their postive test rate was between 2-3% of all tests. 
# 
# Ontario had a similar rate up until 3rd week of March, before positive test rates shot up. As we can see below, up until the 3rd week of March, positive test rate was stable around the 2% mark.
# 
# Let's set the average positive test rate from March 15 to March 25 as the baseline positive test rate. If testing was ramped up such that the positive test rate was maintained at the baseline rate, then the testing results would paint the correct picture of the pandemic's progress for us.

# In[ ]:


base_pct=df1_on['daily_rate'].iloc[19:30].mean() # calculate baseline
base_pct


# If testing was inadequate, I can scale up the reported positive cases to reflect what I think is the actual number by defining the scale factor.
# 
# To keep things simple, I assumed a linear relationship between the positive test rate and the scale factor. Say in tomorrow's batch of data, the positive test rate doubled from 2% (the baseline) to 4% and 50 people were reported as infected, then I will double the number of infected to 100 as my estimate of actual number of people infected. If instead the positive test rate tripled from 2% (the baseline) to 6% and 50 people were reported as infected, then I will triple the number of infected to 150 as my estimate of actual number of people infected.
# 
# \begin{equation}
# Scale\:Factor = \frac{Daily\:Positive\:Rate}{Baseline\:Positive\:Rate}
# \end{equation}
#  
# \begin{equation}
# True\:Infection=Reported\:Infection \times Scale\:Factor
# \end{equation}
# 
# Admittedly this is a big assumption. Further improvements can be made here

# In[ ]:


df1_on=df1_on.assign(scale_factor=df1_on.daily_rate/base_pct) # define scale factor
df1_on.drop(df1_on.index[range(19)],inplace=True)  #drop the first 19 rows, where no significant testing took place
df1_on.head()


# In this way I can come up with my own estimates of how many people are likely infected, both daily and in total:

# In[ ]:


df1_on=df1_on.assign(daily_positive_exp=round(df1_on.daily_positive*df1_on.scale_factor), positive_exp=np.nan) #expected daily positive cases
df1_on.loc[19,'positive_exp']=df1_on.loc[19,'positive']
for i  in range(1,len(df1_on)):
    df1_on.loc[19+i,'positive_exp']=df1_on.loc[18+i,'positive_exp']+df1_on.loc[19+i, 'daily_positive_exp'] #expected total positive cases
df1_on


# I am going to cross-check my results later using infection estimation based on mortality. That is, we assume mortality rate is around 1% of total number of infected to estimate the total number of infected people.

# In[ ]:


df1_on=df1_on.assign(positive_exp2=df1_on.death*100,daily_positive_exp2=np.nan) #expected total positive cases based on deaths
df1_on.loc[19,'daily_positive_exp2']=0
for i  in range(1,len(df1_on)):
    df1_on.loc[19+i,'daily_positive_exp2']=100*(df1_on.loc[19+i,'death']-df1_on.loc[18+i, 'death']) #expected daily positive cases


# # Results
# Let's compare the estimated infection based on tests vs. the reported number

# In[ ]:


fig, ax = plt.subplots()
ax.plot(df1_on.Date, df1_on.daily_positive,'x-', label='reported')
ax.plot(df1_on.Date, df1_on.daily_positive_exp,'x-', label='estimate from tests')
plt.title('Daily New Infections')
plt.xticks(rotation=90)
loc = plticker.MultipleLocator(base=5.0) 
ax.xaxis.set_major_locator(loc)
plt.legend()
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.plot(df1_on.Date, df1_on.positive,'x-', label='reported')
ax.plot(df1_on.Date, df1_on.positive_exp,'x-', label='estimate from tests')
plt.title('Total Infections')
plt.xticks(rotation=90)
loc = plticker.MultipleLocator(base=5.0) 
ax.xaxis.set_major_locator(loc)
plt.legend()
plt.show()


# Yikes, not good. Estimates suggest that there were a few days when at least 3000 people got infected, even though reported new cases plateaued at around 500-600. As a result, there could be close to 50,000 people in Ontario already infected at the time of this writing, almost 5 times the reported number of cases. On the other hand, it does appear the number of daily new cases have plateaued at around 2000 people per day.
# 
# Finally, to verify if my methods yield numbers that are at least in the ball-park, I am going to compare it with the estimates based on mortality
# 

# In[ ]:


fig, ax = plt.subplots()
ax.plot(df1_on.Date, df1_on.daily_positive_exp2,'x-', label='estimate from deaths')
ax.plot(df1_on.Date, df1_on.daily_positive_exp,'x-', label='estimate from tests')
plt.title('Daily New Infections')
plt.xticks(rotation=90)
loc = plticker.MultipleLocator(base=5.0) 
ax.xaxis.set_major_locator(loc)
plt.legend()
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.plot(df1_on.Date, df1_on.positive_exp2,'x-', label='estimate from deaths')
ax.plot(df1_on.Date, df1_on.positive_exp,'x-', label='estimate from tests')
plt.title('Total Infections')
plt.xticks(rotation=90)
loc = plticker.MultipleLocator(base=5.0) 
ax.xaxis.set_major_locator(loc)
plt.legend()
plt.show()


# These results show that the numbers are approximately in-line with estimates based on deaths, confirming the validity of this test-based approach. 
# 
# One interesting behaviour is how the death-based estimate lags behind test-based estimate. This is expected because it would take some time before deaths due to a disease start to appear, while infections would appear right away. From the graph below. The estimates due to deaths seem to be delayed by 3 days.
# 
# This is an advantage of test-based estimate over the mortality-based estimate because the results are more real-time than the mortality-based estimate.

# In[ ]:


fig, ax = plt.subplots()
ax.plot(df1_on.index,df1_on.positive_exp2,'x-', label='estimate from deaths')
ax.plot(df1_on.index+3,df1_on.positive_exp,'x-', label='estimate from tests') #delay the results by 3 days
plt.title('Total Infections')
plt.xticks(rotation=90)
loc = plticker.MultipleLocator(base=5.0) 
ax.xaxis.set_major_locator(loc)
plt.legend()
plt.show()


# Finally, the 2 estimates started to diverge by the middle of April. It's not clear to me what is the cause
