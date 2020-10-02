#!/usr/bin/env python
# coding: utf-8

# Hi everyone,
# 
# Thanks a lot for reading my work! This is my first public project so any constructive criticism is greatly appreicated. Hope you enjoy!

# In[ ]:


#importing
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stat
import os


# 
# New York was the epicenter for the Covid-19 pandamic in the United States during the early parts of 2020 and was subsequently put in one of the strictest quarantine across the country. This makes it the best place to study the effects of quarantine in slowing down the spread of Covid-19.
# 
# Let's start with the exponential model:
# 
# $$ C_{t_{i}} = C_{t_{o}} * e^{(K *(t_{i} - t_{o}))} $$
# 
# where $ C_{t_{i}} $ is the number of cases at time $t_{i}$ and $K$ is the exponential growth constant.
# 
# Covid-19 patients typically show symptoms 7-9 days after infection with the most extreme cases showcasing symptoms after 14 days. To study the effects of the quarantine, we will take a look at the number of cases in new york at the start of the quarantine vs the number of cases 10 days after the start of the quarantine. New York was placed under quarantine on March 20th, 2020, and thus we will be using the data from March 18th to 28th for our growth before quarantine and April 1st to 11th for after quarantine. One could increase the length and thus increase the amount of data we would have. However, this increases the chance that other societal factors can play a role and change the effects of quarantine. 
# 
# To find an accurate measurement of the exponential growth constant, we borrow the subwalk algorithm from molecular dynamics. Briefly, one can imagine that rather than one continous growth measurement, we instead break it down into 9 independent one day growth measurements. We can repeat this for all of the other 2,3,...,9 day time steps. I have implemented a cheat's version of this algorithm which takes in more data, but not all of the data are independent. This effectively increases the data in our fit and increases the accuracy in our determination of the exponential growth constant. 
# 

# In[ ]:


#Loading in the NYC county level data from new york times
us_data=pd.read_csv('../input/uncover/UNCOVER/New_York_Times/covid-19-county-level-data.csv')

#seperation of inital growth data in smaller sub time steps
def exponential_growth(panda):
    days = len(panda.values)
    collection = np.zeros((36,2))

    k = 0
    for i in range(days - 1): 
        inital = panda.values[i]
        data = np.log(panda.iloc[i+1:].values/inital)
        x = np.arange(8-i)+1
        collection[k:k+8-i,0] = x
        collection[k:k+8-i,1] = data
        k += 8-i
    return collection


#determination of exponential growth constant and plots relative change in cases vs time
def New_york(county, data):
    #takes in county name and the data set to pull it from
    
    growth_bq_raw_data = data[(data['county'] == county) & (data['state'] =='New York') & (data['date'] > '2020-03-18') & (data['date'] < '2020-03-28')]['cases']
    growth_bq_exp = exponential_growth(growth_bq_raw_data)
    growth_aq_raw_data = data[(data['county'] == county) & (data['state'] =='New York') & (data['date'] > '2020-04-01') & (data['date'] < '2020-04-11')]['cases']
    growth_aq_exp = exponential_growth(growth_aq_raw_data)

    #scipy linregress to find the optimal growth constant
    bq = stat.linregress(growth_bq_exp[:,0],growth_bq_exp[:,1])
    aq = stat.linregress(growth_aq_exp[:,0],growth_aq_exp[:,1])
    
    #plotting
    x = np.arange(8)+1
    plt.plot(growth_bq_exp[:,0],growth_bq_exp[:,1],'o',label = 'before quarantine (Slope: %0.3f, std: %0.3f)'%(bq.slope,bq.stderr))
    plt.plot(x ,bq.intercept + bq.slope*x)
    plt.plot(growth_aq_exp[:,0],growth_aq_exp[:,1],'^',label = 'after quarantine (Slope %0.3f, std: %0.3f)'%(aq.slope,aq.stderr))
    plt.plot(x ,aq.intercept + aq.slope*x)
    plt.xlabel("T (days)")
    plt.ylabel('$ln(C_{t_{i}}/C_{t_{o}})$')
    plt.legend()


# Now let's take a look at the effects of quarantine in a few counties; New York City, Westchester, Rockland, and Nassau. Please keep in mind that while they vary quite significantly in population density, they are fairly close in geography.

# In[ ]:


plt.figure(1,figsize=(12,12))
plt.subplot(221)
plt.title('New York City')
New_york('New York City',us_data)
plt.subplot(222)
plt.title('Westchester')
New_york('Westchester',us_data)
plt.subplot(223)
plt.title('Rockland')
New_york('Rockland',us_data)
plt.subplot(224)
plt.title('Nassau')
New_york('Nassau',us_data)
plt.show()


# The effects of quarantine in slowing down the spread of Covid-19 is quite apparent. The exponential growth constant, K dropped several folds in all counties once quarantine was enforced. This flattened the infection curve and buys important time for doctors and other health care professionals. Interestingly, K seems relatively similar across all of the counties despite changes in population density. This might be due the geography of these counties and possible of spread across counties. 
# 
# Now lets take a look at less populated county in New York State, Ontario County.

# In[ ]:


New_york('Ontario',us_data)
plt.title('Ontario')
plt.show()


# The growth constant, K, is somewhat comparable to the other counties despite the significant drop in population density. This is slightly surprising, but shows that even in non-dense communities, Covid-19 is extremely infectious and can spread rapidly.
# 
# Finally, lets hypothesize what would happen to New York City if the order for quarantine came a week later. Now here, we aren't considering the effects of testing on the theoretical number of cases. One can imagine that the relative number positive tests would be significantly higher if this occured. This is mainly meant for speculation of what could happen. 

# In[ ]:


def exponential_growth_sim(k,time,C ):
    collection = np.zeros((time))
    for i in np.arange(time):
        collection[i] = C* np.exp(k*i)
    return collection


#Data Prep from previous section
Real_data = us_data[(us_data['county'] == 'New York City') & (us_data['state'] =='New York') & (us_data['date'] > '2020-03-18')& (us_data['date'] < '2020-04-15')]['cases']
N = len(Real_data)
nyc_bq = 0.286
nyc_aq = 0.072
dates = us_data[(us_data['county'] == 'New York City') & (us_data['state'] =='New York') & (us_data['date'] > '2020-03-18')& (us_data['date'] < '2020-04-15')]['date'].values


#Simulation of Delayed quarantine
Sim_data=np.zeros((N))
Sim_data[:16] = exponential_growth_sim(nyc_bq, 16,Real_data.values[0])
Sim_data[15:] = exponential_growth_sim(nyc_aq, N-15,Sim_data[15])

#Simulation of Normal quarantine
Sim_data2=np.zeros((N))
Sim_data2[:10] = exponential_growth_sim(nyc_bq, 10,Real_data.values[0])
Sim_data2[9:] = exponential_growth_sim(nyc_aq, N-9,Sim_data2[9])


#plotting the figure
plt.figure(1,figsize=(8,8))
plt.plot(np.arange(N),Real_data.values,'o-',label='Actual NYC')
plt.plot(np.arange(N),Sim_data,'o-',label='Simulated NYC (Delayed Quarantine)')
plt.plot(np.arange(N),Sim_data2,'o-',label='Simulated NYC')
plt.vlines(2,0,400000,'r','--',label='Quarantine Start Date')
plt.vlines(9,0,400000,'b','--',label='Delayed Quarantine Start')
plt.title('Delayed Quarantine')
plt.ylabel('Cases')

plt.legend()
plt.xticks(np.arange(N),dates,rotation=70,fontsize=8)
plt.show()


# Using the same exponential growth constant from our previous estimate, delaying the quarantine a week would significantly increase the number of cases by several folds by the middle of April. Here we show the importance of early intervention and how even delaying intervention responses by a single week can dramatically change number of cases. Early intervention is one of the best tools in stopping a pandemic and subsequently flatten the infection curve buying valueable time for the medical industry to catch up.
# 
# One can argue that I was too conservative in my choice of dates and the exponential growth a week before the quarantine started was significantly worse. They may be correct in that assertion, but I would also argue that when the number of cases are low, significant increases in testing and awareness play a bigger role in dramatical increase in cases. I would rather trend on the side of caution and select dates in which we can be sure that awareness and effects towards testing are at an all time high.
# 
# Hopefully, you have enjoyed my work, and I would love to hear about ideas to improve and change up the model. Certainly, one can make better models by using more assumptions in growth distribution and decay. However, I find it interesting how even a simple model like exponential growth can tell us a lot about the effects of quarantine and lack of quarantine.
# 
# Thanks for reading!
