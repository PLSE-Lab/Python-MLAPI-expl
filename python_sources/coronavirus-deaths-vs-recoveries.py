#!/usr/bin/env python
# coding: utf-8

# I want to start with a HUGE caveat -- I"m a data scientist with a background in Physics. I know nothing about biology or diseases. What I have put in this kernel is to satisfy my own curiousity and I'm making the kernal public in case others are interested.  I'd love feedback from pythonistas or disease experts on the code and/or the assumptions.  

# I have been curious about the mortality rate for the coronavirus.  I know the official rate is the number of deaths divided by the number of confirmed cases. But I keep thinking that since people who are still sick may yet die, it seems like # deaths/# cases would be an underestimate of the true death rate -- someone who was just confirmed with the virus has awhile to go before they're safe. However, if you look at the number of deaths vs the number of recoveries, that is frightening close to even, though the number of recovered patients is growing much faster than the number of deaths.  So I wanted to play a bit with delays between the different variables.  

# Caveat number 2 -- this is the first kernel I've created and submitted, so apologies if it's a bit amateurish.  

# In[ ]:


#Using the standard python 3 kernel set up.  Added matplotlib, pandas plotting and date/datetime

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime
from datetime import date

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Read in the data

file = '/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv'
df = pd.read_csv(file)
print(df.head())
print(df.tail())


# In[ ]:


#Create a data frame grouped by Date for confirmed, deaths, recovered
mort_df = df.groupby('Date').Confirmed.sum()
mort_df = pd.concat([mort_df, df.groupby('Date').Deaths.sum(), df.groupby('Date').Recovered.sum()], axis=1)
mort_df = mort_df.sort_values(by='Date').reset_index()

#convert Date do datetime
mort_df.Date =  [datetime.strptime(x, "%m/%d/%Y %H:%M:%S").date() for x in mort_df.Date]
#create a variable that keeps track of how many days ago the data is from
mort_df["daysago"] = [-(date.today() - x).days for x in mort_df.Date]
mort_df


# Now I create a couple columns for CFR (case fatality rate -- the official mortality rate) and what I'm calling FC or "final count" where final means that the person has either recovered or died -- so no more change is possible.  
# 
# And now we plot those two columns vs time delta from current day.

# In[ ]:


mort_df["fc"] = [100.0*int(x)/(int(x)+int(y)) if int(x)>0 else 0 for (x, y) in zip(mort_df.Deaths, mort_df.Recovered)]
mort_df["cfr"] = [100.0*int(x)/int(y) if int(x)>0 else 0 for (x, y) in zip(mort_df.Deaths, mort_df.Confirmed)]

plt.plot(mort_df.daysago[1:], mort_df.fc[1:], 'o') #using [1:] to skip the first day which had zeros
plt.xlabel("time delta from today")
plt.title('Final count and CFR vs time')
plt.ylabel('%')
plt.ylim([0, 65])
plt.plot(mort_df.daysago[1:], mort_df.cfr[1:], 'o')
plt.show()

So, the actually "death rate" is somewhere between the above two curves, and we can see that the dead + recovered curve has a nice, reassuring negative slope and will most likely asymptotically approach the lower line as time goes on (important note -- these numbers are only for those sick enough to be tested!  Mild, unreported cases would reduce both numbers considerably!!)

Now, if someone gets really sick, but doesn't die, it will take them a few days to be in the "recovered" category, so we should assume some number of days delay between the "dead" and "recovered" curves.  Let's start with assuming 5 days delay.
# In[ ]:


def fc_add_delay(dead, recovered, daysago, delay):
    """function to add a delay to the recovered number before comparing to dead"""
    mort_fc = []
    timestep = []
    for i in range(1, len(dead) - delay):
        mort_fc.append(100*dead[i]/(dead[i] + recovered[delay+i]))
        timestep.append(daysago[i]+delay)
    return(mort_fc, timestep)
        


# In[ ]:


(mort_fc, timestep) = fc_add_delay(mort_df.Deaths, mort_df.Recovered, mort_df.daysago, 5)


# In[ ]:


plt.plot(timestep, mort_fc)
plt.xlabel("time delta from today")
plt.title('Final count vs time with 5 day delay for recovered')
plt.ylabel('%')


# So this has  the same shape as above, but only peaks at 24%
# 
# Playing with different delays (below), we shorter delays causes a sharper peak/drop, but longer delays causes a flattening. This will be interesting to revisit as time goes on

# In[ ]:


(mort_fc, timestep) = fc_add_delay(mort_df.Deaths, mort_df.Recovered, mort_df.daysago, 3)
plt.plot(timestep, mort_fc)
plt.xlabel("time delta from today")

plt.title('Final count vs time with 3 day delay for recovered')
plt.ylabel('%')


# In[ ]:


(mort_fc, timestep) = fc_add_delay(mort_df.Deaths, mort_df.Recovered, mort_df.daysago, 7)
plt.plot(timestep, mort_fc)
plt.xlabel("time delta from today")
plt.title('Final count vs time with 7 day delay for recovered')
plt.ylabel('%')


# Looking at a range of delays from 1 to 10 we see some flattening for longer delays and a 9 or 10 day delay has a positive slope. Something around 7 or 8 days may make the most sense.  

# In[ ]:


for i in range(10):
    (mort_fc, timestep) = fc_add_delay(mort_df.Deaths, mort_df.Recovered, mort_df.daysago, i)
    plt.plot(timestep, mort_fc)


# We can do the same thing with confirmed vs deaths, though in this case we delay the deaths relative to confirmed cases (someone who has just died was probable confirmed several days ago) -- also, given that it must take several days to get sick enough to die, our time series isn't really long enough yet.  But we do find that assuming longer times increases the mortality rate.  We also see that (at least with this length of time series), the delayed CFR is not stable but decreasing strongly with time, though the 3 day delay may be flatting at the end.

# In[ ]:


def cfr_add_delay(confirmed, dead, daysago, delay):
    mort_cfr = []
    timestep = []

    for i in range(1, len(confirmed) - delay):

        mort_cfr.append(100*dead[i+delay]/confirmed[i])
        timestep.append(daysago[i]+delay)
    return(mort_cfr, timestep)
        


# In[ ]:


(mort_cfr, timestep) = cfr_add_delay(mort_df.Confirmed, mort_df.Deaths, mort_df.daysago, 0)
plt.plot(timestep, mort_cfr)
plt.xlabel("time delta from today")
plt.title('CFR vs time with death count with no delay')
plt.ylabel('%')


# In[ ]:


(mort_cfr, timestep) = cfr_add_delay(mort_df.Confirmed, mort_df.Deaths, mort_df.daysago, 3)
plt.plot(timestep, mort_cfr)
plt.xlabel("time delta from today")
plt.title('CFR vs time with death count delayed by 3 days')
plt.ylabel('%')
    


# In[ ]:


(mort_cfr, timestep) = cfr_add_delay(mort_df.Confirmed, mort_df.Deaths, mort_df.daysago, 7)
plt.plot(timestep, mort_cfr)
plt.xlabel("time delta from today")
plt.title('Final count vs time with death count delayed by 7 days')
plt.ylabel('%')


# In[ ]:





# Plotting delays from 0 to 10 days shows a that the 0 day delay is significantly flatter and longer delays leads to a sharp slope with time. This gives some support to using the current CFR method of # deaths / # confirmed (in short -- listen to the experts). 

# In[ ]:


for i in range(10):
    (mort_cfr, timestep) = cfr_add_delay(mort_df.Confirmed, mort_df.Deaths, mort_df.daysago, i)
    plt.plot(timestep, mort_cfr)


# In[ ]:





# In any case, it was very interesting to use this data and answer some questions I've had. Thanks for uploading it!

# In[ ]:




