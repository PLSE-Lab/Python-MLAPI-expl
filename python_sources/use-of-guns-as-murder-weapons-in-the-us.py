#!/usr/bin/env python
# coding: utf-8

# Hi everyone! This notebook investigates the use of guns a little bit.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar


# In[ ]:


#We are only intereted in murder and manslaughter, so we discard the rest.
mm=pd.read_csv('../input/database.csv',low_memory=False).groupby('Crime Type').get_group('Murder or Manslaughter')


# First we check, which weapons are used most often based on gender of the perpetrator.
# Weapons which are used in less than 5 percent of cases are joined as others.

# In[ ]:


threshold=0.05
weaponCountM= mm[mm['Perpetrator Sex']=='Male']['Weapon'].value_counts(normalize=True).sort_values()
weaponCountF= mm[mm['Perpetrator Sex']=='Female']['Weapon'].value_counts(normalize=True).sort_values()
weaponCountM['Other']=weaponCountM[weaponCountM<threshold].sum()
weaponCountF['Other']=weaponCountF[weaponCountF<threshold].sum()
plt.figure()
plt.pie(weaponCountM[weaponCountM>threshold].values,labels=weaponCountM[weaponCountM>threshold].index)
plt.title('Weapons used by male Perpetrators')
plt.show()
plt.figure()
plt.pie(weaponCountF[weaponCountF>threshold].values,labels=weaponCountF[weaponCountF>threshold].index)
plt.title('Weapons used by female Perpetrators')
plt.show()


# We see that most perpetrators use handguns as weapon, followed by Knifes and blunt objects.
# Males also use shotguns or other firearms in many cases. Females use a wider range of weapons, so they get a bigger piece of "other".

# **How did the use of guns as murder weapons develop over the years?**
# We look at the relative usage of handguns and all firearms in the filed cases.

# In[ ]:


weaponsByYear=mm.groupby('Year')['Weapon'].value_counts(normalize=True)
#To get all firearms we have to include the tags Rifle, Shotgun, Gun and Firearm
GunsByYear=weaponsByYear.loc[:,'Handgun',:]+weaponsByYear.loc[:,'Rifle',:]+weaponsByYear.loc[:,'Shotgun',:]+weaponsByYear.loc[:,'Firearm',:]+weaponsByYear.loc[:,'Gun',:]

f,ax=plt.subplots(1)
weaponsByYear.loc[:,'Handgun',:].plot(ax=ax,label='Handguns')
GunsByYear.plot(ax=ax,label='All guns')
ax2=ax.twinx()
mm.groupby('Year').size().plot(ax=ax2,color='orange',label='Number of cases')
ax2.grid(False)
ax.legend(loc=0)
ax2.legend(loc=1)
plt.show()


# We see that there was a sharp rise in the use of guns starting at 1986 with a peak in 1994. This coincides with a sharp rise in the number of murder cases. While the number of murders cases decreased steadily in the U.S., the proportion of firearms in general used as murder weapons seems to be stable.

# **Does the proportion of guns used as murder weapons depend on the state and its politics about guns?**
# To get a feeling for the answer we look at the proportion based on the state. Furthermore we mark the 10 states
# with the most and the least strictest gun laws according to these sites:
# 
# http://www.deseretnews.com/top/1428/0/10-states-with-the-strictest-gun-laws.html 
# http://www.deseretnews.com/top/3430/0/The-10-states-with-the-least-restrictive-gun-laws.html

# In[ ]:


#group the data
weaponCountStates= mm.groupby('State')['Weapon'].value_counts(normalize=True).sort_index()
wS=weaponCountStates.loc[:,'Handgun',:]+weaponCountStates.loc[:,'Rifle',:]+weaponCountStates.loc[:,'Shotgun',:]+weaponCountStates.loc[:,'Firearm',:]+weaponCountStates.loc[:,'Gun',:]
#in some states the term 'gun' is not used, this results in missing values if we add them up
wSnoGun=weaponCountStates.loc[:,'Handgun',:]+weaponCountStates.loc[:,'Rifle',:]+weaponCountStates.loc[:,'Shotgun',:]+weaponCountStates.loc[:,'Firearm',:]
for state in ['Maine','North Dakota','South Dakota','Vermont']:
    wS.loc[state]=wSnoGun.loc[state]

restrictive=['California','New Jersey','Massachusetts','New York','Connecticut','Hawaii','Maryland','Rhodes Island','Illnoise','Pennsylvania']
loose=['Louisiana','Mississippi','Arizona','Kentucky','Wyoming','Missouri','Alaska','South Dakota','Vermont','Kansas']
plt.figure(figsize=(12,12))
ax=sns.barplot(y=wS.sort_values(ascending=False).index,x=wS.sort_values(ascending=False).values)
for a in ax.get_yticklabels():
    if a.get_text() in restrictive:
        a.set_color('green')
    if a.get_text() in loose:
        a.set_color('orange')
plt.show()


# So if you don't want to get shot, Hawaii seems like a nice place to live ;)
# 
# Gun regulations seem to matter:
# The top state has also the least strict rules.
# The last state has quite strict rules.
# 
# Of course there are also states with strict rules but murderers prefer guns there.
# But we can see that in the top 10 states in terms of firearm usage as murder weapon, there are 3 states with loose gun laws and only 1 with strict laws.
# In the last 10 states there 3 states with strict gun laws and only 1 with loose laws.
# 
# So i would suggest to put more strict gun laws in place for all the states ;)

# In[ ]:




