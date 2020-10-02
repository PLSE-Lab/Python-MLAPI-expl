#!/usr/bin/env python
# coding: utf-8

# ##Where are the tent only campsites?
# The National Park Service, Forest Service, Bureau of Land Management and other agencies maintain campsites in public lands. These campgrounds can accommodate RV-ers, car campers, and backpackers. For an upcoming group camping trip, I was tasked with finding a campground that in between Salt Lake City, UT and Grand Teton NP, WY and struggled to find tent-only campsites enroute to Grand Teton NP. Never having struggled to find a tent-only campsite in California, I wondered if California has more tent-only sites (compared to the usual 'standard nonelectric') compared to other places I've traveled to in the American West. Or have I just gotten lucky? 
# 
# This is the notebook version of the python kernel. 

# In[ ]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/fed_campsites.csv', index_col=0)
df.head()


# In[ ]:


states = df['AddressStateCode'].unique()
print (states)


# In[ ]:


df2 = pd.DataFrame([])

#calculate the percent of tent-only, standard electric, and standard nonelectric campsites per state
for state in states: 
    pc_tent = df[(df.AddressStateCode == state) & (df.CampsiteType == 'TENT ONLY NONELECTRIC')].count()/df[df.AddressStateCode == state].count()
    pc_sne = df[(df.AddressStateCode == state) & (df.CampsiteType == 'STANDARD NONELECTRIC')].count()/df[df.AddressStateCode == state].count()
    pc_se = df[(df.AddressStateCode == state) & (df.CampsiteType == 'STANDARD ELECTRIC')].count()/df[df.AddressStateCode == state].count()
    temp = pd.DataFrame({'state': state,
                         'frac_tent_only': pc_tent,
                         'frac_standard_nonelec': pc_sne,
                         'frac_standard_elec': pc_se}) 
    df2 = df2.append(temp, ignore_index = True)
    
df2 = df2.drop_duplicates(['state'])
df2 = df2.sort(columns = 'state')
df2 = df2.reset_index()
df2 = df2.drop('index', axis=1)
df2.head()


# ##Plotting tent-only campsite data
# 
# Now that I've calculated the fraction of tent-only sites, standard electric sites, and standard nonelectric sites per state, we can graph them on a state-by-state basis. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="white", context="talk")

ax = sns.barplot('state', 'frac_tent_only', data = df2)
ax.set_xticklabels(df2['state'],rotation=45)
plt.show();


# ##Who stands out?
# 
# Based on this graph California doesn't seem to stand out with ~13% of it's campsites designated as tent-only. The interesting outliers are the states with no tent-only campsites (Alabama, Connecticut, Louisiana, Massachusetts, Missouri, and Vermont) and the states with a high proportion of tent-only sites (Maine, New Jersey, and New York). 

# In[ ]:


agg = df.groupby(['AddressStateCode']).count()
print (agg.loc['NJ'],  agg.loc['NY'], agg.loc['CA'])


# In[ ]:


df[df.AddressStateCode == 'NJ']


# In[ ]:


df[df.AddressStateCode == 'NY']


# ##The effect of low N
# 
# When we look at the details of the campgrounds located in NY and NJ, we see that all 20 campsites in the state of NJ are located in one campground and the 68 campsites in the state of NY are split between two campgrounds. Based on the latitude and longitude values, there both in the greater New York City area. Would you want to drive an RV there? I'm not sure I would. 

# ##Testing the relationship between tent-only campsites and RV campsites
# 
# Standard campsites don't exclude car campers per se, but I definitely prefer camping in tent-only sites. I find that I get a bit more solitude, plus it's nice pitching a tent on dirt as opposed to a gravel or asphalt driveway.  But what's the overall trend between the prevalence of tent-only campsites versus standard nonelectric campsites? They're almost certainly negatively correlated, but some kind of linear regression between those two variables would determine which states fall above the trend line. 

# In[ ]:


g = sns.lmplot(x = 'frac_standard_nonelec', y = 'frac_tent_only', data = df2,                legend=False, hue = 'state', fit_reg=False, size = 5, aspect = 2)
sns.regplot(x = 'frac_standard_nonelec', y = 'frac_tent_only', data = df2,             scatter=False, ax=g.axes[0, 0])


box = g.ax.get_position() # get position of figure
g.ax.set_position([box.x0, box.y0, box.width*0.6, box.height]) # resize position

# Put a legend to the right side
g.ax.legend(loc='right', bbox_to_anchor=(1.65, .5), ncol=3)
plt.show();


# ##The final result? California does have a relative enrichment for tent-only sites
# 
# The states that fall above the trend line are AZ, AK, CA, CO, NH, NJ, NY, MD, ME, and VA. 
