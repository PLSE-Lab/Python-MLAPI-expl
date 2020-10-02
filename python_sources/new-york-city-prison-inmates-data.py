#!/usr/bin/env python
# coding: utf-8

# # Starter EDA - NYC Daily Inmates
# 
# Make the necessary library imports

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)
import seaborn as sns

sns.set_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read data & get a gist of the kind of data

# In[ ]:


data = pd.read_csv("../input/daily-inmates-in-custody.csv")
data.sample(5)


# ---

# ### Preliminary Inferences
# 
# 1. Data is largely categorical nature with only two features - **Age** & **Top Charge** being numeric.
# 
# 2. **DISCHARGED_DT** has no entries & therefore can safely be ignored
# 
# 3. Points of interests can be - **Gender** & **Race** around which basic visualizations can be centered

# In[ ]:


data.info()


# ---

# ### Let us begin by plotting a distribution of Ages
# 
# This is quite expected as most of detainees have ages in the range of 20-40 years.

# In[ ]:


plt.figure(figsize=(20,7))
h = plt.hist(pd.to_numeric(data.AGE).dropna(), facecolor='g', alpha=0.75, bins=100)
plt.title("Distribution of Ages")
plt.xlabel("Age of Inmates")
plt.ylabel("Count")


# In[ ]:


def my_autopct(pct):
    return ('%.2f' % pct) if pct > 3 else ''
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']


# ## Proportion of Detainees by GENDER & RACE

# In[ ]:


f, ax = plt.subplots(1,2, figsize=(15,7))
#sns.countplot(x='RACE', hue='GENDER', data=data, ax=ax[1][1], palette="Set2")
pie = ax[0].pie(list(data['GENDER'].value_counts()), 
                   labels=list(data.GENDER.unique())[1:],
                  autopct='%1.1f%%', shadow=True, startangle=90)
pie = ax[1].pie(list(data['RACE'].value_counts()), 
                   labels=list(data.RACE.unique())[1:],
                  autopct=my_autopct, shadow=True, startangle=90, colors=colors)
ax[0].set_title("GENDER DISTRIBUTION AMONG INMATES")
ax[1].set_title("RACE")
#ax[1][1].set_title("RACE - GENDER DISTRIBUTION")


# ### Individual proportions of RACE across the two GENDERS

# In[ ]:


plt.figure(figsize=(20,7))
sns.countplot(x='RACE', hue='GENDER', data=data, palette="Set2",
             order = data['RACE'].value_counts().index)
plt.ylabel("Number of Inmates")


# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(x='GENDER', hue='RACE', data=data, palette="Set2",
             order = data['GENDER'].value_counts().index)
plt.ylabel("Number of Inmates")


# ## Under Mental Observation
# 
# A significant proportion of inmates are under Mental Observation (> 40%)

# In[ ]:


f, ax = plt.subplots(2,1, figsize=(7,15))
#sns.countplot(x='RACE', hue='GENDER', data=data, ax=ax[1][1], palette="Set2")
pie = ax[0].pie(list(data['BRADH'].value_counts()), 
                   labels=list(data.BRADH.unique()),
                  autopct='%1.1f%%', shadow=True, startangle=90)
sns.countplot(x='BRADH', hue='RACE', data=data, palette="Set2",
             order = data['BRADH'].value_counts().index, ax=ax[1])
ax[0].set_title("Distribution on the basis of GENDER of inmates under Mental Observation")
ax[1].set_xlabel("Inmates under Mental Observation? Y-Yes, N-No")
#ax[1].set_title("RACE")


# In[ ]:


plt.figure(figsize=(7,7))
explode = (0,0,0.1)
f, ax = plt.subplots(1,2, figsize=(15,7))
#sns.countplot(x='RACE', hue='GENDER', data=data, ax=ax[1][1], palette="Set2")
pie = ax[0].pie(list(data['CUSTODY_LEVEL'].value_counts()), 
                   labels=list(data.CUSTODY_LEVEL.unique())[1:],
                  autopct='%1.1f%%', shadow=True, startangle=90, explode=explode)
pie = ax[1].pie(list(data.SRG_FLG.value_counts()), 
                   labels=list(data.SRG_FLG.unique()),
                  autopct=my_autopct, shadow=True, startangle=90, colors=colors)
ax[0].set_title("% of detainees with MIN/MAX/MID level of Detention")
ax[1].set_title("Member of the Gang?")


# ## Gang Affiliations

# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(x='SRG_FLG', hue='RACE', data=data, palette="Set2",
             order = data['SRG_FLG'].value_counts().index)
plt.legend(loc="upper right")
plt.title("Affiliation of Gang by Race")
plt.xlabel("Gang Affiliations? Y-Yes, N-No")


# ## Future Work
# 
# There still remains scope for further EDA by using columns like - **Sealed**, ** Inmate Status Code** etc. I'll keep adding as I figure out some other vizualizations for the same. In the meantime, if you like this notebook, do **Upvote**. Thanks!
# 
# One **nuance** that I wasn't able to figure out was the meaning of Varible - **TOP_CHARGE**. Since it is numeric in nature, it must signify the magnitude of felony or something like that, please do comment if you're able to extract some meaningful inferences out of this one. 

# In[ ]:




