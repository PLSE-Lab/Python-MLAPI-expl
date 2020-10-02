#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# This is a work in progress. I hope you guys find it helpful while I'm working on it to make it better :)
# 
# The kernel is pretty large to read all at once so let's read it in chunks of 100MB as suggested by [Robert's kernel](https://www.kaggle.com/datark1/malware-prediction-eda) and see a sample of the training set.

# In[ ]:


chunksize = 100000
train = None

for chunk in pd.read_csv("../input/train.csv", chunksize=chunksize, iterator=True):
    if train is None:
        train=chunk.copy()
    else:
        train.append(chunk)


# In[ ]:


train.sample(5)


# There are so many columns that is hard to see all names at once. Since we also have to check if there are null values, let's check for them and list the names at the same time.

# In[ ]:


print (train.columns)


# In[ ]:


train.isnull().sum().sort_values(ascending = False)[:43]


# Our train dataset was loaded using 100k rows. PuaMode variable has an incredibly huge amount of missing data and hence doesn't seem a good one to use. When using models other than the beloved LGB, dealing with such huge number of missing data will be troublesome, I'd remove all the variables with over 50k missing values (more than half of its values missing). It is also important to notice that most of the variables above are `Census_Something`.
# 
# Now we will start our exploration. Since we want to classify whether the system has a detection or not, it makes sense to check which numerical features are mostly correlated to whether or not there is a detection. 

# In[ ]:


corr_matrix = train.corr()
corr_matrix['HasDetections'].sort_values(ascending = False)


# In[ ]:


plt.figure(figsize=(20,15))
sb.heatmap(corr_matrix)
plt.title("Correlation Matrix", size = 25)


# The most positively or negatively correlated variables (above 5%) to our target are (with their respective correlations):
# 
# - AVProductsInstalled -0.146059
# - AVProductStatesIdentifier 0.116914
# - Census_IsAlwaysOnAlwaysConnectedCapable -0.062527
# - IsProtected  0.055415
# - Census_TotalPhysicalRAM 0.054507
# - Census_PrimaryDiskTotalCapacity  0.052893
# - Census_ProcessorCoreCount 0.052681
# - Census_IsVirtualDevice -0.051790
# - Wdft_IsGamer 0.050511
# 
# But what are those variables? Most seem obvious but let's get the definition given to us in the challenge's page:
# - AVProductsInstalled: No description
# - AVProductStatesIdentifier: ID for the specific configuration of a user's antivirus software
# - IsProtected:  a. TRUE if there is at least one active and up-to-date antivirus product running on this machine. b. FALSE if there is no active AV product on this machine, or if the AV is active, but is not receiving the latest updates. c. null if there are no Anti Virus Products in the report. Returns: Whether a machine is protected.
# - Census_IsAlwaysOnAlwaysConnectedCapable: Retreives information about whether the battery enables the device to be AlwaysOnAlwaysConnected
# - Census_TotalPhysicalRAM: Retrieves the physical RAM in MB
# - Census_PrimaryDiskTotalCapacity: Amount of disk space on primary disk of the machine in MB
# - Census_ProcessorCoreCount: Number of logical cores in the processor
# - Census_IsVirtualDevice: Identifies a Virtual Machine (machine learning model)
# -  Wdft_IsGamer: Indicates whether the device is a gamer device or not based on its hardware combination.
# 
# The three most correlated variables are related to the antivirus setup, which seems logical. The next ones are hardware configurations and whether or not the system is a VM. Now we'll take a look at the distribution of each of those variables.
# 
# ## Univariate Analysis
# ____
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')

sb.countplot(train.AVProductsInstalled.dropna());


# Most users have only one Antivirus installed on their computers, but a considerable number also have two. A small amount of users seems to be very worried about getting a virus and have three different softwares protecting their systems! There are also some even more worried users that run 4 or even 5 different antivirus softwares. Wow. 

# In[ ]:


sb.distplot(train["AVProductStatesIdentifier"].dropna(), kde = False);


# Although these are IDs, there are not much of them that are unique - only 2462. The histogram is showing us that there are some kinds (or just one kind. It is hard to say with this level of granularity on the histogram) of antivirus configurations that are most used by users. 
# 
# What about `IsProtected` field?

# In[ ]:


sb.countplot(train.IsProtected);


# Great! Most systems are protected. Now the hardware features. 

# In[ ]:


sb.countplot(train.Census_IsAlwaysOnAlwaysConnectedCapable.dropna());


# Most users don't have batteries configured to be always connected. What about the RAM?

# In[ ]:


sb.distplot(train.Census_TotalPhysicalRAM.dropna(), kde = False, bins = 1000)
plt.xlim(0,30000)


# *NOTE: I've limited the plot to 30GB of Total RAM, but there are a few outliers above this value.*
# 
# We can see that most users have 4GB of RAM, followed by those who have 8GB. Other values are less common but we can see expressive peaks at 2GB and 16GB. 
# 
# Let's see the disk space distribution.

# In[ ]:


sb.distplot(train.Census_PrimaryDiskTotalCapacity.dropna(), kde = False, bins = 1000)
plt.xlim(0, 1000000)


# Most computers have 500GB or 1TB of disk available, the most common sizes in market, so this is expected. Just like in the memory plot, there are some outliers above those values but I've limited the plot to 1TB.
# 
# Now the number of processor cores.

# In[ ]:


sb.distplot(train.Census_ProcessorCoreCount.dropna(), kde = False)
plt.xlim(0,10);


# There are lots of systems using 4 and 2 cores. To a less extent there are some computers with 8 cores. 

# In[ ]:


sb.countplot(train.Census_IsVirtualDevice.dropna());


# Only a minimum amount of users are using virtual machines.  
# 
# Lastly, how many gamer configurations we have in the training set?

# In[ ]:


sb.countplot(train.Wdft_IsGamer.dropna())


# The gamer configurations are almost 1/3 the total amount of configurations! This is a quite impressive number. Although some may not be gamers, but just bought their PCs with a good configuration.
# 
# ## Bivariate Analysis
# ___
# Now that we've seen how each of those variables are distributed across the training sample, let's check how they related to each other. If two of them are highly correlated, then it may be better to left one out of the final model not to account the same information twice. Although these correlations can be seen on the first heatmap, we are better off looking at a new heatmap containing only those columns. 

# In[ ]:


cols_to_use = ['AVProductsInstalled', 'AVProductStatesIdentifier', 
               'Census_IsAlwaysOnAlwaysConnectedCapable','IsProtected',
               'Census_TotalPhysicalRAM', 'Census_PrimaryDiskTotalCapacity',
               'Census_ProcessorCoreCount', 'Census_IsVirtualDevice', 'Wdft_IsGamer']
plt.figure(figsize=(15,10))
sb.heatmap(train[cols_to_use].corr(), annot = True);


# So, AVProductsInstalled are considerably correlated to AVProductStatesIdentifier and so are PhysicalRAM and ProcessorCoreCount. The latter ones are probably so because, as we know, the standard products are sold in similar specifications. For instance, a non-gamer  i3 laptop usually has 4GB RAM. 
