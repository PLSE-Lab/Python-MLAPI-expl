#!/usr/bin/env python
# coding: utf-8

# For all tennis fans, out there, just a small small analysis done here.
# Just a beginner's try at something interesting to experiment with.
# 
# All the ATP tours begin with 3 main tournaments: Brisbane International, Chennai Open and the Doha Open (Qatar Exxon Mobil Open). Players are allowed to choose between these 3 tournaments, which happen simultaneously. 
# The biggest tournaments in the tour are the Grand Slams: Australian Open, French Open, Wimbledon and the US Open. 
# 
# This small piece of code is to find out which opening tournaments help prepare players best for the Grand Slams. Or in other words, which winners of the opening tournaments go on to win the Grand Slams. 

# Let's import the necessary libraries and set up the workspace.

# In[3]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import os
print(os.listdir("../input"))


# Let's read the data into a dataframe called data. 

# In[5]:


data=pd.read_csv('../input/ATP Dataset_2012-01 to 2017-07_Int_V4.csv')


# First, we find out the winners of the opening tournaments. 
# 
# We create a list of the opening tournaments, and create a subset of the dataframe with just the opening tournaments.
# 
# Next, we create another subset of the previous subset, with the finals only, to help us find the tournament winners. 

# In[6]:


opentours=("BrisbaneInternational","ChennaiOpen","QatarExxonMobilOpen")
dataopen=data[data['Tournament'].isin(opentours)]
dataopenwin=dataopen[dataopen['Round']=="TheFinal"]
dataopenwin.Winner


# We now have the winners of all the opening tournaments.
# 
# Next, we get the winners of the Grand Slam tournaments in a similar manner. 

# In[8]:


slamtours=("AustralianOpen","FrenchOpen","Wimbledon","USOpen")
dataslam=data[data['Tournament'].isin(slamtours)]
dataslamwin=dataslam[dataslam['Round']=="TheFinal"]
dataslamwin.Winner


# Next, we compare both the dataframe subsets we created. We use the "Winner" column for this.
# 
# We check if the winners of the opening tournaments are in the list of winners of the Grand Slams. We then enter the data that satisfies that condition in to another dataframe, dataopenslam.

# In[10]:


dataopenslam=dataopenwin[dataopenwin.Winner.isin(dataslamwin.Winner)]


# Now that we have that, all that's left is to utilise the data in the most useful way: plots. 
# 
# We create simple bar plots using the seaborn library, as follows. 

# In[13]:


sns.countplot(dataopenslam.Winner)


# In[12]:


sns.countplot(dataopenslam.Tournament)


# We can conclude that:
# 1. All the opening tournaments are equally successful in this measure. 
# 2. Stan Wawrinka has won 3 Grand Slams (all his 3 Grand Slams) whenever he has won an opening tournament (3 Chennai Open wins) 
# 
# I was unable to make it according to the years they won each tournament, because the date column in this dataset wasn't properly calibrated.
# 
# Please feel free to make changes to the script as necessary by forking the script.
