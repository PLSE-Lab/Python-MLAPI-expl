#!/usr/bin/env python
# coding: utf-8

# **After the Iowa Caucus and the New Hampshire debate I was curious if the endorsments had any effect that may have been missed on the results.**

# In[ ]:


import numpy as np 
import pandas as pd 

data = pd.read_csv("/kaggle/input/2020-democratic-primary-endorsements/endorsements-2020.csv")
data['date'] = pd.to_datetime(data['date'])


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
names, values = data['endorsee'].value_counts().index, data['endorsee'].value_counts().values
plt.bar(names,values)
plt.xticks(rotation = 90)
plt.title("Total Endorsments")
plt.xlabel("Candidate name")
plt.ylabel("Number of Endorsments")


# **As expected Joe Biden was a very popular candidate. However these endorsments go back to 2017, out of curiosity have candidates that have done better recently (Bernie and Buttigieg are those that standout) gotten more recent endorsments? **

# In[ ]:


data2019 = data[data['date'] > pd.datetime(2019,1,1)]

names, values = data2019['endorsee'].value_counts().index, data2019['endorsee'].value_counts().values
plt.bar(names,values)
plt.xticks(rotation = 90)
plt.title("Total Endorsments since 2019")
plt.xlabel("Candidate name")
plt.ylabel("Number of Endorsments")


# **Surprisingly enough Joe Biden endorsments still ranks well above the other prominent candidates.**

# In[ ]:


data2020 = data[data['date'] > pd.datetime(2020,1,1)]

names, values = data2020['endorsee'].value_counts().index, data2020['endorsee'].value_counts().values
plt.bar(names,values)
plt.xticks(rotation = 90)
plt.title("Total Endorsments in 2020")
plt.xlabel("Candidate name")
plt.ylabel("Number of Endorsments")


# **Then next thing I wanted to check was if Iowa was a stand out state for either Bernie or Pete.**

# In[ ]:


iaData = data[data['state'] == 'IA']
names, values = iaData['endorsee'].value_counts().index, iaData['endorsee'].value_counts().values
plt.bar(names,values)
plt.xticks(rotation = 90)
plt.title("Total Endorsments from Iowa")
plt.xlabel("Candidate name")
plt.ylabel("Number of Endorsments")


# In[ ]:


uniqueStatesNum = len(np.unique(data['state']))
bernieAveragePerState = len(data[data['endorsee'] == 'Bernie Sanders']) / uniqueStatesNum
peteAveragePerState = len(data[data['endorsee'] == 'Pete Buttigieg']) / uniqueStatesNum
print("Bernie average endorsment per state: ",bernieAveragePerState)
print("Pete   average endorsment per state: ",peteAveragePerState)
print("Bernie endorsments in Iowa         : ",len(iaData[iaData['endorsee'] == 'Bernie Sanders']))
print("Pete   endorsments in Iowa         : ",len(iaData[iaData['endorsee'] == 'Pete Buttigieg']))


# **Pete Buttigieg had 1 and Bernie had 0 endorsments in the state of Iowa, so their wins still seem as surprising as ever. **

# **My next thought was maybe they receieved endorsments from the neighboring states of Iowa**

# In[ ]:


neighborStatesIA = data[data['state'].isin(['IL','WI','MN','SD','NE','MO'])]
names, values = neighborStatesIA['endorsee'].value_counts().index, neighborStatesIA['endorsee'].value_counts().values
plt.bar(names,values)
plt.xticks(rotation = 90)
plt.title("Total Endorsments From IA Neighbors")
plt.xlabel("Candidate name")
plt.ylabel("Number of Endorsments")


# **This is the first evidence of Joe Biden not being on top, Amy Klobuchar is ranked higher because she is a senator from the state of Minnesota, one of the neighboring states. However, Bernie's home state of Vermont is still far from Iowa and Pete Buttigieg is nowhere to be found. **

# **I think the key question is how can this information give us any indicator of what to expect in New Hampshire on Tuesday Febuary 11 2020**

# In[ ]:


nhData = data[data['state'] == 'NH']
names, values = nhData['endorsee'].value_counts().index, nhData['endorsee'].value_counts().values
plt.bar(names,values)
plt.xticks(rotation = 90)
plt.title("Total Endorsments from New Hampshire")
plt.xlabel("Candidate name")
plt.ylabel("Number of Endorsments")


# In[ ]:


neighborStatesNH = data[data['state'].isin(['MA','VT','MA'])]
names, values = neighborStatesNH['endorsee'].value_counts().index, neighborStatesNH['endorsee'].value_counts().values
plt.bar(names,values)
plt.xticks(rotation = 90)
plt.title("Total Endorsments From NH Neighbors")
plt.xlabel("Candidate name")
plt.ylabel("Number of Endorsments")


# **With Bernie and Eliabeth Warren being from neighboring states, you may expect them to be prominent leaders in the upcoming caucus. However Buttigieg only needed one endorsment from Iowa and performed well, so having an endorsment from New Hampshire and Masschusetts may help him benefit.**

# **Learnings:**
# 1. Endorsments may not always correlate in the state someone is voting in
# 2. These Endorsments may not be the most telling indicator of a canidates performance
# 3. Pete Buttigieg's last name is hard to spell 

# **Appendix:**
# 
# 
# **I did some other preliminary searchs through the data but couldn't find anything that put Buttigieg and Sanders above everyone else **
# 

# In[ ]:


# Average points for each canidate by endorser 
data.groupby('endorsee')['points'].mean().sort_values(ascending = False)


# In[ ]:


for cat in np.unique(data['category']):
    catSpecific = data[data['category'] == cat]
    print(cat + ":",'\n')
    print(catSpecific.groupby('endorsee')['date'].count().sort_values(ascending = False).to_string(),"\n")


# In[ ]:




