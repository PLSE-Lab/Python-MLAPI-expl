#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


d = pd.read_csv("../input/accidents_2017.csv")


# In[ ]:


d.head()


# In[ ]:


list(d)


# In[ ]:


d1 = d['Weekday'].value_counts()
d1 = pd.DataFrame(d1)
#d1.columns = ['Weekdays', 'Percentage']
y1 = d1
d1


# In[ ]:


d1 = d1 / sum(d1['Weekday'])


# In[ ]:


d1


# In[ ]:


d1.sort_values( by ='Weekday')


# In[ ]:


d1 = d1 * 100
#Percentages are prettier to look at:)


# In[ ]:


d1


# In[ ]:


s = np.std(d1['Weekday'])


# In[ ]:


s


# In[ ]:


m = np.mean(d1['Weekday'])


# In[ ]:


d1 = d1 - m


# In[ ]:


d1


# In[ ]:


d1 = d1 / s


# In[ ]:


d1


# # Inference
# 
# What we have done here is we checked how many standard deviations is our data above or below the mean. This clearly gives us some interesting insights:
# 1. Fridays record the most number of accidents. almost 0.95(STD) above the average accidents per weak.
# 2. Saturday, Sunday encounter the least. Sunday lower among the both. What might be the reason? No Jobs ? Not sure Sunday's also mean more amount of outings. But still, since weekends are seeing a sudden decline in number of accidents, it might mean due to no job goers.
# 
# Now let's look at the part of the day when accidents occur the most.

# In[ ]:


d.head()


# In[ ]:


d2 = d.iloc[:,[4,8]]
d2.head()


# In[ ]:


d3 = d2.loc[d2['Weekday'] == 'Friday']


# In[ ]:


d3.head()


# In[ ]:


pd.unique(d['Part of the day'])


# So now we would look at how accidents occur on Fridays. Also we look at what time do they occur mostly

# In[ ]:


d4 = d3['Part of the day'].value_counts()


# In[ ]:


d4 = pd.DataFrame(d4)


# In[ ]:


d4


# In[ ]:


d4 = d4 / sum(d4['Part of the day'])


# In[ ]:


d4 = d4 * 100


# In[ ]:


d4


# # Inference
# So if we look closely, on Friday's (The most accident prone day) more than half of the accidents occur during afternoons. About 40% occur during Morning and very less at nights (as normally expected)
# 
# 

# In[ ]:


d5 = d.iloc[:, 8]


# In[ ]:


d5.head()


# In[ ]:


d5 = d5.value_counts()


# In[ ]:


d5 = pd.DataFrame(d5)


# In[ ]:


d5
ei = d5
ei


# In[ ]:


d5 = d5 / sum(d5['Part of the day'])
d5 = d5 * 100


# In[ ]:


d5


# Proportion of accidents for entire week are similar to Friday data. Let's check for Monday

# In[ ]:


d6 = d2.loc[d2['Weekday'] == 'Tuesday']


# In[ ]:


d6 = d6['Part of the day'].value_counts()


# In[ ]:


d6 = pd.DataFrame(d6)
d6


# In[ ]:


d6 = d6 / sum(d6['Part of the day'])
d6 = d6 * 100
d6


# Even on monday the distribution of accidents is similar [50 ,40, 10]

# In[ ]:


d.head()


# Until now we have seen how Accidents varied according to day of the week and part of the day. We concluded 
# 1. Friday Afternoons are the most accident prone times. 
# 2. We also saw the accident distribution over the part of days was almost [50, 40, 10] --> [Afternoon, Morning, Night].
# 3. Weekdays are safer, Sundays being the safest.

# Now let's see how accidents vary over the month. For that we divide a month in 4 quarters and analyse them separately.

# In[ ]:


d7 = d.iloc[:, 6]
d7 = pd.DataFrame(d7)
d7.head()


# In[ ]:


w1 = d7.loc[d7['Day']<=7]
w2 = d7.loc[(d7['Day']>7) & (d7['Day']<=14)]
w3 = d7.loc[(d7['Day']>14) & (d7['Day']<=21)]
w4 = d7.loc[(d7['Day']>21) & (d7['Day']<=31)]
w1 = w1.describe()
w4.describe()


# In[ ]:


w1 = 2342
w2 = 2346
w3 = 2508
w4 = 3143
w = w1 + w2 + w3 + w4


# In[ ]:


w1 = w1 / w
w1 = w1 * 100
w2 = w2 / w
w2 = w2 * 100
w3 = w3 / w
w3 = w3 * 100
w4 = w4 / w
w4 = w4 * 100


# In[ ]:


w1


# In[ ]:


w2


# In[ ]:


w3


# In[ ]:


w4


# Looking at the results there is not much deviation of accidents in four weeks. However week 4 experiences a sudden increase in number of accidents.
# So probabilistically, week 4 Friday Afternoons are the worst for accidents.

# In[ ]:


d.head()


# In[ ]:


e1 = d.loc[d['Serious injuries']!=0]


# In[ ]:


e1.head()


# In[ ]:


e1 = e1.iloc[:, [1,3,4,6,7,8,10]]


# In[ ]:


e1.head()


# In[ ]:


e2 = e1['Part of the day'].value_counts()


# In[ ]:


type(e2)
eii = e2


# In[ ]:


e2 = pd.DataFrame(e2)
e2 = e2 / sum(e2['Part of the day'])
e2 = e2 * 100


# In[ ]:


e2


# ## Distribution of Serious Injuries over Afternoons, Mornings and Nights

# In[ ]:


e2


# Now let us see how serious injuries per accidents varry over the parts of days

# In[ ]:


d5


# In[ ]:


ei


# In[ ]:


eii = pd.DataFrame(eii)
eii


# In[ ]:


e1W = eii['Part of the day'] / ei['Part of the day']


# In[ ]:


e1W * 100


# e1W shows what proportion of accidents occuring in a part of the day cause serious injuries.
# 1. 2.2% of accidents occuring during afternoons are serious and so on.
# 2. More proportion of accidents occuring at night cause serious injuries.
# 3. Mornings have less proportion of serious injusries. Maybe due to more attentive driving during mornings.

# Now let's see what proportion of total accidents account for serious injuries.

# In[ ]:


ei


# In[ ]:


ei = sum(ei['Part of the day'])
ei


# In[ ]:


eii


# In[ ]:


eii = eii['Part of the day'] / ei


# In[ ]:


eii = eii * 100


# In[ ]:


eii = pd.DataFrame(eii)
eii


# <img src = "source.jpg">

# I always write down numbers when dealing with Simpson's Paradoxes. Yeah! Here is one. Just looking at the proportion of serious injuries we might conclude nights are safer as long as serious injuries are concerned. However are they really?
# 
# Looking at the secound column, we can see more proportion of total accidents occuring at night are serious. This is a paradox. Blindly looking at a single data reveals wrong results. 
# 
# Nights actually witness larger proportion of serious accidents off total accidents happening at night. 

# In[ ]:


er = e1['Weekday'].value_counts()
er2 = er
er2 = pd.DataFrame(er2)


# In[ ]:


er = pd.DataFrame(er)
er


# In[ ]:


er = er['Weekday'] / sum(y1['Weekday'])
er = er * 100
er = pd.DataFrame(er)
er


# In[ ]:


er1 = er2['Weekday'] / y1['Weekday']
er1 = er1 * 100
er1 = pd.DataFrame(er1)
er1


# In[ ]:


er1 = er1.sort_values(by = ['Weekday'], ascending = False)


# In[ ]:


er1 = pd.DataFrame(er1)


# In[ ]:


#er1
er1 = er1.rename(columns = {'Weekday' : 'P(S_Accident | Weekday)'})
er1


# In[ ]:


er = er.rename(columns = {'Weekday' : 'P(S_Accident)'})
er


# In[ ]:


pd.concat([er, er1], axis = 1)


# Looking at the final DataFrame we see how conditional probability is changing the entire perception again. Probability of having serious accidents is highest during Fridays, it seems! However the second column which is the conditional probability of a serious accident given that it's a particular weekday shows different story alltogether. 
# 
# # Inference
# Sundays have the highest chance of experiencing serious accidents. Also as we previously concluded, nights have the highest risk of serious accidents we can say Sunday nights have the highest risk of experiencing a serious accident.
