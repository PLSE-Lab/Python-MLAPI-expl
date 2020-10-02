#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import os
print(os.listdir("../input"))


# **1. Donors**

# In[2]:


don = pd.read_csv("../input/Donors.csv", low_memory=False)
teach_not = don["Donor Is Teacher"].value_counts()


# In[3]:


fig_teach_not = plt.figure(figsize=(5,5))
plt.bar([1,2], [teach_not["No"], teach_not["Yes"]])
plt.xticks([1, 2], ["not teachers", "teachers"], fontsize=15)
plt.title("a teacher or not")
plt.show()


# **2. Donors in cities**

# In[4]:


cities = don["Donor City"].value_counts()


# In[5]:


perc = cities.sum() / 100 
percent_each = []
for i in cities.index:
    part = cities[i] / perc
    percent_each.append((i, part))
    
cont = []
load  = 0 
for i in percent_each:
    load += i[1]
    cont.append(load)


# - cumulative load of cities into the total donorship seems "logarithmic"-like
# - so, first 2000 cities out of  more than 14000 have about 80% of donors

# In[7]:


fig_cum = plt.figure(figsize=(10,6))
plt.plot(cont, lw=5, color="darkred")
plt.fill_between(list(range(len(cont))), cont, color="salmon")
plt.xlabel("cities sorted in descending order by donors", fontsize=15)
plt.ylabel("donor %", fontsize=15)
plt.title("city cumulative load into donor quantity", fontsize=20)
plt.show()


# In[36]:


fig_cities = plt.figure(figsize=(15, 8))
plt.bar(list(range(50)), cities.iloc[:50], color="salmon")
plt.xticks(list(range(50)), cities.index[:50], rotation="vertical", fontsize=15)
plt.xlabel("city name", fontsize=15)
plt.ylabel("number of donors", fontsize=15)
plt.title("donors per city", fontsize=20)
plt.show()


# In[9]:


dist_cities = plt.figure(figsize=(10,8))
sns.distplot(cities, hist=False)
plt.title("distribution of donors per city", fontsize=20)
plt.show()


# **3. Donors in states**

# In[15]:


states = don["Donor State"].value_counts()


# In[16]:


# lets show which cities explain most donorship
perc = states.sum() / 100 
percent_each_state = []
for i in states.index:
    part = states[i] / perc
    percent_each_state.append((i, part))

cont_state = []
load  = 0 
for i in percent_each_state:
    load += i[1]
    cont_state.append(load)


# In[17]:


# cumulative load of states into the donorship
fig_cum = plt.figure(figsize=(10,8))
plt.plot(cont_state, lw=2, color="darkgreen")
plt.fill_between(list(range(len(cont_state))), cont_state, color="lightgreen")

plt.xlabel("states sorted in descending order by donors", fontsize=15)
plt.ylabel("donor %", fontsize=15)
plt.title("states cumulative load into donor quantity", fontsize=20)
plt.show()


# In[18]:


fig_cities = plt.figure(figsize=(15, 8))
plt.bar(list(range(len(states))), states.iloc[:], color="green")
plt.xticks(list(range(len(states))), states.index[:], rotation="vertical", fontsize=15)
plt.xlabel("state", fontsize=15)
plt.ylabel("number of donors", fontsize=15)
plt.title("donors per state", fontsize=25)
plt.show()


# In[35]:


dist_states = plt.figure(figsize=(10,8))
sns.distplot(states)
plt.title("distribution of donors per state", fontsize=20)
plt.show()


# **4. Schools**

# In[20]:


sch = pd.read_csv("../input/Schools.csv")


# In[21]:


sch_type = sch["School Metro Type"].value_counts()


# In[22]:


fig = plt.figure(figsize=(8,6))
plt.bar(range(len(sch_type)), sch_type, color="darkblue", alpha=0.5)
plt.xticks(range(len(sch_type)), sch_type.index)
plt.title("school types", fontsize=20)
plt.show()


# In[23]:


state_sch = sch["School State"].value_counts()


# In[24]:


perc = state_sch.sum() / 100 
percent_each = []
for i in state_sch.index:
    part = state_sch[i] / perc
    percent_each.append((i, part))
    
cont = []
load  = 0 
for i in percent_each:
    load += i[1]
    cont.append(load)


# In[25]:


fig_cum = plt.figure(figsize=(10,6))
plt.plot(cont, lw=5, color="blue")
plt.fill_between(list(range(len(cont))), cont, color="lightblue")
plt.xlabel("states sorted in descending order by number of donated schools", fontsize=15)
plt.ylabel("%", fontsize=15)
plt.title("load of states into the donated schools", fontsize=20)
plt.show()


# In[26]:


fig_cities = plt.figure(figsize=(15, 8))
plt.bar(list(range(len(state_sch))), state_sch.iloc[:], color="lightblue")
plt.xticks(list(range(len(state_sch))), state_sch.index[:], 
           rotation="vertical", fontsize=15)
plt.xlabel("state", fontsize=15)
plt.ylabel("number of schools", fontsize=15)
plt.title("number of donated schools by state", fontsize=25)
plt.show()


# In[27]:


city_sch = sch["School City"].value_counts()


# In[28]:


perc = city_sch.sum() / 100 
percent_each = []
for i in city_sch.index:
    part = city_sch[i] / perc
    percent_each.append((i, part))
    
cont = []
load  = 0 
for i in percent_each:
    load += i[1]
    cont.append(load)


# In[29]:


fig_cum = plt.figure(figsize=(10,6))
plt.plot(cont, lw=5, color="darkred")
plt.fill_between(list(range(len(cont))), cont, color="salmon")
plt.xlabel("cities sorted in descending order by number of schools", fontsize=15)
plt.ylabel("%", fontsize=15)
plt.title("load of cities into the donated schools", fontsize=20)
plt.show()


# In[30]:


fig_cities = plt.figure(figsize=(15, 8))
plt.bar(list(range(50)), city_sch.iloc[:50], color="salmon")
plt.xticks(list(range(50)), city_sch.index[:50], 
           rotation="vertical", fontsize=12)
plt.xlabel("city", fontsize=15)
plt.ylabel("number of schools", fontsize=15)
plt.title("number of donated schools by city", fontsize=25)
plt.show()


# **5. Donations**

# In[31]:


donation = pd.read_csv("../input/Donations.csv")


# In[32]:


fig_money = plt.figure(figsize=(10,8))
sns.distplot(donation["Donation Amount"], hist=False)
plt.title("distribution of donation amount ($)", fontsize=20)
plt.show()


# donation summary:

# In[33]:


print("min donation: ", donation["Donation Amount"].min())
print("max donation : ", donation["Donation Amount"].max())
print("mean donation: ", donation["Donation Amount"].mean())
print("median donation: ", donation["Donation Amount"].median())
print("mode donation: ", donation["Donation Amount"].mode())


# In[ ]:




