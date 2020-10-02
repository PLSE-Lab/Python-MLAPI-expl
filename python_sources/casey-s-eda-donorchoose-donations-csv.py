#!/usr/bin/env python
# coding: utf-8

# # Casey's EDA: DonorChoose "Donations.csv"

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.


# **Viewing the data-types for donations.csv.  (This information is also included in the data description included.)**

# In[2]:


donations = pd.read_csv('../input/Donations.csv')
donations.head()


# **5 Number summary for 'Donation Amount'.**

# In[3]:


donations['Donation Amount'].describe()


# ### Question: How often do donations include the optional donation?

# In[4]:


levels=Counter(donations['Donation Included Optional Donation']).keys()
values=Counter(donations['Donation Included Optional Donation']).values()
for a, b in zip(levels, values):
    print("{0}: {1}".format(a,b))


# In[5]:


# Data to plot
labels = levels
sizes = values
colors = ['green', 'red']
 
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.2f%%', shadow=True, startangle=180)
plt.axis('equal')
plt.title('Donation Included Optional Donation')
plt.show()


# *So, majority of donations include something called the optional donation.  What will be interesting is whether or not the optional donation is included more often for repeat donors or first-time donors.*

# ### Question: What is the distribution of donation amounts for the optional included donation?

# In[6]:


data = [donations['Donation Amount'][donations['Donation Included Optional Donation']=="No"],donations['Donation Amount'][donations['Donation Included Optional Donation']=="Yes"]]
plt.boxplot(data,labels=('No','Yes'))
plt.title('Distribution of Donation Amounts')
plt.show()


# In[7]:


SequenceSkew=donations['Donor Cart Sequence'].skew()
SequenceKurt=donations['Donor Cart Sequence'].kurt()
AmountSkew=donations['Donation Amount'].skew()
AmountKurt=donations['Donation Amount'].kurt()

labels = ('Sequence Skew','Sequence Kurtosis','Donation Amt Skew','Donation Amt Kurtosis')
values = (SequenceSkew,SequenceKurt,AmountSkew,AmountKurt)

dict(zip(labels,values))
    


# *For both of these categories, and overall, there is a large right skew to the data.  Not normally distributed, which will be important later.*

# ### Question: What percentage of donors are first-time donors?

# In[8]:


num_donation=list(Counter(donations['Donor ID']).values())
first_timer = num_donation.count(1)
total = len(np.unique(donations['Donor ID']))
print(str(100*round(first_timer*1.0/total,ndigits=4))+"% of reported donors are first time donors.")


# In[9]:


byoptional = donations.groupby('Donation Included Optional Donation')


# **5 Number summaries for both donation amounts and donor cart sequence number.**

# In[10]:


byoptional['Donation Amount'].describe()


# In[11]:


byoptional['Donor Cart Sequence'].describe()


# In[12]:


donation_time=Counter(donations['Donor ID'])
donation_time_df=pd.DataFrame(list(donation_time.items()), columns=['Donor ID', 'Donation Count']) 
donation_time_df.head()


# ### Question: What is the distribution of donation amounts between first-time, second-time, third-time, and more than three-time donators?
# ### Question: Do first-time donors go straight to donating? (What is the donor cart sequence for first-time donors?)

# *First, merge the donations.csv dataframe with the donation time dataframe*

# In[13]:


donation_count_df=donations.merge(donation_time_df, how='inner', on='Donor ID')
donation_count_df.head()


# In[14]:


second_timer = num_donation.count(2)
third_timer = num_donation.count(3)
mult_timer = 0
for d in num_donation:
    if d>3:
        mult_timer=mult_timer+1


# In[15]:


labels = list(["First","Second","Third","Multiple"])
sizes = list([first_timer, second_timer, third_timer, mult_timer])
colors = ['green', 'red','blue','orange']
 
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.2f%%', shadow=True, startangle=180)
plt.axis('equal')
plt.title('Number of Donations For Donor')
plt.show()


# It looks like a majority of donors do it just once, but there is what seems to be a decreasing proportion of donors that donate twice, three times, and so on.

# In[16]:


donation_count_df['Timer']=""
donation_count_df.head()


# In[34]:


def f(row):
    if row['Donation Count'] == 1:
        val = "First"
    elif row['Donation Count']  ==2:
        val = "Second"
    elif row['Donation Count'] ==3:
        val = "Third"
    else:
        val = "Multiple"
    return val

###https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition#21711869


# In[35]:


donation_count_df['Timer'] = donation_count_df.apply(f,axis=1)


# In[38]:


bytimer = donation_count_df.groupby('Timer')
bytimer['Donation Amount'].describe()


# In[51]:


plt.hist(donation_count_df['Donation Amount'][donation_count_df['Timer']=="First"], range=[0,100],bins=20, histtype='stepfilled', normed=True, color='b', label='First Time')
plt.hist(donation_count_df['Donation Amount'][donation_count_df['Timer']=="Second"], range=[0,100],bins=20, histtype='stepfilled', normed=True, alpha=0.8,color='g', label='Second Time')
plt.hist(donation_count_df['Donation Amount'][donation_count_df['Timer']=="Third"], range=[0,100],bins=20, histtype='stepfilled', normed=True,alpha=0.8, color='r', label='Third Time')
plt.hist(donation_count_df['Donation Amount'][donation_count_df['Timer']=="Multiple"], range=[0,100],bins=20, histtype='stepfilled', normed=True, alpha=0.5,color='yellow', label='Multiple')
plt.title("Donation Amounts & Donation Time")
plt.xlabel("Donation Amount")
plt.ylabel("Probability")
plt.legend()
plt.show()


# It looks like a donor will generally donate more the more times they donate.

# In[55]:


plt.hist(donation_count_df['Donation Amount'][donation_count_df['Timer']=="First"], range=[0,100],bins=20, histtype='stepfilled', normed=True, color='b', label='First Time')
plt.hist(donation_count_df['Donation Amount'][donation_count_df['Timer']=="Second"], range=[0,100],bins=20, histtype='stepfilled', normed=True, alpha=0.8,color='g', label='Second Time')
plt.hist(donation_count_df['Donation Amount'][donation_count_df['Timer']=="Third"], range=[0,100],bins=20, histtype='stepfilled', normed=True,alpha=0.8, color='r', label='Third Time')
plt.hist(donation_count_df['Donation Amount'][donation_count_df['Timer']=="Multiple"], range=[0,100],bins=20, histtype='stepfilled', normed=True, alpha=0.5,color='yellow', label='Multiple')
plt.title("Donation Amounts & Donation Time")
plt.xlabel("Donation Amount")
plt.ylabel("Probability")
plt.legend()
plt.show()


# **Need help** with splitting dataframe on a condition when making a plot.  Here, I only know how to make two dataframes based on the condition and make separate plots.  Any help is appreciated.

# In[60]:


no_option = donation_count_df[donation_count_df['Donation Included Optional Donation']=="No"]
yes_option = donation_count_df[donation_count_df['Donation Included Optional Donation']=="Yes"]


# In[61]:


plt.hist(no_option['Donation Amount'][no_option['Timer']=="First"], range=[0,100],bins=20, histtype='stepfilled', normed=True, color='b', label='First Time')
plt.hist(no_option['Donation Amount'][no_option['Timer']=="Second"], range=[0,100],bins=20, histtype='stepfilled', normed=True, alpha=0.8,color='g', label='Second Time')
plt.hist(no_option['Donation Amount'][no_option['Timer']=="Third"], range=[0,100],bins=20, histtype='stepfilled', normed=True,alpha=0.8, color='r', label='Third Time')
plt.hist(no_option['Donation Amount'][no_option['Timer']=="Multiple"], range=[0,100],bins=20, histtype='stepfilled', normed=True, alpha=0.5,color='yellow', label='Multiple')
plt.title("Donation Amounts & Donation Time For No Optional Donation")
plt.xlabel("Donation Amount")
plt.ylabel("Probability")
plt.legend()
plt.show()


# In[62]:


plt.hist(yes_option['Donation Amount'][yes_option['Timer']=="First"], range=[0,100],bins=20, histtype='stepfilled', normed=True, color='b', label='First Time')
plt.hist(yes_option['Donation Amount'][yes_option['Timer']=="Second"], range=[0,100],bins=20, histtype='stepfilled', normed=True, alpha=0.8,color='g', label='Second Time')
plt.hist(yes_option['Donation Amount'][yes_option['Timer']=="Third"], range=[0,100],bins=20, histtype='stepfilled', normed=True,alpha=0.8, color='r', label='Third Time')
plt.hist(yes_option['Donation Amount'][yes_option['Timer']=="Multiple"], range=[0,100],bins=20, histtype='stepfilled', normed=True, alpha=0.5,color='yellow', label='Multiple')
plt.title("Donation Amounts & Donation Time For Included Optional Donation")
plt.xlabel("Donation Amount")
plt.ylabel("Probability")
plt.legend()
plt.show()


# This is interesting.  It seems that donation amounts might be slightly higher with no optional donation included.

# In[87]:


plt.scatter(donation_count_df['Donor Cart Sequence'],donation_count_df['Donation Amount'])
plt.show()


# In[92]:


fig, ax = plt.subplots()

ax.grid(True,linestyle='-',color='0.75')
ax.set_title('Donation Amount by Cart Sequence')
ax.plot(donation_count_df['Donor Cart Sequence'],donation_count_df['Donation Amount'],'o')

ax.legend(numpoints=1, loc='upper right')
ax.set_xlim([0, 7500])


# Is it true that the longer the donor uses the cart the less they donate?

# In[95]:


fig, ax = plt.subplots()

ax.grid(True,linestyle='-',color='0.75')
ax.set_title('Donation Amount by Cart Sequence')
ax.plot(donation_count_df['Donor Cart Sequence'],donation_count_df['Donation Amount'],'o')

ax.legend(numpoints=1, loc='upper right')
ax.set_ylim([0,10000])


# Either bigger donations are done early in the cart sequence or the majority of donors do not spend a lot of time in the cart?

# ## Conclusion
# 
# Majority of donations are from first time donors, but there are considerable multiple time donors (2 or more).  The distributions of donation amounts are highly skewed due to some outliers in the high donation amount range.  This is good to keep in my mind if trying to use a linear model later.  Most donations include something called a 'optional donation'.
