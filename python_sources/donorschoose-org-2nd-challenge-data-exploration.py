#!/usr/bin/env python
# coding: utf-8

# ![http://www.dropbox.com/s/dgmag55pxxv5nsu/donorsChoose.png](http://www.donorschoose.org/)
# 
# In this Kaggle Challenge, we are trying to understand the characteristics of donations and donors of **Donors Choose**. It is a non-profit organization that works on improving the quality of classes in US.
# 
# I organized this kernel according to a Question and Answers approach as I look into data. Let's see some insights below!

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# # Exploring donations data (per si)

# In[3]:


donations = pd.read_csv("../input/Donations.csv")
donations.dropna(inplace=True)
donations.head()


# ## Questions & Answers Regarding Donations
# 
# 1. **What is the most frequent donation amount?**
# 
#       _Answer_: First of all,  the most frequent donations are under 1K dollars. Less than 0.000002% donations are upper 10K dollars. If a donation is going to be made, it is more likely to be from a lower value, maybe from a common person that sees that such help is not going to affect his/hers finances. 
#        
# 2. **From the donations under 1K dollars, what is their distribution?**
#      
#       _Answer_: We can see a more strong commitment on donations under 200 dollars. They make up 60% of all donated amount. Taking a even closer look on typical donations, we see that donations from 25, 50 or 100 dollars are more frequent.
#       
# 3. ** If a new donation is made,  how much it will be**?
#  
#      _Answer_: With more than 90% certainty, it will be of less than 200 dollars. Being more specific, we can consider donations of more than 100 dollars as atypical. It can help us focus on gathering new potential donors as showing them that even small amount donations can make a big difference.
#    

# In[ ]:


print(len(donations.loc[donations['Donation Amount'] < 1000])/len(donations))
print(len(donations.loc[donations['Donation Amount'] >= 10000])/len(donations))


# In[6]:


donationsUnder1K = (donations.loc[donations['Donation Amount'] < 1000])['Donation Amount']


# In[26]:


fig = plt.figure()
plt.title("Donations under 1K dollars")
ax = plt.gca()
ax.hist(donationsUnder1K,bins=50)
plt.xlabel("Donation Amount")
plt.ylabel("Count")
plt.show()


# In[25]:


# Closer look on donations under 200 dolars
donationsUnder200 = (donations.loc[donations['Donation Amount'] < 200])['Donation Amount']
fig = plt.figure()
plt.title("Donations under 200 dollars")
ax = plt.gca()
ax.hist(donationsUnder200,bins=20)
plt.xlabel("Donation Amount")
plt.ylabel("Count")
plt.show()


# In[22]:


sum(donationsUnder200)/sum(donations['Donation Amount'])


# In[24]:


# Closer look on donations under 1K by their frequency
results, edges = np.histogram(donationsUnder1K, normed=True,bins=20)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth)
plt.title("Probability distribution on donations")
plt.xlabel("Donation Amount")
plt.ylabel("Probability")
plt.show()


# In[32]:


# Even closer look on under 200 dollars donations
plt.title("Donation Amount Distribution Under 200 dollars")
ax = plt.gca()
ax.boxplot(donationsUnder200,vert=False)
plt.xlabel("Donation Amount")
plt.show()

