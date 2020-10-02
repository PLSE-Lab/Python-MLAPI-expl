#!/usr/bin/env python
# coding: utf-8

# # INDIA
# 
# I'm flexing my heatmap muscles in this notebook. Thus almost every other thing will be a heatmap. First off we take a look at the population via states. This is census data after all.  

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


df = pd.read_csv('../input/all.csv', index_col=0)
df.info()


# There are a lot of columns to choose from. To allow me to cheat a little, I'm going to use a boxplot here because I cannot think of any sane heatmap which can represent what this can.

# In[ ]:


plt.figure(figsize=(10, 7))
ax = plt.gca()
sns.boxplot(x='State', y='Persons', data=df, linewidth=1)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
plt.title('Population in the "States". <pun intended>')
#plt.savefig('population_per_state.png')


# Delhi is almost the same as Maharashtra and the difference in area is amazing! Delhi is 1,484  and Maharashtra is 307,713 square kilometers.
# 
# The education system in India is comprised of a lot of moving parts. Let's see how the people in states are distributed over these education brackets.
# 
# Most of the education brackets are self explanatory.

# In[ ]:


education_cols = ['Below.Primary', 'Primary', 'Middle', 'Matric.Higher.Secondary.Diploma',
                'Graduate.and.Above']
temp = df[education_cols + ['State']].groupby('State').sum()

plt.figure(figsize=(4, 7))
sns.heatmap(np.round(temp.T / temp.sum(axis=1), 2).T, cmap='gray_r',
            linewidths=0.01, linecolor='white', annot=True)
plt.title('Which state has what fraction of people in what bracket?')


# States like Delhi, Manipur, Chandigarh have a lot of highly educated people. A large fraction of Meghalaya is below primary though. Lots of kids? or perhaps lots of people who never got a chance to go to school because of the harsh mountains?
# 
# Age is another good thing to measure. Where are all the kids and all the old people? Which state houses them?

# In[ ]:


age_cols = ['X0...4.years','X5...14.years',
            'X15...59.years','X60.years.and.above..Incl..A.N.S..']
temp = df[age_cols+['State']].groupby('State').sum()

plt.figure(figsize=(15, 3))
ax = plt.gca()
temp.columns=['0 to 4 years', '5 to 14 years', '15 to 59 years', '60 years +']
sns.heatmap(np.round(temp / temp.sum(axis=0), 2).T, linecolor='white',
            linewidths=0.01, cmap='gray_r', annot=True)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
plt.title('Age group distribution over states')


# This ends up reflecting the population distribution of the states I suppose. Since almost every category ends up being in UP. 
# 
# 
# The working population of my country interests me to no end. Where are the hard workers? Where are those who would prefer not to work?

# In[ ]:


worker_cols = ['Main.workers', 'Marginal.workers', 'Non.workers']
temp = df[worker_cols+['State']].groupby('State').sum()

plt.figure(figsize=(5, 7))
temp2 = temp.T / temp.sum(axis=1) # What fraction of the group is in what state
sns.heatmap(np.round(temp2 / temp2.sum(axis=0), 2).T, linecolor='white',
            linewidths=0.01, cmap='gray_r', annot=True)
plt.title('Working class distribution over states. Rows sum to 1')


# States like Lakshwadeep, Jharkhand, Tripura and UP have among the lowest fraction of people who are main workers. It is hilarious that UP is lumped together with the islands of Lakshwadeep, jungles of Jharkhand and mountains of Tripura. It's bang in the middle of India and still manages such a low working fraction.
# 
# "Dadra and Nager Haveli"(D_N_H) and Mizoram are the only two states with the distinction of having < 50% people who are non workers.
# 
# It might be because of the "housewife syndrome". If we assume that half of the population is women, and all of them are married; it leads us to half of the population being non.workers because of being housewives.
# 
# Another major force in India is religion. Who worships whom and where do they worship?

# In[ ]:


religion_cols = ['Religeon.1.Name','Religeon.1.Population',
                 'Religeon.2.Name','Religeon.2.Population',
                 'Religeon.3.Name','Religeon.3.Population']
temp = df[religion_cols + ['State']].copy()
for i in '123':
    temp['Religeon.'+i+'.Name'] = temp['Religeon.'+i+'.Name'].str.split('.').str[-1]
temp2 = pd.DataFrame([], columns=['Name', 'Population', 'State'])
for i in '123':
    a = temp[['Religeon.'+i+'.Name', 'Religeon.'+i+'.Population', 'State']].copy()
    a.columns = ['Name', 'Population', 'State']
    temp2 = pd.concat([a, temp2])
grouped = temp2.groupby(['State', 'Name']).sum()
temp2 = grouped.reset_index().fillna(1)
ct = pd.crosstab(temp2.State, temp2.Name, temp2.Population, aggfunc=np.sum)
ct = ct.fillna(1)
plt.figure(figsize=(7, 7))
sns.heatmap(np.round(ct / ct.sum(axis=0), 2), cmap='gray_r', linecolor='black',
            linewidths=0.01, annot=True)
plt.title('Which religion resides in which state? (Columns sum to 1)')


# Almost all the Budhists in the country reside in Maharashtra.  Bihar has been rounded to 0 despite being the state which has [Bodh Gaya ](https://en.wikipedia.org/wiki/Bodh_Gaya). Christians are spread out all over the country, as are Hindus and Muslims.
# 
# Jains as expected are spread across Rajasthan, Gujarat and Madhya Pradesh. Since Jharkhand holds the largest fraction of "Others" one can safely assume that "Others" represent the indigenous tribal religions of India.
# 
# One would say that "Religion not stated"  would be atheists, but I have never heard of an atheist from Bihar. Karnataka does have Bangalore lending it a view of the rest of the world but Bihar has never been associated with world views in public opinion other than [Nalanda](https://en.wikipedia.org/wiki/Nalanda) 
# 
# Sikhs are concentrated in Punjab and the states around it. This is as I expected.

# In[ ]:


plt.figure(figsize=(7,7))
sns.heatmap(np.round(ct.T / ct.sum(axis=1), 2).T, cmap='gray_r',
            linecolor='black', linewidths=0.01, annot=True)
plt.title('States have what fraction of which religion?(Rows sum to 1)')


# Hindus are literally the majority everywhere. Punjab as expected is a Sikh state, but even there the Sikh only comprise of 59% of the population. The north east has it's fair share of states composed mainly of Christians.
# 
# The state of Jammu and Kashmir is a mostly Muslim states as are the Lakshwadeep islands. Arunachal Pradesh has a good fraction of it's people as "Others", nearly equaling the Hindu population.
# 
# "Religion not Stated" actually got completely rounded to zero. If you are in India and you meet an atheist, **you have met a very rare person***.
# 
# Punjab and it's neighbors Chandigarh and Haryana are having a healthy fraction of Sikhs. 
