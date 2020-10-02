#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Analysis of "120 Years of Olympics History Athletes and Results"

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


regions.head()


# In[ ]:


merged = pd.merge(data, regions, on='NOC', how='left')


# In[ ]:


merged.head()


# ### Lets analyze the gold medal winners of all times

# In[ ]:


goldMedals = merged[merged.Medal == 'Gold']
goldMedals.head()


# In[ ]:


goldMedals.isnull().any()


# #### Thus, the features like region, notes, age, height and weight have missing values. We need to consider that case while evaluating on them.

# In[ ]:


goldMedals = goldMedals[np.isfinite(goldMedals['Age'])]


# ### Gold Medal Winners Ages Distribution : Maximum of them are 23/24 years old!

# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
p1 = sns.countplot(goldMedals['Age'])
p1.set_xticklabels(p1.get_xticklabels(),rotation=45)
plt.title('Distribution of Gold medals with Athlete Age!')


# ### Gold Medal Winners Heights Distribution 
# #### It is not balanced since some sports don't have hieght as their primary requirement or factor.

# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(goldMedals['Height'])
plt.tight_layout()
plt.title('Distribution of Heights')


# ### Some athletes with gold medals are above the age of 50!

# In[ ]:


masterDisciplines = goldMedals['Sport'][goldMedals['Age'] > 50]


# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
p2 = sns.countplot(masterDisciplines)
plt.title('Sports for athletes over age 50')


# ### Our senior sportmen excel in the following sports
# #### Equestrianism, Sailing, Alpinism,  Art Competetions, Curling Sport, Roque, Shooting, Archery, Croquet, etc.
# ### Clearly, the above sports don't require high physical requirements.

# ### Women Athletes Analysis

# In[ ]:


women = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]


# In[ ]:


women.head()
women.shape


# ### Women Medals Evolution Distribution! Glad to see that women have evolved over years in sports!

# In[ ]:


sns.set(style = 'darkgrid')
plt.figure(figsize=(20,10))
sns.countplot(women['Year'])
plt.title('Women Medals Evolution')


# ### Countries that dominate Olympics medals : USA and Russia!

# In[ ]:


goldMedals.region.value_counts().head()


# ### No Doubt USA slays in Basketball! 186 Gold Medals!

# In[ ]:


goldMedalsUSA = goldMedals.loc[goldMedals['NOC'] == 'USA']


# In[ ]:


goldMedalsUSA.Event.value_counts().head(20)


# ### USA Basketball players with maximum number of Gold Medals!

# In[ ]:


basketballGoldUSA = goldMedalsUSA.loc[(goldMedalsUSA['Sport'] == 'Basketball') & (goldMedalsUSA['Sex'] == 'M')].sort_values(['Year'])


# In[ ]:


basketballGoldUSA.head(15)


# In[ ]:


groupedBasketUSA = basketballGoldUSA.groupby(['Year']).first()


# In[ ]:


groupedBasketUSA.head()


# ### Height-Weight Correlation of Olympic Medalists

# In[ ]:


notNullMedals = goldMedals[(goldMedals['Height'].notnull()) & (goldMedals['Weight'].notnull())]


# In[ ]:


notNullMedals.head()


# In[ ]:


plt.figure(figsize=(20,20))
ax = plt.scatter(x='Height', y='Weight', data=notNullMedals)
plt.title('Height vs Weight of Olympics Medalists')


# ### Some Medalists with Extreme Heights and Weigths

# ### 1) Medalists with more than 150 Kilograms of weight!

# In[ ]:


notNullMedals.loc[notNullMedals['Weight'] > 150].head()


# ### 2) Medalists with Height greater than 7 Feet. Whoa!

# In[ ]:


notNullMedals.loc[notNullMedals['Height'] > 215].head()


# ### 2) Medalists with Height lesser than 4 Feet. Whoa Again!

# In[ ]:


notNullMedals.loc[notNullMedals['Height'] < 140].head()


# ### 4) Wang Xin (China) was just 28 Kilograms to win Gold in Beijing Olympics for Synchronized Diving Event!

# In[ ]:


notNullMedals.loc[notNullMedals['Weight'] < 30].head()


# ### Women Gold Medal Winners Height Distribution 

# In[ ]:


womenData = merged[(merged.Sex == "F") & (merged.Sport == "Athletics")]
plt.figure(figsize=(20,10))
sns.countplot(womenData['Height'])
plt.tight_layout()
plt.title("Women Height Distribution in Athletics")


# ### Some insights from Indian Athletes at Olympics

# In[ ]:


indianData = merged[(merged.Team == "India") & (merged.Season == "Summer")]
plt.figure(figsize=(20,10))
sns.countplot(indianData['Year'])
plt.tight_layout()
plt.title('Indian Gold Medal Events')


# In[ ]:


indiansInOlympics = merged[merged.Team == "India"]
indiansInOlympics.head()


# ### Medal Ratio of Indian Athletes at Olympics

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(indiansInOlympics['Medal'])
plt.tight_layout()
plt.title('Medal Ratio of Indian Olympians')


# ### Evolution of Medals from Indian Athletes : 2016 Most Valuable year in Indian Olymipcs History!

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(indiansInOlympics['Year'])
plt.tight_layout()
plt.title('Medal evolution of Indian Olympians')


# ### Gender wise Indian Gold medals ratio 

# In[ ]:


plt.figure(figsize=(20,10))
goldIndia = indiansInOlympics[indiansInOlympics.Medal == "Gold"]
sns.countplot(goldIndia['Sex'])
plt.tight_layout()
plt.title('Medal Ratio of Indian Olympians')


# ### Unfortunately, there have been no gold medals from Indian women in the Olympics history!

# ## There could be many more interesting insigts drawn from the rich Olympics history data! I would keep updating this kernel with more analysis as I work more on this dataset.
