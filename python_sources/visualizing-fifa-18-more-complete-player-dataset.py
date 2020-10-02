#!/usr/bin/env python
# coding: utf-8

# In[37]:


# import useful libraries
import numpy as np 
import pandas as pd

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Data cleaning and preparation

# In[3]:


# Check the encoding of the dataset and then decode it
import chardet
with open("../input/complete.csv", "rb") as f:
    result = chardet.detect(f.read(100000))
print(result)


# In[7]:


complete_df = pd.read_csv("../input/complete.csv", encoding="utf-8", index_col=0)
complete_df.head()


# # 2. Some data visualization

# In[80]:


# I will merely select some columns that I am interested in
interesting_columns = ["name", "age", "height_cm", "weight_kg", 
                       'eur_value', 'eur_wage', 'eur_release_clause', 
                       'overall', 'potential', 'international_reputation']
df = complete_df[interesting_columns].copy()
# create a column expressing the remaining potential of players
df["remaining_potential"] = df["potential"] - df["overall"]
df.head()


# ## 2.1. Visualizing player values

# ### 2.1.1. Top 10 players with highest value

# In[26]:


df.sort_values(by="eur_value", ascending=False).head(10)[["name", "eur_value"]]


# ### 2.1.2. Top 10 players with highest value of release clause

# In[27]:


df.sort_values(by="eur_release_clause", ascending=False).head(10)[["name", "eur_release_clause"]]


# ### 2.1.3. Top 100 players value distribution

# In[68]:


top100 = df.sort_values(by="eur_value", ascending=False).head(100)
sns.distplot(top100["eur_value"] / 1e6, 
             kde_kws=dict(cumulative=True))
plt.title("Top 1000 player value distribution")
plt.xlabel("Value (in millions euro)")
plt.ylabel("CDF")
plt.show()


# ## 2.2. Plot the age PMF of all players

# In[56]:


sns.distplot(df["age"], kde=False, fit=stats.gamma)
plt.title("Age distribution of all players")
plt.xlabel("Age")
plt.ylabel("Probability")
plt.show()


# The above figure only gives us an overview of the age distribution. For deeper visualization, we can display the age distribution based on the preferred position of player. For example, I will plot the age distribution of Central Back (CB) players. 

# In[55]:


cbs = complete_df[complete_df["prefers_cb"] == True]

sns.distplot(cbs["age"], kde=False, fit=stats.gamma)
plt.title("Age distribution of all goalkeepers")
plt.xlabel("Age")
plt.ylabel("Probability")
plt.show()


# ## 2.3. Scatter plot the age and the remaining potential of players

# In[70]:


plt.scatter(df["age"], df["remaining_potential"])
plt.title("Age by remaining potential")
plt.xlabel("Age")
plt.ylabel("Remaining potential")
plt.show()


# If we would like to make the age by remaining potential plot more cleaning (for example, I would like to see a line instead of a bunch of points like this plot), we can first group the remaining potential by age. Afterwards, we create the age by mean remaining potential of each group. 

# In[77]:


age = df.sort_values("age")['age'].unique()
remaining_potentials = df.groupby(by="age")["remaining_potential"].mean().values


# In[79]:


plt.title("Age vs remaining potential")
plt.xlabel("Age")
plt.ylabel("Remaining potential")
plt.plot(age, remaining_potentials)
plt.show()


# ## 2.4. Overall and Potential skill point vs International Reputation

# In[89]:


overall_skill_reputation = df.groupby(by="international_reputation")["overall"].mean()
potential_skill_reputation = df.groupby(by="international_reputation")["potential"].mean()


# In[105]:


plt.plot(overall_skill_reputation, marker='o', c='r', label='Overall Skillpoint')
plt.plot(potential_skill_reputation, marker='x', c='b', label='Potential Skillpoint')
plt.title('Overall, Potential vs Reputation')
plt.xlabel('Reputation')
plt.ylabel('Skill point')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




