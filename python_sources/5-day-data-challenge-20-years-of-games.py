#!/usr/bin/env python
# coding: utf-8

# 

# # 5 day data challenge: 20 years of games
# 
# [**Day 1 - Reading data into kernel**](#Day-1---Reading-data-into-kernel)  
# [**Day 2 - Plot a Numeric Variable with a Histogram**](#Day-2---Plot-a-Numeric-Variable-with-a-Histogram)  
# [**Day 3 - Perform a t-test**](#Day-3---Perform-a-t-test)  
# [**Day 4 - Visualize categorical data with a bar chart**](#Day-4---Visualize-categorical-data-with-a-bar-chart)  
# [**Day 5 - Using a Chi-Square Test**](#Day-5---Using-a-Chi-Square-Test)

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ### **Day 1 - Reading data into kernel**

# In[2]:


games = pd.read_csv("../input/ign.csv")
games.head()


# **describe for numerical data**

# In[3]:


games.describe()


# **describe for categorical data**

# In[4]:


games.describe(include = ['O'])


# ### **Day 2 - Plot a Numeric Variable with a Histogram**

# In[5]:


import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
plt.hist(games['score'], bins=19)
plt.xlim([0,11])   
plt.xlabel('Score')
plt.ylabel('Counts')
plt.title('Distribution of score for all games')
plt.grid(linestyle='dotted')
plt.show()


# The scores are not normally distributed, there are much less games with a score above 9.  
# The probplot (qqplot) from scipy.stats shows this in more detail,  
# comparing the distribution of quantiles with a normal distribution (red line)

# In[6]:


from scipy.stats import probplot # for a qqplot
import pylab
probplot(games["score"], dist="norm", plot=pylab)  
plt.show()


# ### **Day 3 - Perform a t-test**

# The t-test can be used, for example, to determine if two sets of data are significantly different from each other.  
# It compares the means of a continuous variable from two groups.  
# The Null Hypothesis is that there is no statistical difference between the means of the two groups.
# 
# Here, we apply the t-test on these two data sets:
# scores for PS2 games and scores for Xbox360 games

# In[7]:


from scipy.stats import ttest_ind

scores_PS2 = games["score"][games["platform"] == "PlayStation 2"]
scores_Xbox360 = games["score"][games["platform"] == "Xbox 360"]

ttest_ind(scores_PS2, scores_Xbox360, equal_var=False)


# In[8]:


print(scores_PS2.mean())
print(scores_PS2.std())
print(scores_Xbox360.mean())
print(scores_Xbox360.std())


# Although the mean scores are quite close and well within the standard deviation, the two datasets are significantly different:  
# The p value of approx. 5e-05 is the probability that they are from the same distribution.

# In[9]:


plt.figure(figsize=(7,5))
plt.hist(scores_PS2, alpha=0.6, bins=19, label="PS2")
plt.hist(scores_Xbox360, alpha=0.6, bins=19, label="Xbox360")
plt.xlim([0,11])   
plt.xlabel('Score')
plt.ylabel('Counts')
plt.title('Distribution of score for PS2 and Xbox360 games')
plt.grid(linestyle='dotted')
plt.legend()
plt.show()


# In[10]:


probplot(scores_PS2 , dist="norm", plot=pylab)
plt.show()


# In[11]:


probplot(scores_Xbox360 , dist="norm", plot=pylab)
plt.show()


# ### **Day 4 - Visualize categorical data with a bar chart**

# In[12]:


games['score_phrase'].value_counts()


# In[13]:


import seaborn as sns
fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
sns.countplot(games['score_phrase'],ax=ax)
plt.xticks(rotation=45)
plt.show()


# In[14]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
sns.barplot(x='release_year', y='score', data=games, ax=ax)
plt.xticks(rotation=45)
plt.show()


# ### **Day 5 - Using a Chi-Square Test**

# The Chi2 test of independence is used to determine if two categorical variables are related.  
# The Null Hypothesis is that there is no relationship between the two variables, i.e. they are independent.
# 
# Here, we test if the score_phrase for a game is related to the game genre.   
# What game genres are in the dataset?

# In[15]:


games['genre'].value_counts().head(10)


# First, let's look at the score_phrases for Adventure and Racing games. 

# In[16]:


advt_racg = games[(games.genre == 'Adventure') | (games.genre == 'Racing')]
contingencyTable = pd.crosstab(advt_racg.score_phrase, advt_racg.genre, margins=True)
contingencyTable


# The absolute number of Adventure and Racing games is quite similar.  
# However, the contingency table shows different numbers for the score_phrase, especially for good and great games.  
# So, we might expect that the variable could be related to the genre.  
# Let's now look what the chi2 test says.

# In[17]:


from scipy import stats
chi2, p, dof, expctd = stats.chi2_contingency(contingencyTable)
print("chi2 :", chi2)
print("p value :", p)


# With the p value of approx 0.17 being larger than 0.05 we can conclude that the Null Hypothesis holds,  
# meaning for Adventure and Racing games the score_phrase is independent of the game genre.

# What about Adventure and Strategy games?

# In[18]:


advt_strg = games[(games.genre == 'Adventure') | (games.genre == 'Strategy')]
contingencyTable_2 = pd.crosstab(advt_strg.score_phrase, advt_strg.genre, margins=True)
contingencyTable_2


# This time the numbers seem to be even more different, for good and great games, but also for Masterpiece and Mediocre.  
# Again, the total number of Adventure and Strategy games is about the same.

# In[19]:


chi2, p, dof, expctd = stats.chi2_contingency(contingencyTable_2)
print("chi2 :", chi2)
print("p value :", p)


# This time,  the Null Hypothesis is wrong, with the p value of approx. 9e-4 being well below 0.05.  
# So, the two variables (score_phrase and genre) are not independent for Adventure and Strategy games.

# In[ ]:




