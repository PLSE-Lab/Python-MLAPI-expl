#!/usr/bin/env python
# coding: utf-8

# I'm interested in finding out if there's a relationship between having programming background and having taken statistics. First, though, I'll need to read in my data.

# In[1]:


# import our libraries
import scipy.stats # statistics
import pandas as pd # dataframe

# read in our data
surveyData = pd.read_csv("../input/5day-data-challenge-signup-survey-responses/anonymous-survey-responses.csv")


# Now let's do a chi-square test! The chisquare function from scipy.stats will only do a one-way comparison, so let's start with that.

# In[2]:


surveyData["Have you ever taken a course in statistics?"].value_counts()


# In[10]:


digimon = pd.read_csv("../input/digidb/DigiDB_digimonlist.csv")
digimon


# In[11]:


digimon["Attribute"].value_counts()


# In[12]:


digimon["Stage"].value_counts()


# In[3]:


# first let's do a one-way chi-squared test for stats background
scipy.stats.chisquare(surveyData["Have you ever taken a course in statistics?"].value_counts())


# In[14]:


print(scipy.stats.chisquare.__doc__)


# In[16]:


scipy.stats.chisquare(digimon["Attribute"].value_counts())


# In[18]:


digimon_two_categorical_values = pd.crosstab(digimon["Attribute"], digimon["Stage"])


# In[19]:


scipy.stats.chi2_contingency(digimon_two_categorical_values)


# Statistic here is the chi-square value (larger = more difference from a uniform distrobution) and pvalue is the p-value, which is very low here.

# In[4]:


surveyData["Do you have any previous experience with programming?"].value_counts()


# In[5]:


# first let's do a one-way chi-squared test for programming background
scipy.stats.chisquare(surveyData["Do you have any previous experience with programming?"].value_counts())


# In[ ]:





# And, again, our p-value is very low. This means that we can be sure, for both these questions, that the people who answered them are not drawn from a pool of people who are uniformly likely to have chosen each answer.

# Now let's do a two-way comparison. Is there a relationship between having programming background and having taken statistics?

# In[27]:


# now let's do a two-way chi-square test. Is there a relationship between programming background 
# and stats background?

contingencyTable = pd.crosstab(surveyData["Do you have any previous experience with programming?"],
                              surveyData["Have you ever taken a course in statistics?"])

contingencyTable
scipy.stats.chi2_contingency(contingencyTable)


# Here, the first value (16.827) is the $\chi^2$ value, the second value (0.032) is the p-value and the third value (8) is the degrees of freedom. Since our p-value is under our alpha of 0.05, we can say that it seems unlikely that there *isn't* a connection between these two things, right?
# 
# BUT! Becuase we have preformed three tests of statistical significance we need to correct for the fact that the probability that we're going to get a significant effect just by chance increases with each test. (If you set your alpha to 0.05, you'll be wrong just by chance 1/20 times, so if you preform 20 test you're very likely to get the wrong answer on one of them & you need to correct for that.) We can do by dividing our alpha by x, where x is the number of tests we have preformed. So in this case, our p-value would have to be below a value of 0.016 to have an overall alphs of 0.05.
# 
# TL;DR because we did three tests, this final result is not significant at alpha = 0.05. 
