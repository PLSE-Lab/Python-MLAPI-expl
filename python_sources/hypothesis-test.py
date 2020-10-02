#!/usr/bin/env python
# coding: utf-8

# Hypothesis Test for Gender Differences in Money saving
# ------------------------------------------------------

# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# Import data 
dta_all = pd.read_csv('../input/responses.csv')


# We want to observe there is significant difference between younger females and males in terms of money saving habit.  In other way, we can say that we are testing the association between gender and saving habit. Here I apply the Chi-sq test, because the two variables are categorical, so we are comparing their proportions. <br />
# 
# **Research questions: Is there any gender differences money saving (finances)?**<br />
# 
# H0: There is no correlation between genders and finances. 
# H1: females and males have different perspectives in finances. 
# 

# In[ ]:


test = pd.DataFrame()
def table_building(row, col):
    test = pd.crosstab(index=row,columns=col,margins=True)
    test.columns = ["1.0","2.0","3.0","4.0","5.0","rowtotal"]
    return(test);
    
def chisq_test(t, i):
    # Get table without totals for later use
    observed = t.ix[0:2,0:5]   
    #To get the expected count for a cell.
    expected =  np.outer(t["rowtotal"][0:2],t.ix["All"][0:5])/1010
    expected = pd.DataFrame(expected)
    expected.columns = ["1.0","2.0","3.0","4.0","5.0"]
    expected.index= test.index[0:2]
    #Calculate the chi-sq statistics
    chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
    print("Chi-sq stat")
    print(chi_squared_stat)
    crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = i)   # *
    print("Critical value")
    print(crit)
    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=i)
    print("P value")
    print(p_value)
    return;


# In[ ]:


test = table_building(dta_all["Gender"],dta_all["Finances"])
print(test)
chisq_test(test,4)


# As per the results, the chi-sq test statistics is smaller than the critical value, and the p-value is larger than 0.05. So we fail to reject the non-hypothesis at 95% confidence level. In sum, there is no enough evidence to show females and males have different money saving habits.
# Second, chi-sq result is telling us, the distribution of male and female in each level of saving habit are closed, and we can observe this pattern from histogram below. 

# In[ ]:


sns.countplot(x='Finances', hue = 'Gender', data = dta_all)


# The similar test also can be applied to other research questions, for example, 'Is there any differences in money saving (finances) between people from city or village?', 

# In[ ]:


test = table_building(dta_all["Village - town"],dta_all["Finances"])
print(test)
chisq_test(test,4)


# As per the results, the chi-sq test statistics is larger than the critical value, and the p-value is smaller than 0.05. So we reject the non-hyponthesis at 95% confidence level. This indicates that people from cities and people from villages have different point of view in money saving. From the table, it can be observed that the people from village is more conservative than people from city , which is sensible; people from cities are more liberal and are likely to have better income than village people that might affect their money spending habits.

# In[ ]:


sns.countplot(x='Finances', hue = "Village - town", data = dta_all)


# In[ ]:




