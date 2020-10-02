#!/usr/bin/env python
# coding: utf-8

# **Objective:**  The objective of this analysis is to determine whether smokers have statistically higher mean individual medical costs billed by health insurance than do non-smokers.  Furthermore, is a person's BMI correlated with individual medical costs billed by health insurance?

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew, stats
from math import sqrt
from numpy import mean, var


# In[ ]:


data = pd.read_csv("../input/insurance2.csv")
print(data.head())


# In[ ]:


print("Summary Statistics of Medical Costs")
print(data['charges'].describe())
print("skew:  {}".format(skew(data['charges'])))
print("kurtosis:  {}".format(kurtosis(data['charges'])))
print("missing charges values: {}".format(data['charges'].isnull().sum()))
print("missing smoker values: {}".format(data['smoker'].isnull().sum()))


# In[ ]:


f, axes = plt.subplots(1, 2)
sns.kdeplot(data['charges'], bw=10000, ax=axes[0])
sns.boxplot(data['charges'], ax=axes[1])
plt.show()


# There are 1338 observations in this dataset.  Both the boxplot and kernel density estimation plot reveal that the charges data is right skewed.  Furthermore, there are some outliers but no missing charges and smoker values.

# **Objective Part 1:**  Do smokers have statistically higher mean individual medical costs billed by health insurance than do non-smokers?

# In[ ]:


#prepare our 2 groups to test
smoker = data[data['smoker']==1]
non_smoker = data[data['smoker']==0]


# In[ ]:


plt.title('Distribution of Medical Costs for Smokers Vs Non-Smokers')
ax = sns.kdeplot(smoker['charges'], bw=10000, label='smoker')
ax = sns.kdeplot(non_smoker['charges'], bw=10000, label='non-smoker')
plt.show()


# In[ ]:


plt.title('Distribution of Medical Costs for Smokers Vs Non-Smokers')
ax = sns.boxplot(x="smoker", y="charges", data=data)


# The boxplots and kernel density estimation plots reveal that the 2 datasets are likely different.

# In[ ]:


statistic, pvalue = stats.ttest_ind(non_smoker['charges'], smoker['charges'], equal_var = False)
print("2 sample, 2 sided t-test pvalue:  {} t-stat: {}".format(pvalue,statistic))


# In[ ]:


# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	n1, n2 = len(d1), len(d2)
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	u1, u2 = mean(d1), mean(d2)
	return (u1 - u2) / s
	
d = cohend(smoker['charges'], non_smoker['charges'])
print("cohen's d:  {}".format(d))


# Results from the 2 sample, 2 sided t test indicate that non-smokers have significantly less mean individual medical costs billed by health insurance than do smokers.  Furthermore, Cohen's D indicates that the difference between the means is more than 3 standard deviations which is interpreted as a large effect size.

# **Objective Part 2:**  Is a person's BMI correlated with individual medical costs billed by health insurance?

# In[ ]:


plt.title("BMI Versus Charges")
ax = sns.scatterplot(x="bmi", y="charges", data=data)
plt.show()


# In[ ]:


data.bmi.corr(data.charges)


# The scatterplot and correlation coefficient both reveal that bmi and charges have a very weak correlation.  However, for charges larger than a specified amount, there might be a stronger correlation.

# In[ ]:


def corr_converge(data=data):
    for i in range(0,60000,1000):
        data_new = data[data['charges'] >= i]
        print("lower bound: {} \t correlation coefficient: {} \t number of observations: {}".format(i,data_new.bmi.corr(data_new.charges),len(data_new)))
        pass
    
corr_converge()


# In[ ]:


data_new = data[data['charges']>=21000]
plt.title("BMI Versus Charges Greater Than 21000")
ax = sns.scatterplot(x="bmi", y="charges", data=data_new)
plt.show()


# In[ ]:


data_new.bmi.corr(data_new.charges)


# After examining the convergence of correlation coefficients, I looked at charges larger than 21,000 USD. The scatterplot and correlation coefficient reveal a "moderate" positive relationship between bmi and charges larger than 21,000 USD.

# **Results:**  Smokers have statistically higher mean individual medical costs billed by health insurance than do non-smokers.  Furthermore, a person's BMI is correlated with charges amounts greater than or equal to 21,000 USD.  Thus, for this group as bmi increases so does the individual medical costs billed by health insurance.
