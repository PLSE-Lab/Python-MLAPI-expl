#!/usr/bin/env python
# coding: utf-8

# # Applying Inferential Statistics

# ### Hypotheses to test (formed during storytelling):
# 1. Null: Age of people who left the bank and who did not are similar. Alternative: Not similar.
# 2. Null: Credit score of people who left the bank and who did not are similar. Alternative: Not similar.
# 3. Null: Balance of people who left the bank and who did not are similar. Alternative: Not similar.
# 4. Null: Estimated Salary of people who left the bank and who did not are similar. Alternative: Not similar.
# 
# #### The most appropriate test to analyse data here is Frequentist test.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
from scipy.stats import t
from scipy.special import stdtr
from numpy.random import seed
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (8.0, 5.0)


# In[ ]:


file_1 = pd.read_csv('../input/churn-prediction-of-bank-customers/Churn_Modelling.csv')


# In[ ]:


df = pd.DataFrame(file_1)


# In[ ]:


df.head()


# In[ ]:


df_0 = df[df.Exited == 0]
df_1 = df[df.Exited == 1]


# ## Hypothesis 1: Age

# In[ ]:


sns.distplot(df_0.Age, color='blue', label='Still with Bank')
sns.distplot(df_1.Age, color='green', label='Left the Bank')
plt.legend()


# In[ ]:


df_0.Age.mean() , df_0.Age.std()


# In[ ]:


df_1.Age.mean() , df_1.Age.std()


# In[ ]:


t_1,p_1 = scipy.stats.ttest_ind(df_0.Age, df_1.Age, equal_var=False)
t_1, p_1


# ### Using Bootstrapping

# In[ ]:


def bs_choice(data, func, size):
    bs_s = np.empty(size)
    for i in range(size):
        bs_abc = np.random.choice(data, size=len(data))
        bs_s[i] = func(bs_abc)
    return bs_s


# In[ ]:


diff_means = np.mean(df_1.Age) - np.mean(df_0.Age)
mean_age = np.mean(df.Age)
age_shifted_0 = df_0.Age + mean_age - np.mean(df_0.Age)
age_shifted_1 = df_1.Age + mean_age - np.mean(df_1.Age)


# In[ ]:


bs_n_0 = bs_choice(age_shifted_0, np.std, 10000)
bs_n_1 = bs_choice(age_shifted_1, np.std, 10000)
bs_mean = bs_n_1 - bs_n_0


# In[ ]:


p = np.sum(bs_mean >= diff_means) / len(bs_mean)
p


# ### Conclusion
# We reject Null hypothesis. The probability of null hypothesis is almost zero which is less than significance level of 0.05.

# ## Hypothesis 2: Credit Score

# In[ ]:


sns.distplot(df_0.CreditScore, color='blue', label='Still with bank')
sns.distplot(df_1.CreditScore, color='green', label='Left the bank')
plt.legend()


# In[ ]:


t_2,p_2 = scipy.stats.ttest_ind(df_0.CreditScore, df_1.CreditScore, equal_var=False)
t_2, p_2


# ### Conclusion
# We reject Null hypothesis. The probability of null hypothesis is 0.0085 or 0.85 % which is less than significance level of 0.05.

# ## Hypothesis 3: Balance

# In[ ]:


sns.distplot(df_0.Balance, color='blue', label='Still with bank')
sns.distplot(df_1.Balance, color='green', label='Left the bank')
plt.legend()


# In[ ]:


t_3,p_3 = scipy.stats.ttest_ind(df_0.Balance, df_1.Balance, equal_var=False)
t_3, p_3


# In[ ]:


sns.distplot(df_0[df_0.Balance != 0].Balance, color='blue', label='Still with bank')
sns.distplot(df_1[df_1.Balance != 0].Balance, color='green', label='Left with bank')
plt.legend()


# In[ ]:


t_3,p_3 = scipy.stats.ttest_ind(df_0[df_0.Balance != 0].Balance, df_1[df_1.Balance != 0].Balance, equal_var=False)
t_3, p_3


# ### Conclusion
# The Balances of Zero are too many in the data. When we consider all the data, we reject Null Hypothesis.
# When we only remove the Balances which are Zero, the probability of null hypothesis becomes 19.06% which is significant. Then we reject Alternative Hypothesis.

# ## Hypothesis 4: Estimated Salary

# In[ ]:


sns.distplot(df_0.EstimatedSalary, color='blue', label='Still with bank')
sns.distplot(df_1.EstimatedSalary, color='green', label='Left with bank')
plt.legend()


# In[ ]:


t_3,p_3 = scipy.stats.ttest_ind(df_0.EstimatedSalary, df_1.EstimatedSalary, equal_var=False)
t_3, p_3


# ### Using Bootstrapping

# In[ ]:


diff_means = np.mean(df_1.EstimatedSalary) - np.mean(df_0.EstimatedSalary)
mean_salary = np.mean(df.EstimatedSalary)
salary_shifted_0 = df_0.EstimatedSalary + mean_salary - np.mean(df_0.EstimatedSalary)
salary_shifted_1 = df_1.EstimatedSalary + mean_salary - np.mean(df_1.EstimatedSalary)


# In[ ]:


bs_n_0 = bs_choice(salary_shifted_0, np.mean, 10000)
bs_n_1 = bs_choice(salary_shifted_1, np.mean, 10000)
bs_mean = bs_n_1 - bs_n_0


# In[ ]:


p = np.sum(bs_mean >= diff_means) / len(bs_mean)
p


# ### Conclusion
# We do not reject Null hypothesis. The probability of null hypothesis using t-test is 0.2416 or 24.16% and using bootstrapping is 0.1222 or 12.22% which is more than significance level of 0.05.

# ## Final Conclusion
# The variables CreditScore and Age will be most helpful in predicting churning. The variable Balance will also be helpful only in cases where Balance is zero.
# 

# In[ ]:





# In[ ]:




