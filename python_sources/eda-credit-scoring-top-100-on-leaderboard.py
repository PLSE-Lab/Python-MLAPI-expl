#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis - Preparing for one of the top performing models
# 
# In this notebook, an exploratory data analysis is performed on Give Me Some Credit's training set and preprocessing steps will be listed. These preprocessing steps will be the preparatory work for training a XGBoost model on the dataset, which is able to attain private and public scores of **0.86756** and **0.86104** respectively. The private and public scores are ranked top 100 and top 130 respectively (at the point of time of submitting this notebook).
# 
# More comprehensive README and Python scripts can be found at 
# 
# https://github.com/nicholaslaw/kaggle-credit-scoring
# 
# ## Table of Contents
# 
# 1. [Import Packages](#1)
# 2. [Import Data](#2)
# 3. [EDA](#3)
# 4. [Preprocessing Suggestions](#4)
# 5. [References](#5)

# ## Import Packages <a class="anchor" id="1"></a>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Import Data <a class="anchor" id="2"></a>

# In[ ]:


df = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")
df.head()


# ## EDA <a class="anchor" id="3"></a>
# 
# - Around 6% of samples defaulted
# - MonthlyIncome and NumberOfDependents have 29731 (19.82%) and 3924 (2.61%) null values respectively
# - We also notice that when NumberOfTimes90DaysLate has values above 17, there are 267 instances where the three columns NumberOfTimes90DaysLate, NumberOfTime60-89DaysPastDueNotWorse, NumberOfTime30-59DaysPastDueNotWorse share the same values, specifically 96 and 98.
#     - We can see that sharing the same values of 96 and 98 respectively is not logical since trivial calculations can reveal that being 30 days past due for 96 times for a single person within a timespan of 2 years is not possible.
# - RevolvingUtilizationOfUnsecuredLines
#     - Defined as ratio of the total amount of money owed to total credit limit
#     - distribution of values is right-skewed, consider removing outliers
#     - It is expected that as this value increases, the proportion of people defaulting should increase as well
#     - However, we can see that as the minimum value of this column is set to 13, the proportion of defaulters is smaller than that belonging to the pool of clients with total amount of money owed not exceeding total credit limit.
#     - Thus we should remove those samples with RevolvingUtilizationOfUnsecuredLines's value more than equal to 13
# - age
#     - There seems to be more younger people defaulting and the distribution seems fine on the whole
# - NumberOfTimes90DaysLate
#     - It is interesting to note that there are no one who is 90 or more days past due between 17 and 96 times.
# - NumberOfTime60-89DaysPastDueNotWorse
#     - It is interesting to note that there are no one who is 60-89 days past due between 11 and 96 times.
# - NumberOfTime30-59DaysPastDueNotWorse
#     - It is interesting to note that there are no one who is 30-59 days past due between 13 and 96 times.
# - DebtRatio
#     - 2.5% of clients owe around 3490 or more times what they own
#     - For the people who have monthly income in this 2.5%, only 185 people have values for their monthly incomes and the values are either 0 or 1.
#     - There are 164 out of these 185 people who are of two different types, first with no monthly income and does not default and second with monthly income and does default.
# - MonthlyIncome
#     - Distribution of values is skewed, we can consider imputation with median.
#     - We can also consider imputing with normally distributed values with its mean and standard deviation.
# - Numberof Dependents
#     - We can consider imputing with its mode, which is zero.

# ### Derive Balance of Classes

# In[ ]:


sns.countplot(x="SeriousDlqin2yrs", data=df)
print("Proportion of People Who Defaulted: {}".format(df["SeriousDlqin2yrs"].sum() / len(df)))


# ### Null Values and Proportions

# In[ ]:


null_val_sums = df.isnull().sum()
pd.DataFrame({"Column": null_val_sums.index, "Number of Null Values": null_val_sums.values,
             "Proportion": null_val_sums.values / len(df) })


# ### RevolvingUtilizationOfUnsecuredLines

# In[ ]:


df["RevolvingUtilizationOfUnsecuredLines"].describe()


# In[ ]:


sns.distplot(df["RevolvingUtilizationOfUnsecuredLines"])


# In[ ]:


default_prop = []
for i in range(int(df["RevolvingUtilizationOfUnsecuredLines"].max())):
    temp_ = df.loc[df["RevolvingUtilizationOfUnsecuredLines"] >= i]
    default_prop.append([i, temp_["SeriousDlqin2yrs"].mean()])
default_prop


# In[ ]:


sns.lineplot(x=[i[0] for i in default_prop], y=[i[1] for i in default_prop])
plt.title("Proportion of Defaulters As Minimum RUUL Increases")


# In[ ]:


print("Proportion of Defaulters with Total Amount of Money Owed Not Exceeding Total Credit Limit: {}"     .format(df.loc[(df["RevolvingUtilizationOfUnsecuredLines"] >= 0) & (df["RevolvingUtilizationOfUnsecuredLines"] <= 1)]["SeriousDlqin2yrs"].mean()))


# In[ ]:


print("Proportion of Defaulters with Total Amount of Money Owed Not Exceeding or Equal to 13 times of Total Credit Limit:\n{}"     .format(df.loc[(df["RevolvingUtilizationOfUnsecuredLines"] >= 0) & (df["RevolvingUtilizationOfUnsecuredLines"] < 13)]["SeriousDlqin2yrs"].mean()))


# ### age

# In[ ]:


df["age"].describe()


# In[ ]:


sns.distplot(df["age"])


# In[ ]:


sns.distplot(df.loc[df["SeriousDlqin2yrs"] == 0]["age"])


# In[ ]:


sns.distplot(df.loc[df["SeriousDlqin2yrs"] == 1]["age"])


# ### Late Payment Columns
# 
# - NumberOfTimes90DaysLate
# - NumberOfTime60-89DaysPastDueNotWorse
# - NumberOfTime30-59DaysPastDueNotWorse

# In[ ]:


late_pay_cols = ["NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTime30-59DaysPastDueNotWorse"]
df["NumberOfTimes90DaysLate"].value_counts().sort_index()


# In[ ]:


df["NumberOfTime60-89DaysPastDueNotWorse"].value_counts().sort_index()


# In[ ]:


df["NumberOfTime30-59DaysPastDueNotWorse"].value_counts().sort_index()


# In[ ]:


df.loc[df["NumberOfTimes90DaysLate"] > 17][late_pay_cols].describe()


# In[ ]:


distinct_triples_counts = dict()
for arr in df.loc[df["NumberOfTimes90DaysLate"] > 17][late_pay_cols].values:
    triple = ",".join(list(map(str, arr)))
    if triple not in distinct_triples_counts:
        distinct_triples_counts[triple] = 0
    else:
        distinct_triples_counts[triple] += 1
distinct_triples_counts


# ### DebtRatio

# In[ ]:


df["DebtRatio"].describe()


# In[ ]:


df["DebtRatio"].quantile(0.95)


# In[ ]:


df.loc[df["DebtRatio"] > df["DebtRatio"].quantile(0.95)][["DebtRatio", "MonthlyIncome", "SeriousDlqin2yrs"]].describe()


# In[ ]:


len(df[(df["DebtRatio"] > df["DebtRatio"].quantile(0.95)) & (df['SeriousDlqin2yrs'] == df['MonthlyIncome'])])


# In[ ]:


df.loc[df["DebtRatio"] > df["DebtRatio"].quantile(0.95)]["MonthlyIncome"].value_counts()


# In[ ]:


print("Number of people who owe around 2449 or more times what they own and have same values for MonthlyIncome and SeriousDlqin2yrs: {}"     .format(len(df.loc[(df["DebtRatio"] > df["DebtRatio"].quantile(0.95)) & (df["MonthlyIncome"] == df["SeriousDlqin2yrs"])])))


# In[ ]:


df["DebtRatio"].quantile(0.975)


# In[ ]:


df.loc[df["DebtRatio"] > df["DebtRatio"].quantile(0.975)][["DebtRatio", "MonthlyIncome", "SeriousDlqin2yrs"]].describe()


# In[ ]:


len(df[(df["DebtRatio"] > df["DebtRatio"].quantile(0.975)) & (df['SeriousDlqin2yrs'] == df['MonthlyIncome'])])


# In[ ]:


df.loc[df["DebtRatio"] > df["DebtRatio"].quantile(0.975)]["MonthlyIncome"].value_counts()


# In[ ]:


print("Number of people who owe around 3490 or more times what they own and have same values for MonthlyIncome and SeriousDlqin2yrs: {}"     .format(len(df.loc[(df["DebtRatio"] > df["DebtRatio"].quantile(0.975)) & (df["MonthlyIncome"] == df["SeriousDlqin2yrs"])])))


# ### MonthlyIncome

# In[ ]:


sns.distplot(df["MonthlyIncome"].dropna())


# In[ ]:


df["MonthlyIncome"].describe()


# In[ ]:


sns.distplot(df.loc[df["DebtRatio"] <= df["DebtRatio"].quantile(0.975)]["MonthlyIncome"].dropna())


# ### NumberOfOpenCreditLinesAndLoans

# In[ ]:


df["NumberOfOpenCreditLinesAndLoans"].describe()


# In[ ]:


df["NumberOfOpenCreditLinesAndLoans"].value_counts()


# In[ ]:


sns.distplot(df["NumberOfOpenCreditLinesAndLoans"])


# ### NumberRealEstateLoansOrLines

# In[ ]:


df["NumberRealEstateLoansOrLines"].describe()


# In[ ]:


df["NumberRealEstateLoansOrLines"].value_counts()


# In[ ]:


sns.countplot(x="NumberRealEstateLoansOrLines", data=df.loc[df["NumberRealEstateLoansOrLines"] <= 10])


# In[ ]:


df.loc[df["NumberRealEstateLoansOrLines"] > 13]["SeriousDlqin2yrs"].describe()


# ### NumberOfDependents

# In[ ]:


df["NumberOfDependents"].describe()


# In[ ]:


df["NumberOfDependents"].value_counts()


# In[ ]:


df.loc[df["NumberOfDependents"] <= 10]["SeriousDlqin2yrs"].describe()


# In[ ]:


sns.countplot(x="NumberOfDependents", data=df.loc[df["NumberOfDependents"] <= 10])


# ## Preprocessing Suggestions <a class="anchor" id="4"></a>
# 
# - Remove samples with values of DebtRatio above its 97.5 percentile
# - Set 0 <= RevolvingUtilizationOfUnsecuredLines < 13
# - Set NumberOfTimes90DaysLate <= 17
# - Impute MonthlyIncome with its median, or with a normally distributed variable with MonthlyIncome's mean and standard deviation
# - Impute NumberOfDependents with its mode

# ## References <a class="anchor" id="5"></a>
# 
# https://github.com/nicholaslaw/kaggle-credit-scoring
