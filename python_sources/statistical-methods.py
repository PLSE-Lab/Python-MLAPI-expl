#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("/kaggle/input/statistical-learning-dataset/cs1.csv")
df.head()


# ## Income Vs gender:Two sample t test

# In[ ]:


male=df[df["gender"]=="MALE"]
female=df[df["gender"]=="FEMALE"]


# In[ ]:


male.income.mean()


# In[ ]:


female.income.mean()


# In[ ]:


## 1) the sample is drawn highly randomised
## 2) normality of income


# ## checking the normality of "income" distribution

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# ***Stating of Hypothesis:
# H0:mu_male=mu_female
# H1:mu_male!=mu_female***

# In[ ]:


sns.distplot(df["income"])


# In[ ]:


import scipy.stats as st


# In[ ]:


df.income.describe()


# In[ ]:


st.shapiro(df.income)


# **Since p value is less than alpha (5%) so the distribution does not follow normality.
# Hence we do non parametric test:mannwhitney u**

# In[ ]:


st.mannwhitneyu(male.income,female.income)


# **Assume the income follows normal
# Check for variance of male income and female income**

# In[ ]:


st.levene(male.income,female.income)


# In[ ]:


# since p value is greater than alpha(5%) accept H0; means variance of male income and female income are same.


# In[ ]:


st.ttest_ind(male.income,female.income)


# Conclusion: Both male and female income are same.

# ## Income vs Married (Two sample t-test)

# In[ ]:


Yes=df[df["married"]=="YES"]
No=df[df["married"]=="NO"]


# In[ ]:


st.mannwhitneyu(Yes.income,No.income)


# In[ ]:


st.levene(Yes.income,No.income)


# In[ ]:


st.ttest_ind(Yes.income,No.income)


# ## Income Vs Pl

# In[ ]:


ipl=df[df["pl"]=="YES"]

inpl=df[df["pl"]=="NO"]


# In[ ]:


st.mannwhitneyu(ipl.income,inpl.income)


# In[ ]:


st.levene(ipl.income,inpl.income)


# In[ ]:


st.ttest_ind(ipl.income,inpl.income)


# **The income of people who have taken pl is not same as income of people who have not taken pl.**

# ## Income vs region

# Stating of Hypothesis:
# H0:mu_R1=mu_R2=mu_R3=mu_R4
# H1:mu_R1!=mu_R2!=mu_R3!=mu_R4

# In[ ]:


## Assumptions : 1.Randomness 2.Normality 3.Variance equality
## 1. assumed randomness
## 2. Normality is already checked,assume for a while it follows noemal
## 3. Variance equality (need to check)


# In[ ]:


df.region.unique()


# In[ ]:


iic=df[df["region"]=="INNER_CITY"]["income"]
it=df[df["region"]=="TOWN"]["income"]
ir=df[df["region"]=="RURAL"]["income"]
isu=df[df["region"]=="SUBURBAN"]["income"]


# In[ ]:


st.levene(iic,it,ir,isu)


# In[ ]:


sns.boxplot(x="region",y="income",data=df)


# In[ ]:


st.f_oneway(iic,it,ir,isu)


# In[ ]:


## since income is not satisfied the normality, we should do kruskal wallis test


# In[ ]:


st.kruskal(iic,it,ir,isu)


# ## Income Vs Children

# Stating of Hypothesis:
# H0:mu_c0=mu_c1=mu_c2=mu_c3
# H1:mu_c0!=mu_c1!=mu_c2!=mu_c3

# In[ ]:


c0=df[df["children"]==0]["income"]
c1=df[df["children"]==1]["income"]
c2=df[df["children"]==2]["income"]
c3=df[df["children"]==3]["income"]


# In[ ]:


st.levene(c0,c1,c2,c3)


# In[ ]:


st.f_oneway(c0,c1,c2,c3)


# **The income of people who have children is same as income of people who have no children.**

# ## Categorical vs Categorical

# ### Pl Vs gender
# H0: There is no association between Pl status and gender
# H1: There is an assosiation between Pl status and gender

# In[ ]:


## Z proportion test
from statsmodels.stats.proportion import proportions_ztest


# In[ ]:


tab=pd.crosstab(df["pl"],df["gender"])
tab


# In[ ]:


proportions_ztest([86,62],[170,160])


# In[ ]:


tab.loc["YES"]


# In[ ]:


proportions_ztest(tab.loc["YES"],tab.sum(axis=0))


# In[ ]:


st.chi2_contingency(tab)


# ## Pl vs Marital status

# In[ ]:


tab1=pd.crosstab(df["pl"],df["married"])
tab1


# In[ ]:


proportions_ztest([63,85],[116,214])


# In[ ]:


st.chi2_contingency(tab1)


# ## Pl,income and Car

# In[ ]:


from statsmodels.formula.api import ols


# In[ ]:


df = pd.get_dummies(df,columns=["pl","car"],drop_first=True)


# In[ ]:


lin_model=ols("income~pl_YES+car_YES",data=df).fit()
lin_model.summary()


# ## Two-Way Anova

# In[ ]:


from statsmodels.stats.anova import anova_lm


# In[ ]:


formula = 'income ~ pl_YES + car_YES'
model = ols(formula,df).fit()
aov_table = anova_lm(model, typ = 2)
aov_table


# In[ ]:


model1 = ols('income ~ pl_YES + car_YES',data = df).fit()
model1.summary()


# In[ ]:


from scipy.stats import f


# In[ ]:


f.sf(17.03,2,327)


# In[ ]:




