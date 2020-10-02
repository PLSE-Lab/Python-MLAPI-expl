#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


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


df = pd.read_csv('/kaggle/input/statistical-learning-dataset/cs1.csv')
df.head()


# # Income VS Gender : two sample t test

# Ho : mu_male = mu_female
# 
# 
# H1 : mu_male != mu_female

# In[ ]:


male = df[df['gender'] == 'MALE']
female = df[df['gender'] == 'FEMALE']


# In[ ]:


male.income.mean()


# In[ ]:


female.income.mean()


# ### Assumptions:
# 
# 1 : Sample is drawn highly randomized
# 
# 2 : Normality of income

# ## Cheking the normality of 'income' distribution

# In[ ]:


sns.distplot(df['income'])


# In[ ]:


import scipy.stats as st

st.shapiro(df.income)


# #### Since p-value < alpha(assuming as 5%) so the disribution does not follow normality. Hence we do non-parametric test (mannwhitney)

# In[ ]:


st.mannwhitneyu(male.income,female.income)


# ### Assume the income follows normal 
# 3. check for variance of male and female income

# In[ ]:


st.levene(male.income,female.income)


# #### Since p-value > alpha, we accept H0 i.e variance of female and male are same.

# In[ ]:


st.ttest_ind(male.income,female.income)


# ### Conclusion : Male & Female income are the same.

# # Income VS Married

# In[ ]:


married = df[df['married'] == 'YES']
unmarried = df[df['married'] == 'NO']


# In[ ]:


st.mannwhitneyu(married.income,unmarried.income)


# In[ ]:


st.ttest_ind(married.income,unmarried.income)


# ### Accept H0: i.e married and unmarried ppl have same income

# # Income VS PL

# In[ ]:


ply = df[df['pl'] == 'YES']
pln = df[df['pl'] == 'NO']


# In[ ]:


st.mannwhitneyu(ply.income,pln.income)


# In[ ]:


st.ttest_ind(ply.income,pln.income)


# ### Reject H0 as p-value < alpha it means that there is difference in income of people with PL and without PL.

# # Income VS Region

# H0 : mu_R1 = mu_R2 = mu_R3 = mu_R4
# 
# H1 : H0 : mu_R1 != mu_R2 != mu_R3 != mu_R4

# ### Assumptions : 
#     1. Randomness
#     2. Normality
#     3. Variance equality

# #### Randomness : we assume its random.
# #### Normality  : already checked
# 

# #### Variance equality
# 

# In[ ]:


df.region.unique()


# In[ ]:


ic = df[df['region'] == 'INNER_CITY']
t = df[df['region'] == 'TOWN']
r = df[df['region'] == 'RURAL']
s = df[df['region'] == 'SUBURBAN']


# In[ ]:


st.levene(ic.income,t.income,r.income,s.income)


# In[ ]:


sns.boxplot(df['region'],df['income'])


# #### Parametric

# In[ ]:


st.f_oneway(ic.income,t.income,r.income,s.income)


# #### Since income does not satisfy normality, we should do kruskal wallis test.

# #### Non-parametric

# In[ ]:


st.kruskal(ic.income,t.income,r.income,s.income)


# ### Accept H0: 

# # Income VS Children

# In[ ]:


df.children.unique()


# In[ ]:


zero = df[df['children'] == 0]
one = df[df['children'] == 1]
two = df[df['children'] == 2]
three = df[df['children'] == 3]


# In[ ]:


st.f_oneway(zero.income,one.income,two.income,three.income)


# In[ ]:


st.kruskal(zero.income,one.income,two.income,three.income)


# # Age VS Gender

# In[ ]:


sns.distplot(df.age)


# In[ ]:


st.shapiro(df['age'])


# #### It is not normal as p-value < alpha.

# In[ ]:


st.levene(male.age,female.age)


# In[ ]:


st.mannwhitneyu(male.age,female.age)


# In[ ]:


st.ttest_ind(male.age,female.age)


# # Age VS Married

# In[ ]:


st.levene(married.age,unmarried.age)


# In[ ]:


st.mannwhitneyu(married.age,unmarried.age)


# In[ ]:


st.ttest_ind(married.age,unmarried.age)


# # Age VS PL

# In[ ]:


st.levene(ply.age,pln.age)


# In[ ]:


st.mannwhitneyu(ply.age,pln.age)


# In[ ]:


st.ttest_ind(ply.age,pln.age)


# # Age VS Region

# In[ ]:


st.levene(ic.age,t.age,r.age,s.age)


# In[ ]:


st.f_oneway(ic.age,t.age,r.age,s.age)


# In[ ]:


st.kruskal(ic.age,t.age,r.age,s.age)


# # Age VS Children

# In[ ]:


st.levene(zero.age,one.age,two.age,three.age)


# In[ ]:


st.f_oneway(zero.age,one.age,two.age,three.age)


# In[ ]:


st.kruskal(zero.age,one.age,two.age,three.age)


# # CATEGORICAL Vs CATEGORICAL

# # PL VS Gender

# H0 : there is no association b/w PL status and gender 
# 
# H1 : there is an association b/w PL status and gender
# 

# #### Z proportion test

# Checking the proportion of female and male in those who have taken PL

# In[ ]:


from statsmodels.stats.proportion import proportions_ztest

ct = pd.crosstab(df['pl'],df['gender'])


# In[ ]:


ct


# In[ ]:


proportions_ztest([86,62],[170,160])


# In[ ]:


st.chi2_contingency(ct)


# # PL VS Married

# In[ ]:


cm = pd.crosstab(df['pl'],df['married'])


# In[ ]:


st.chi2_contingency(cm)


# # PL VS Children

# In[ ]:


c = pd.crosstab(df['pl'],df['children'])
st.chi2_contingency(c)


# # PL VS Region

# In[ ]:


r = pd.crosstab(df['pl'],df['region'])
st.chi2_contingency(r)


# # PL VS Car

# In[ ]:


st.chi2_contingency(pd.crosstab(df['pl'],df['car']))


# # PL VS SaveAct

# In[ ]:


st.chi2_contingency(pd.crosstab(df['pl'],df['save_act']))


# # PL VS CurrentAct

# In[ ]:


st.chi2_contingency(pd.crosstab(df['pl'],df['current_act']))


# ### Conclusion : Gender and Married have association with PL whereas rest of it doesn't have

# # Two way annova

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# In[ ]:


res = ols('income ~ pl +car',data=df).fit()
res.summary()


# In[ ]:


anova_lm(res,typ=1)


# In[ ]:


res1 = ols('income ~ pl +car+pl:car',data=df).fit()
res1.summary()

