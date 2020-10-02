#!/usr/bin/env python
# coding: utf-8

# ***Please UpVote if you like the work!!!***

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/treatment-of-migraine-headaches/KosteckiDillon.csv',index_col=0)


# In[ ]:


df.head()


# ### Check whether 'age' is a factor affecting 'headache'

# In[ ]:


df['headache'].value_counts()


# In[ ]:


df_headache_yes = df[df['headache'] == 'yes']['age']
df_headache_no = df[df['headache'] == 'no']['age']


# ##### Check for normality (Shapiro Test)

# In[ ]:


from scipy.stats import shapiro,bartlett,mannwhitneyu
print(shapiro(df_headache_yes))
print(shapiro(df_headache_no))


# In[ ]:


df_headache_no.plot(kind = 'density')


# In[ ]:


df_headache_yes.plot(kind = 'density')


# As p-value is very less, we reject the null hypothesis that the samples are normally distributed. So now we will check for Bartllet test for checking the variance.
# ##### Bartlett test (Test for equal variances)

# In[ ]:


bartlett(df_headache_no,df_headache_yes)


# Even the bartlett test fails as pvalue is less than 0.05. So the variances of both the samples are also not equal. So we need to go for non-parametric test.
# ##### Manwhitneyu
# H0 : avg_age_headache = avg_age_no_headache
# 
# H1 : avg_age_headache != avg_age_no_headache

# In[ ]:


mannwhitneyu(df_headache_no,df_headache_yes)


# As we can see, the probability of H0 being true is very very negligible, so we reject H0 and can say that age is a factor which affects the headache.

# ### Check whether 'gender' is a factor affecting 'headache'

# In[ ]:


df['sex'].value_counts()


# In[ ]:


ct = pd.crosstab(df['headache'],df['sex'])
print(ct)


# ##### H0 : prop_female_headache = prop_male_headache
# ##### H1 : prop_female_headache != prop_male_headache

# In[ ]:


prop_female_headache = 2279/3545
prop_male_headache = 387/607
print(prop_female_headache,prop_male_headache)


# In[ ]:


from statsmodels.stats.proportion import proportions_ztest
x = np.array([2279,387])
n=  np.array([3545,607])
proportions_ztest(x,n)


# Fail to reject the null hypothesis. So gender is not the factor affecting the headaches.

# ### Check whether 'gender' is a factor affecting 'hatype'(headache type)

# In[ ]:


ct = pd.crosstab(df['hatype'],df['sex'])
print(ct)


# In[ ]:


from scipy.stats import chi2_contingency
chi2_contingency(ct)


# Reject the null hypothesis. So gender is a factor affecting the types of headaches.

# ##### Post hoc analysis

# In[ ]:


print('Aura : Female - ',1593/3545,", Male - ",117/607)
print('Mixed : Female - ',291/3545,", Male - ",166/607)
print('No Aura : Female - ',1661/3545,", Male - ",324/607)


# We can see, Female are highly sensitive to Aura type of headaches, Male are highly sensitive to Miixed type of headache and for No Aura, both Female and Male are quite equally sensitive.

# ### Adjusted Rsquare

# adj_rsquare = (1-rsquare)*(N-1)/(N-p-1)
# 
# N -> total number of observations in sample
# 
# p -> number of features

# In[ ]:


df = pd.read_csv('../input/ibm-data/ibm.csv')


# In[ ]:


df.drop('Over18',axis = 1,inplace = True)


# In[ ]:


df.head()


# In[ ]:


df['Attrition'].value_counts()


# In[ ]:


df['Gender'].value_counts()


# 1. Check whether attrition rate is based on Gender.

# In[ ]:


pd.crosstab(df['Attrition'],df['Gender'])


# H0 : prop_attrition_female = prop_attrition_male
# 
# H1 : prop_attrition_female != prop_attrition_male

# In[ ]:


prop_attrition_female = 87/588
prop_attrition_male = 150/882


# In[ ]:


from statsmodels.stats.proportion import proportions_ztest
count = np.array([87,150])
nobs = np.array([588,882])
proportions_ztest(count,nobs)


# As the p-value is greater than 0.05, we fail to reject the null hypothesis. So we can say that the attrition rate of the organisation is not based on Gender.

# 2. Check whether attrition rate is based on Department.

# H0 : prop_attrition_HR = prop_attrition_R&D = prop_attrition_Sales
# 
# H1 : prop_attrition_HR != prop_attrition_R&D != prop_attrition_Sales

# In[ ]:


df['Department'].value_counts()


# In[ ]:


ct = pd.crosstab(df['Attrition'],df['Department'])
ct


# In[ ]:


from scipy.stats import chi2_contingency
chi2_contingency(ct)


# We can see from the chi2 test that the attrition rate is affected by deparment as the pvalue is less than 0.05.
# ##### Post hoc analysis

# In[ ]:


prop_attrition_yes_HR = 12/63
prop_attrition_yes_RnD = 133/961
prop_attrition_yes_Sales = 92/446
prop_attrition_no_HR = 51/63
prop_attrition_no_RnD = 828/961
prop_attrition_no_Sales = 354/446
print('prop_attrition_yes_HR : ',prop_attrition_yes_HR)
print('prop_attrition_yes_RnD : ',prop_attrition_yes_RnD)
print('prop_attrition_yes_Sales : ',prop_attrition_yes_Sales)
print('prop_attrition_no_HR : ',prop_attrition_no_HR)
print('prop_attrition_no_RnD : ',prop_attrition_no_RnD)
print('prop_attrition_no_Sales : ',prop_attrition_no_Sales)


# Attrition rate of Research and Development Department is quiet less as compared to HR and Sales deparment.

# 3. Is there any discrepancy in monthly income avg with respect to Gender.

# H0 : avg_monthlyincome_female = avg_monthlyincome_male
# 
# H1 : avg_monthlyincome_female != avg_monthlyincome_male

# In[ ]:


df_male_income = df[df['Gender'] == 'Male']['MonthlyIncome']
df_female_income = df[df['Gender'] == 'Female']['MonthlyIncome']


# ##### Shapiro test

# In[ ]:


from scipy.stats import shapiro
print(shapiro(df_male_income))
print(shapiro(df_female_income))


# ##### Bartlett test

# In[ ]:


from scipy.stats import bartlett
print(bartlett(df_male_income,df_female_income))


# ##### Manwhitneyu

# In[ ]:


from scipy.stats import mannwhitneyu
print(mannwhitneyu(df_male_income,df_female_income))


# As the p-value is less than 0.05, we reject the null hypothesis. So we can say that the avg monthly income of the empoyees in the organisation is Gender based.

# ##### Post hoc analysis

# In[ ]:


print(df_male_income.mean(),df_female_income.mean())


# As we can see quite evidently that females an average monthly income of around 6700 whereas males have an average monthly income of around 6400. So there seems to be quiet a significant difference in their salaries.

# 4. Is there any discrepancy in monthly income avg with respect to Department.

# H0 : avg_monthlyincome_HR = avg_monthlyincome_R&D = avg_monthlyincome_Sales
# 
# H1 : avg_monthlyincome_HR != avg_monthlyincome_R&D != avg_monthlyincome_Sales

# In[ ]:


df_HR_income = df[df['Department'] == 'Human Resources']['MonthlyIncome']
df_RnD_income = df[df['Department'] == 'Research & Development']['MonthlyIncome']
df_Sales_income = df[df['Department'] == 'Sales']['MonthlyIncome']


# In[ ]:


from scipy.stats import f_oneway
f_oneway(df_HR_income,df_RnD_income,df_Sales_income)


# We can see from the oneway test that the monthly income of employees is affected by deparment as the pvalue is less than 0.05.

# ##### Post hoc analysis

# In[ ]:


print('Avg monthly income of HR Department : ',df_HR_income.mean())
print('Avg monthly income of R&D Department : ',df_RnD_income.mean())
print('Avg monthly income of Sales Department : ',df_Sales_income.mean())


# As we can see quite evidently that the employees from Sales department has an average monthly income of around 7000 whereas employees from Reasearch and Development have an average monthly income of around 6300. So there seems to be quiet a significant difference in their salaries.

# 5. Is there any discrepancy in monthly income avg with respect to Education.

# In[ ]:


df['Education'].value_counts()


# H0 : avg_mnthInc_EduLvl1 = avg_mnthInc_EduLvl2 = avg_mnthInc_EduLvl3 = avg_mnthInc_EduLvl4 = avg_mnthInc_EduLvl5
# 
# H1 : avg_mnthInc_EduLvl1 != avg_mnthInc_EduLvl2 != avg_mnthInc_EduLvl3 != avg_mnthInc_EduLvl4 != avg_mnthInc_EduLvl5

# In[ ]:


df_mnthInc_EduLvl1 = df[df['Education'] == 1]['MonthlyIncome']
df_mnthInc_EduLvl2 = df[df['Education'] == 2]['MonthlyIncome']
df_mnthInc_EduLvl3 = df[df['Education'] == 3]['MonthlyIncome']
df_mnthInc_EduLvl4 = df[df['Education'] == 4]['MonthlyIncome']
df_mnthInc_EduLvl5 = df[df['Education'] == 5]['MonthlyIncome']


# In[ ]:


f_oneway(df_mnthInc_EduLvl1,df_mnthInc_EduLvl2,df_mnthInc_EduLvl3,df_mnthInc_EduLvl4,df_mnthInc_EduLvl5)


# We can see from the oneway test that the monthly income of employees is affected by education level of the employee as the pvalue is less than 0.05.

# ##### Post hoc analysis

# In[ ]:


print('Avg monthly income for employees with Education Level 1 : ',df_mnthInc_EduLvl1.mean())
print('Avg monthly income for employees with Education Level 2 : ',df_mnthInc_EduLvl2.mean())
print('Avg monthly income for employees with Education Level 3 : ',df_mnthInc_EduLvl3.mean())
print('Avg monthly income for employees with Education Level 4 : ',df_mnthInc_EduLvl4.mean())
print('Avg monthly income for employees with Education Level 5 : ',df_mnthInc_EduLvl5.mean())


# As we can see quite evidently that the employees with Higher Education level has higher average monthly income.

# In[ ]:


df.boxplot(column='MonthlyIncome',by = 'Education')


# In[ ]:


df.boxplot(column='MonthlyIncome',by = 'Department')


# In[ ]:


df.boxplot(column='MonthlyIncome',by = 'Gender')


# ***Please UpVote if you like the work!!!***
