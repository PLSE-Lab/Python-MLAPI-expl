#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


diet = pd.read_csv('../input/dietstudy.csv')
diet.head()


# # CI : 0.95 or 95%
# # alpha = 0.05
# # if p value < alpha then reject NULL and 
# # if it is >= alpha, we fail to reject NULL

# In[ ]:


print('pre diet weight: ', diet.wgt0.mean())
print('pre diet triglyceride level: ', diet.tg0.mean())


# In[ ]:


print('post 1 month diet weight: ', diet.wgt1.mean())
print('post 1 month diet triglyceride level: ', diet.tg1.mean())


# ### First month
# ### Ho: weight before diet == weight after 1 month (no change)
# ### Ha: weight before diet != weight after 1 month (change)
# 
# ### Ho: triglyceride before diet == triglyceride after 1 month 
# ### Ha: triglyceride before diet != triglyceride after 1 month

# In[ ]:


month1_wt = stats.ttest_rel(a=diet.wgt0, b=diet.wgt1)
month1_tg = stats.ttest_rel(a=diet.tg0, b=diet.tg1)


# In[ ]:


print(month1_wt.pvalue < 0.05)
print(month1_tg.pvalue < 0.05)


# ### Conclusion: Afer 1 month of dieting, there is a change in weight but there is no significant change in triglyceride level.

# ### Second month
# ### Ho: weight before diet == weight after 2 month
# ### Ha: weight before diet != weight after 2 month
# 
# ### Ho: triglyceride before diet == triglyceride after 2 month
# ### Ha: triglyceride before diet != triglyceride after 2 month

# In[ ]:


print('post 2 month diet weight: ', diet.wgt2.mean())
print('post 2 month diet triglyceride level: ', diet.tg2.mean())


# In[ ]:


month2_wt = stats.ttest_rel(a=diet.wgt0, b=diet.wgt2)
month2_tg = stats.ttest_rel(a=diet.tg0, b=diet.tg2)


# In[ ]:


print(month2_wt.pvalue < 0.05)
print(month2_tg.pvalue < 0.05)


# ### Conclusion: Afer 2 months of dieting, there is a change in weight but there is no significant change in triglyceride level.

# # -----------------------------------------------------------------------

# ### Third month
# ### Ho: weight before diet == weight after 3 months
# ### Ha: weight before diet != weight after 3 months
# 
# ### Ho: triglyceride before diet == triglyceride after 3 months
# ### Ha: triglyceride before diet != triglyceride after 3 months

# In[ ]:


print('post 3 months diet weight: ', diet.wgt3.mean())
print('post 3 months diet triglyceride level: ', diet.tg3.mean())


# In[ ]:


month3_wt = stats.ttest_rel(a=diet.wgt0, b=diet.wgt3)
month3_tg = stats.ttest_rel(a=diet.tg0, b=diet.tg3)


# In[ ]:


print(month3_wt.pvalue < 0.05)
print(month3_tg.pvalue < 0.05)


# ### Conclusion: Afer 3 months of dieting, there is a change in weight but there is no significant change in triglyceride level.

# # ---------------------------------------------------------------------

# ### Fourth month
# ### Ho: weight before diet == weight after 4 months
# ### Ha: weight before diet != weight after 4 months
# 
# ### Ho: triglyceride before diet == triglyceride after 4 months
# ### Ha: triglyceride before diet != triglyceride after 4 months

# In[ ]:


print('post 4 months diet weight: ', diet.wgt4.mean())
print('post 4 months diet triglyceride level: ', diet.tg4.mean())


# In[ ]:


month4_wt = stats.ttest_rel(a=diet.wgt0, b=diet.wgt4)
month4_tg = stats.ttest_rel(a=diet.tg0, b=diet.tg4)


# In[ ]:


print(month4_wt.pvalue < 0.05)
print(month4_tg.pvalue < 0.05)


# ### Conclusion: Afer 4 months of dieting, there is a change in weight but there is no significant change in triglyceride level.

# # -----------------------------------------------------------------------

# > **Credit Promo**
# 
# 1. An analyst at a department store wants to evaluate a recent credit card promotion. To this end, 500 cardholders were randomly selected. Half received an ad promoting a reduced interest rate on purchases made over the next three months, and half received a standard seasonal ad. Is the promotion effective to increase sales?

# In[ ]:


credit = pd.read_csv('../input/creditpromo.csv')
credit.head()


# In[ ]:


standard = credit.dollars[credit['insert']=='Standard']
new_promo = credit.dollars[credit['insert']=='New Promotion']


# In[ ]:


print('Spend of customer with standard promo: ', standard.mean())
print('Spend of customer with new promo: ', new_promo.mean())


# ### Ho: standard promo spend == new promo spend (promo failure)
# ### Ha: standard promo spend != new promo spend (promo success)
# 

# In[ ]:



credit_equalv = stats.ttest_ind(a= standard, b=new_promo, equal_var=True)
credit_unequalv = stats.ttest_ind(a= standard, b=new_promo, equal_var=False)


# In[ ]:


credit_equalv.statistic - credit_unequalv.statistic


# In[ ]:


credit_equalv.pvalue < 0.05


# ### Conclusion: Customers who received new promotion spend more than the customers with standard promo.
# ### Hence, promotion is effective.

# # -----------------------------------------------------------------------

# # > **Pollination**
# 1. An experiment is conducted to study the hybrid seed production of bottle gourd under open field conditions. The main aim of the investigation is to compare natural pollination and hand pollination. The data are collected on 10 randomly selected plants from each of natural pollination and hand pollination. The data are collected on fruit weight (kg), seed yield/plant (g) and seedling length (cm). (Data set: pollination.csv) a. Is the overall population of Seed yield/plant (g) equals to 200? b. Test whether the natural pollination and hand pollination under open field conditions are equally effective or are significantly different.

# In[ ]:


pol = pd.read_csv('../input/pollination.csv')
pol.head()


# In[ ]:


pol.Seed_Yield_Plant.mean()


# ### Ho: Seed yield/plant == 200
# ### Ha: Seed yield/plant <  200

# In[ ]:


seed_yield = stats.ttest_1samp(a=pol.Seed_Yield_Plant, popmean=200)


# In[ ]:


seed_yield.pvalue < 0.05


# ### Conclusion: Seed yield/plant is not equal to 200, it is lower.

# In[ ]:


natural_pol = pol.loc[pol.Group == 'Natural']

hand_pol = pol.loc[pol.Group == 'Hand']


# In[ ]:


natural_pol.Seedling_length.mean()


# In[ ]:


hand_pol.Seedling_length.mean()


# #Ho: Seedling_length from natural pollination == Seedling_length from hand pollination (equally effective)
# #Ha: Seedling_length from natural pollination != Seedling_length from hand pollination (significantly different)

# In[ ]:


seed_length = stats.ttest_ind(a=natural_pol.Seedling_length, b=hand_pol.Seedling_length)
seed_length


# In[ ]:


seed_length.pvalue


# In[ ]:


print(natural_pol.Fruit_Wt.mean())
print(hand_pol.Fruit_Wt.mean())

print(natural_pol.Seed_Yield_Plant.mean())
print(hand_pol.Seed_Yield_Plant.mean())


# In[ ]:


fruit_wt = stats.ttest_ind(a=natural_pol.Fruit_Wt, b=hand_pol.Fruit_Wt)
seed_yield = stats.ttest_ind(a=natural_pol.Seed_Yield_Plant, b=hand_pol.Seed_Yield_Plant)


# In[ ]:


fruit_wt.pvalue


# In[ ]:


seed_yield.pvalue


# ### Conclusion: As p-value is less than 0.05, we reject null hypothesis, there is a significant differenc in natural pollination and hand pollination under open field conditions.

# # DVD player
# 1. An electronics firm is developing a new DVD player in response to customer requests. Using a prototype, the marketing team has collected focus data for different age groups viz. Under 25; 25-34; 35-44; 45-54; 55-64; 65 and above. Do you think that consumers of various ages rated the design differently?

# In[ ]:


dvd = pd.read_csv('../input/dvdplayer.csv')
dvd.head()


# In[ ]:


dvd.agegroup.value_counts()


# In[ ]:


grp1 = dvd.dvdscore.loc[dvd.agegroup == 'Under 25']
grp2 = dvd.dvdscore.loc[dvd.agegroup == '25-34']
grp3 = dvd.dvdscore.loc[dvd.agegroup == '35-44']
grp4 = dvd.dvdscore.loc[dvd.agegroup == '45-54']
grp5 = dvd.dvdscore.loc[dvd.agegroup == '55-64']
grp6 = dvd.dvdscore.loc[dvd.agegroup == '65 and over']


# ### Ho: mean grp1 == mean grp2 == mean grp3 == mean grp4 == mean grp5 == mean grp6 (rated equally similar)
# ### Ha: mean grp1 != mean grp2 != mean grp3 != mean grp4 != mean grp5 != mean grp6 (rated differently)

# In[ ]:


dvd_anova = stats.f_oneway(grp1, grp2, grp3, grp4, grp5, grp6)
dvd_anova


# ### Conclusion: Consumers of different age groups rated the design differently.

# ## Sample Survey
# A survey was conducted among 2800 customers on several demographic characteristics. Working status, sex, age, age-group, race, happiness, no. of child, marital status, educational qualifications, income group etc. had been captured for that purpose. (Data set: sample_survey.csv). a. Is there any relationship in between labour force status with marital status? b. Do you think educational qualification is somehow controlling the marital status? c. Is happiness is driven by earnings or marital status?

# In[ ]:


sample_data = pd.read_csv('../input/sample_survey.csv')
sample_data.head(2)


# In[ ]:


sample_data.info()


# ### Ho: Observed == Expected (no influence)
# ### Ha: Observed != Expected (influence)

# In[ ]:


wrk_mar_xtab = pd.crosstab(sample_data.wrkstat, sample_data.marital, margins=True)
wrk_mar_xtab


# In[ ]:


wrk_mar_test = stats.chi2_contingency(observed=wrk_mar_xtab)
wrk_mar_test


# ### Conclusion:  labour force status have relationship with marital status

# In[ ]:


degree_mar_xtab = pd.crosstab(sample_data.degree, sample_data.marital, margins=True)
degree_mar_xtab


# In[ ]:


degree_mar_test = stats.chi2_contingency(observed=degree_mar_xtab)
degree_mar_test[1]


# ### Conclusion: Educational qualification has influence on the marital status.

# In[ ]:


happy_mar_xtab = pd.crosstab(sample_data.happy, sample_data.marital, margins=True)
happy_mar_xtab


# In[ ]:


happy_mar_test = stats.chi2_contingency(observed=happy_mar_xtab)
happy_mar_test


# In[ ]:


happy_income_xtab = pd.crosstab(sample_data.happy, sample_data.income, margins=True)
happy_income_xtab


# In[ ]:


happy_income_test = stats.chi2_contingency(observed=happy_income_xtab)
happy_income_test


# ### Conclusion: As p-value for both cases (happiness with marital status vs. happiness with earnings) is less, 
# ### both has influence on happiness, but as chi-square test score is more with marital status (260) as compared with earnings (178),
# ### happiness is driven more by marital status.
