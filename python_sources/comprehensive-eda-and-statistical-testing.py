#!/usr/bin/env python
# coding: utf-8

# # Read in Essential Libraries

# In[ ]:


import pandas as pd
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read in Data

# In[ ]:


df = pd.read_excel("../input/boroughpop.xlsx")
df2 = pd.read_excel("../input/londonimddecile.xlsx")
df_map = pd.read_excel("../input/mmap.xlsx")


# In[ ]:


df.head()


# In[ ]:


df2.head()


# In[ ]:


df_map.head()


# # Merging Data

# In[ ]:


# Change name of column 'ladname' to 'ladnm' for merging data
df2 = df2.rename(columns={'ladname':'ladnm'})


# In[ ]:


# First merge (df and df2)
data = pd.merge(df,df2, on='ladnm',how='outer')


# In[ ]:


# Second merge (merged df,df2 and map data)
data = pd.merge(data,df_map, on=['ladnm','lsoacode'],how='outer')


# In[ ]:


# Overall information about merged data
data.info()


# In[ ]:


# Look at first 5 rows of the merged data
data.head()


# # Missing Data

# The missingno package allows us to visualize how many missing values we have for each variable!

# In[ ]:


msno.bar(data)


# # Univariate EDA

# ###### Population of each Borough

# In[ ]:


# Number of unique boroughs in London
data.ladnm.nunique()


# In[ ]:


df.sort_values('population', ascending=False).set_index('ladnm').plot.barh(figsize=(10,8))


# ###### Index of Deprivation(imd decile)

# https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/579151/English_Indices_of_Deprivation_2015_-_Frequently_Asked_Questions_Dec_2016.pdf
# 
# The Index of Multiple Deprivation, commonly known as the IMD, is the official measure of relative deprivation for small areas in England. The Index of Multiple Deprivation ranks every small area in England from 1 (mostdeprived area) to 32,844 (least deprived area). The Index of Multiple Deprivation (IMD) combines information from the seven domains to produce an overall relative measure of deprivation. Those domains include income deprivation, employment deprivation, barriers to housing etc.

# In[ ]:


# Distribution of Index of Multiple Deprivation
sns.distplot(data[data.imddecile.notnull()].imddecile)


# But the variable "imddecile" is given in deciles and there is no additional information about whether the higher deciles (closer to 12) represent more deprived areas or not, so unsure how to interpret this variable for now!

# ###### Where Gang is Present in a certain Borough (Binary variable: Yes or No) 

# In[ ]:


# Proportion of boroughs where gangs are present (In Orange)
(data.groupby('ladnm').first().gangpresent.value_counts()*100 / data.groupby('ladnm').first().gangpresent.value_counts().sum()).plot('bar')


# London has slightly more (about 57%) boroughs that don't have gangs. But is this proportion statistically significant? We can use the Chi-Squared test of goodness-of-fit to check this.

# In[ ]:


# Chi-Squared test of goodness-of-fit 
from scipy import stats
stats.chisquare(f_obs= data.groupby('ladnm').first().gangpresent.value_counts()*100 / data.groupby('ladnm').first().gangpresent.                value_counts().sum(),
                f_exp= [0.5,0.5])


# The p-value is smaller than 0.05 which is the typical significance level for statistical significance and thus **London has statistically higher proportion of boroughs that don't have gangs.**

# ###### Age distribution of victims and suspects

# In[ ]:


f, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax1 = sns.distplot(data[data.vicage.notnull()].vicage,ax=axes[0],color='darkred')
ax2 = sns.distplot(data[data.susage.notnull()].susage,ax=axes[1],color='darkblue')


# ###### Sex proportions of victims and suspects

# In[ ]:


# Victim sex proportion
labels='Male','Female'
plt.pie(data['vicsex'].value_counts(), labels=labels,autopct='%1.1f%%', startangle = 150, shadow=True)


# In[ ]:


# Suspect sex proportion
labels='Male','Female'
plt.pie(data['sussex'].value_counts(), labels=labels,autopct='%1.1f%%', startangle = 150,shadow=True)


# The proportion of males is higher suspects than victims (92.4% > 76.1%)

# ###### Types of weapons used

# In[ ]:


data.weapon.value_counts().plot('bar')


# I guess knife is the most commonplace weapon people can easily access to kill people

# In[ ]:


(data.Status.value_counts() * 100 / data.Status.value_counts().sum()).plot('bar')
plt.title("% proportion homicide cases by Status")


# # Bivariate & Multivariate EDA

# We first extract year and month information from the date column and make them into sepearate columns for convenience

# In[ ]:


data['date_year'] = data.date.dt.year
data['date_month'] = data.date.dt.month


# ###### How has the total number of homicides change over time (year and month)

# In[ ]:


data.groupby('date_year').size().plot()
plt.title("Number of homicides in London over time")


# The total number of homicides was on the decreasing trend until 2014 but it has been increasing again ever since!

# In[ ]:


data.groupby('date_month').size().plot('bar')
plt.title("Number of homocides by month")


# ###### Which weapons are male suspects more likely to use than female suspects?

# In[ ]:


f, axes = plt.subplots(1,2, figsize=(10,4),sharey=True)

ax1 = (data[data.sussex=='M'].weapon.value_counts() * 100 / data[data.sussex=='M'].weapon.value_counts().sum()).plot('bar', ax=axes[0])
ax1.set_ylabel("% Proportion")
ax1.set_title("Proportion of Male Suspects by weapon")

ax2 = (data[data.sussex=='F'].weapon.value_counts() * 100 / data[data.sussex=='F'].weapon.value_counts().sum()).plot('bar', ax=axes[1])
ax2.set_ylabel("% Proportion")
ax2.set_title("Proportion of Female Suspects by weapon")
plt.tight_layout()


# From the observed data, we see that knives and guns are common weapons of choie for male suspects while non-traditional ways of homicide that don't require weapons (as indicated by "None", "Other", "Poison") are preferred by female suspects. But is this difference statistically significant? We can use the Chi Squared Test of Indepence for this!

# In[ ]:


from scipy.stats import chi2_contingency
table = pd.crosstab(data.sussex, data.weapon)
stat, p, dof, expected = chi2_contingency(table)
print("p-value: ",p) 


# The p-value is smaller than 0.05 and thus suspect's sex and the choice of weapon are not independent. They are correlated! Male suspects are more likely to use some form of weapon most likely because of their stronger physical strength.

# ###### Is there a statistically significant difference in average number of homicides in which victims were females between gang present and non-present boroughs?

# In[ ]:


# Add information about number of homicides in which victims were females to the original dataframe
data = data.merge(pd.DataFrame(data[data.vicsex=='F'].groupby('ladnm').size(),columns=['num_of_female_vic_cases']).reset_index(),
           how='left',on=['ladnm'])


# In[ ]:


# Number of homicides in which victims were females by gang present status
data.groupby('ladnm').first().groupby('gangpresent').mean().reset_index().plot(y='num_of_female_vic_cases',x= 'gangpresent', kind='barh')


# In[ ]:


# Variance of Number of homicides in which victims were females in gang present boroughs
data.groupby('ladnm').first().groupby('gangpresent').get_group('Y').num_of_female_vic_cases.var()


# In[ ]:


# Variance of Number of homicides in which victims were females in gang non-present boroughs
data.groupby('ladnm').first().groupby('gangpresent').get_group('N').num_of_female_vic_cases.var()


# Boroughs where gangs are present have more cases of homicide cases where victims were females. Is this statistically significant? We can use the independent two sample t-test with non-equal variance(we just checked above that the variances of the two groups are different) to test this!

# In[ ]:


from scipy.stats import ttest_ind
d1,d2 = data.groupby('ladnm').first().groupby('gangpresent').get_group('Y').num_of_female_vic_cases,data.groupby('ladnm').first().groupby('gangpresent').get_group('N').num_of_female_vic_cases

ttest_ind(d1,d2, equal_var=False)


# p-value is smaller than 0.05 and thus it is statistically significant! Thus, boroughs with gangs have significantly more cases of homicide where victims were females!

# ###### Victim Profile

# In[ ]:


# Chi Squared Test of Indepenence on Victims' ethnicity and sex
chi2_contingency(pd.crosstab(data.vicethnic, data.vicsex))


# P-value is 1.865400421831358e-14 < 0.05 and thus there is statistically significant correlation between victims' ethnicity and age. We can perform 2x2 post-hoc tests to find out which category is correlated to which category

# In[ ]:


data2 = data.copy() # Make a copy of data just for this test
data2 = data2.join(pd.get_dummies(data2.vicethnic)) # Join the dummy variables onto the original dataframe


# In[ ]:


# Chi Squared test of indepence on relationship between victims having any other ethnic appearance and their sex
chi2_contingency(pd.crosstab(data2['Any Other Ethnic Appearance'], data2.vicsex))


# In[ ]:


# Chi Squared test of indepence on relationship between victims being White or White British and their sex
chi2_contingency(pd.crosstab(data2['White or White British'], data2.vicsex))


# In[ ]:


# Chi Squared test of indepence on relationship between victims being Black or Black British and their sex
chi2_contingency(pd.crosstab(data2['Black or Black British'], data2.vicsex))


# In[ ]:


# Chi Squared test of indepence on relationship between victims being Asian or Asian British and their sex
chi2_contingency(pd.crosstab(data2['Asian or Asian British'], data2.vicsex))


# In[ ]:


# Percentage Proportion normalized by row
pd.crosstab(data.vicethnic, data.vicsex,normalize='index')


# Asian females had higher risk of being killed than females of other races. Black men had higher risk of being killed than men of other racces. 

# ###### What are some variables that are statistically significant with "Status"(Whether cases are resolved or unsolved etc.)

# In[ ]:


# Chi Squared Test of Indepenence on suspects' sex and status
chi2_contingency(pd.crosstab(data.sussex, data.Status))


# Suspects' sex is not much correlated with the case Status (statistically insignificant)

# In[ ]:


# Chi Squared Test of Indepenence on choice of weapons and status
chi2_contingency(pd.crosstab(data.weapon, data.Status))


# In[ ]:


pd.crosstab(data.weapon, data.Status, normalize='columns')


# Choice of weapons and case status is correlated but need to perform post-hoc chi squared tests on every single weapon. We skip this for now. But the table above shows that 51% among all solved cases were homicides that involved knives and 38% among all unsolves cases were homicides that involved guns.

# In[ ]:


# Chi Squared Test of Indepenence on victims' ethnicity and status
chi2_contingency(pd.crosstab(data.vicethnic, data.Status))


# In[ ]:


pd.crosstab(data.vicethnic, data.Status,normalize='index')


# If the victim was Black, it is more likely that the case has been unsolved compared to other cases regarding White victims. If the victim was White, it is more likely that the case has been solved compared to other cases regarding victims of color. Some evidence of ethnic bias?

# In[ ]:


# Chi Squared Test of Indepenence on Victims' sex and status
chi2_contingency(pd.crosstab(data.vicsex, data.Status))


# In[ ]:


pd.crosstab(data.vicsex, data.Status, normalize='index')


# If the victim was a man, it is less likely that the case was solved

# ###### Geospatial patterns of victim profile

# In[ ]:


# Victims' ethnicity in various boroughs(location)
sns.scatterplot(x='latitude',y='longitude',data=data,hue='vicethnic')
plt.legend(loc='best', bbox_to_anchor=(1,1))


# The scatter plot above suggests that White victims(blue dots) were killed in broader regions across London while Asian(red dots) or Black(Green dots) victims were mainly killed in central regions (less variance)

# In[ ]:


# Victims' sex in various boroughs(location)
sns.scatterplot(x='latitude',y='longitude',data=data,hue='vicsex')
plt.legend(loc=1)


# In[ ]:


# Suspects' sex in various boroughs(location)
sns.scatterplot(x='latitude',y='longitude',data=data,hue='sussex')


# ###### Relationship amongst population, IMD Decile and number of cases where victims were females of boroughs

# In[ ]:


sns.pairplot(data.groupby('ladnm').first().reset_index(), vars=['population','imddecile','num_of_female_vic_cases'], hue='ladnm')


# Boroughs with bigger population are more likely to have higher number of homicide cases where victims are females. I guess this this natural because if the population is bigger, more people are exposed to crimes like homicide. Also, IMD decile and the number of homicide cases where victims are females also has some correlation although the strength of the correlation doesn't seem to be that strong. The higher the IMD decile (presumably, the more deprived a certain borough is... need more info on this variable), the lower the number of homicide cases where victims are females.

# ### Thank you for reading my kernel! If you liked my kernel, please upvote it! Happy new year and happy Kaggling! :)
