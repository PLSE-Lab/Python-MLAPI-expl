#!/usr/bin/env python
# coding: utf-8

# # Step 1. Import necessary libraries and read dataset

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import ttest_1samp, ttest_ind,wilcoxon,mannwhitneyu
from statsmodels.stats.power import ttest_power


# In[ ]:


#Using pandas to read csv file
data = pd.read_csv('../input/insurance.csv')


# **2. Read the data as a Dataframe**

# In[ ]:


mydata = pd.DataFrame(data)
mydata


# In[ ]:


mydata.head() 
#displays the top 5 rows of the dataset


# # Step 2. Perform Basic EDA

# **3.a). Shape of data**

# In[ ]:


mydata.shape


# Gives shape of the data, 1338 rows and 7 columns

# **3.b). Datatype of each attribute**

# In[ ]:


mydata.info() 
#can also use mydata.dtypes to find datatype of each attribute


# **3.c). Checking presence of missing values**

# In[ ]:


mydata.isnull().sum()


# No Null values present

# **3.d). 5 point summary of all numerical attributes**

# In[ ]:


#5 point summary will give Xmin, 25th percentile, Median, 75th percentile,Xmax values
mydata.describe()


# In[ ]:


#Additional information of all attributes
mydata.describe(include = 'all')


# **3.e). Distribution of 'bmi', age and charges columns**

# In[ ]:


#Since these columns are continuous variables hence using pairplot
#Pairplot plots frequency distribution (histogram) & scatter plots
sns.pairplot(mydata[['bmi','age','charges']])


# Plotting individual distribution charts for BMI, Age, Charges columns
# 
# 1). Plot the Distribution of 'bmi' column, it follows Normal distribution

# In[ ]:


sns.distplot(data['bmi'])


# 2). Plot Distribution of 'age' column

# In[ ]:


sns.distplot(data['age'])


# 3). Plot the Distribution of 'charges' column, which is right skewed 

# In[ ]:


sns.distplot(data['charges'])


# 4). Plot distribution of 'children' using countplot

# In[ ]:


sns.countplot(data['children'], hue = data['sex'])


# There's significant count of population who don't have children

# **3.f). Measure of skewness of Bmi, age & charges columns**

# In[ ]:


#Skewness is a measure of attribute's symmetry.
mydata.skew(axis = None)


# when,
# * skewness = 0 : normally distributed
# * skewness > 0 : right skewed
# * skewness < 0 : left skewed

# **3.g). Check presence of outliers in Bmi, age, charges**

# In[ ]:


#Outliers are exceptions which are undesirable. Boxplots depict outliers as *
sns.boxplot(data = mydata, orient = 'h')


# 1). Displaying outliers in bmi data

# In[ ]:


sns.boxplot(mydata['bmi'])


# 2). Displaying outliers in age data

# In[ ]:


sns.boxplot(mydata['age'])


# 3). Displaying outliers in charges data

# In[ ]:


sns.boxplot(mydata['charges'])


# There are outliers in bmi & charges columns. While there are no outliers in age column.

# **3.h). Distribution of Categorical columns include children**

# There are 4 categorical columns Age, Sex, Children, Region, Smoker
# 
# To show distribution of Categorical columns, Violinplot can be used to depict scatter of the data

# In[ ]:


sns.catplot(x = 'region', y = 'children', data = mydata, hue = 'sex', kind = 'violin', col = 'smoker')


# **3.i). Pairplot that includes all of the columns of the data**

# Pairplot is useful to show distribution between various columns

# In[ ]:


sns.pairplot(data = mydata, hue = 'region')


# In[ ]:


sns.pairplot(data = mydata, hue = 'smoker')


# In[ ]:


sns.pairplot(data = mydata, hue = 'sex')


# # Step 3. Perform Statistical Analysis

# **4.a).Do charges of people who smoke differ significantly from the people who don't?**

# In[ ]:


#Null hypothosis: H0 = Smoking does not affect Insurance Charges
#Alternate hypothesis: Ha = Smoking does affect Insurance Charges

Yes = np.array(mydata[mydata['smoker'] == 'yes']['charges'])
No = np. array(mydata[mydata['smoker'] == 'no']['charges'])
fig = plt.figure(figsize = (10,6))
sns.distplot(Yes)
sns.distplot(No)
fig.legend(labels = ["Yes","No"])


# As per the above distribution, smoking does affect Insurance charges. Let's verify this using statistical analysis 

# In[ ]:


#Using 2 sided T test for independent samples

t_statistic, p_value = stats.ttest_ind(Yes,No)
t_statistic, p_value


# t_statistic is 46.66 standard deviations away from the expected value while p_value helps to determine significance of the results. Here p_value is negligible

# In[ ]:


if p_value < 0.05:
    print("Reject Null hypothesis")
else:
    print("Fail to reject Null hypothesis")


# **Answer: Charges of people who smoke differs significantly from the people who don't smoke**

# Using Mannwhitneyu Test to further test this statistically

# In[ ]:


#Using mannwhitneyu test
u_statistic, p_value = mannwhitneyu(Yes, No)
u_statistic, p_value
#u_statistic, p_value leads us to reject Null hypothesis


# Mannwhitneyu test gives a large u_statistic and a very insignificant p value, hence we reject Null Hypothesis   

# In[ ]:


#Calculate Power of test - This is the probability of rejecting the Null hypothesis
#To show how statistically significant is the mannwhitneyu test
(np.mean(Yes) - np.mean(No))/ np.sqrt(((len(Yes) - 1)*np.var(Yes) + (len(No) - 1)*np.var(No))/ len(Yes) + len(No)-2)


# In[ ]:


print(ttest_power(1.4333, nobs = (len(Yes) + len(No)), alpha = 0.05, alternative = 'two-sided'))


# This clearly states the Power of test is 1.0 which means Null hypothesis will be rejected 10 out of 10 times

# **4.b).Does bmi of males differ significantly from that of females?**

# In[ ]:


#Null Hypothesis: H0 = Bmi of Males do not differ significantly from that of Females
#Alternate Hypothesis: Ha = Bmi of Males differs significantly from that of Females
#2 sided T test for independent samples
bmi_male = np.array(mydata[mydata['sex'] == 'male']['bmi'])
bmi_female = np.array(mydata[mydata['sex'] == 'female']['bmi'])

fig = plt.figure(figsize = (10,6))
sns.distplot(bmi_male)
sns.distplot(bmi_female)
fig.legend(labels = ["BMI_Male","BMI_Female"])


# As per the above distribution, we can say distribution between BMI of Male and Female does not differ significantly. Let's verify this using statistical analysis 

# In[ ]:


#Using 2 sided T test for independent samples
t_statistic,p_value = stats.ttest_ind(bmi_male,bmi_female)
t_statistic,p_value


# t_statistic is 1.69 standard deviations away from the expected value while p_value helps to determine significance of the results.

# In[ ]:


if p_value < 0.05:
    print("Reject Null hypothesis")
else:
    print("Fail to Reject Null hypothesis")


# **Answer: BMI of Males does not differ significantly with that of BMI of Females**

# In[ ]:


#Using mannwhitneyu test
u_statistic, p_value = mannwhitneyu(bmi_male, bmi_female)
u_statistic, p_value
#u_statistic, p_value leads us to Fail to reject Null hypothesis


# Mannwhitneyu test gives a large u_statistic while p_value is 0.05, hence we Fail to reject Null hypothesis

# In[ ]:


#Calculate Power of test - This is the probability of rejecting the Null hypothesis
#To show how statistically significant is the mannwhitneyu test
(np.mean(bmi_male) - np.mean(bmi_female))/ np.sqrt(((len(bmi_male) - 1)*np.var(bmi_male) + (len(bmi_female) - 1)*np.var(bmi_female))/ len(bmi_male) + len(bmi_female)-2)


# In[ ]:


print(ttest_power(0.020, nobs = (len(bmi_male) + len(bmi_female)), alpha = 0.05, alternative = 'two-sided'))


# This clearly states the Power of test is 0.1 which means Null hypothesis will be rejected 1 out of 10 times as in Null hypothesis will be accepted 9 out of 10 times

# **4.c). Is the proportion of smokers significantly different in different genders?**

# In[ ]:


#As Smokers and Gender are categorical variables hence we use Proportions test
#Null hypothesis: H0 = Proportion of smokers do not differ significantly in different genders
#Alternate hypothesis: Ha = Proportion of smokers differs significantly in different genders

pd.crosstab(mydata['sex'], mydata['smoker'], margins = True)


# In[ ]:


sns.countplot(mydata['sex'], hue = mydata['smoker'])


# As per above graph, proportion of smokers seem to be significantly different in different genders. Let's study this further using statistical analysis.

# In[ ]:


#Ex11 = Expected value of Smoker = No & Sex = Female
#Ex12 = Expected value of Smoker = No & Sex = Male
#Ex21 = Expected value of Smoker = Yes & Sex = Female
#Ex22 = Expected value of Smoker = Yes & Sex = Male
Ex11 = (1064 * 662) / 1338
Ex12 = (1064 * 676) / 1338
Ex21 = (274 * 662) / 1338
Ex22 = (274 * 676) / 1338


# In[ ]:


from statsmodels.stats.proportion import proportions_ztest

z_stats, p_val = proportions_ztest([115,159], [662,676])
z_stats, p_val


# As per z_statistics, the proportion differs 2.78 standard deviations below the expected value

# In[ ]:


if p_val < 0.05:
    print('Reject Null Hypothesis')
else:
    print('Fail to Reject Null Hypothesis')


# **Answer: Proportion of smokers differs significantly with Gender**

# As Smoker & Gender are categorical variables hence we use Chi-Square test to further test the relationship between them

# In[ ]:


#Using Chi square test sum(Obs - Exp)^2/ Exp
observed_values = scipy.array([547,517,115,159])
n = observed_values.sum()
expected_values = scipy.array([Ex11,Ex12,Ex21,Ex22])
chi_square_stat, p_value = stats.chisquare(observed_values, f_exp=expected_values)
chi_square_stat, p_value


# In[ ]:


#Degree of freedom for chi square test = (row - 1)(col -1)
dof = (2-1)*(2-1)
dof


# In[ ]:


#Using Chi-square distribution table, we should check if chisquare stat of 7.765 exceeds 
#critical value of chisquare distribution. Critical value of alpha is 0.05 for 95% confidence 
#which is 3.84. As 7.765 > 3.84, we can reject Null Hypothesis

if chi_square_stat > 3.84:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# **Answer: Proportion of smokers differs significantly with Gender**

# **4.d). Is the distribution of bmi across women with no children, one child and two children, the same ?**

# In[ ]:


#Null Hypothesis: H0 = Distribution of BMI for women with 0,1,2 children is same
#Alternate Hypothesis: Ha = Distribution of BMI for women with 0,1,2 children is not same

bmidata = mydata[(mydata['children'] <= 2) & (mydata['sex'] == 'female')][['sex','bmi', 'children']]
bmidata.head()


# In[ ]:


#Grouping into 3 groups, 0,1,2 children
zero_ch = np.array(bmidata[bmidata['children'] == 0]['bmi'])
one_ch = np.array(bmidata[bmidata['children'] == 1]['bmi'])
two_ch = np.array(bmidata[bmidata['children'] == 2]['bmi'])


# In[ ]:


#Relationship between Bmi and children for women
bmigraph = sns.jointplot(bmidata['bmi'],bmidata['children'])
bmigraph = bmigraph.annotate(stats.pearsonr, fontsize=10, loc=(0.2, 0.8))


# In[ ]:


sns.boxplot(x = 'children', y = 'bmi', data = bmidata)


# From both the above graph we can derive BMI distribution for women with 0,1,2 children are same.
# 
# Let's analyze this further using statistical analysis.

# In[ ]:


#Use One Way ANOVA for 3 sample groups 
#Null Hypothesis: H0: mean(zero_ch) = mean(one_ch) = mean(two_ch)
#Alternate Hypothesis: Ha: One of the means would differ


# Here we have three groups. Analysis of variance can determine whether the means of three or more groups are different. ANOVA uses F-tests to statistically test the equality of means.

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
 
mod = ols('bmi ~ children', data = bmidata).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


# Here p_value is 0.79376 > 0.05 hence we Fail to Reject Null Hypothesis therefore 

if 0.79 > 0.05:
    print("Fail to Reject Null Hypothesis")
else:
    print("Reject Null Hypothesis")


# ****Answer: Distribution of BMI for women with 0,1,2 children is same****
