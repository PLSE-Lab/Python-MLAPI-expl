#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Data science is the practice of using scientific methods and algorithms to obtain insights from data. Fitted models are often used to predict future outcomes; however, they can also be used to obtain effect measures. Although not commonly used for performing epidemiological analyses, Python in conjunction with packages such as pandas and statsmodels can be a powerful tool very much worth learning. 
# 
# Using the Heart Disease UCI data set, we will perform an example of an epidemiological analysis of heart disease. The presence of heart disease in a patient is measured through a target variable as 1 (heart disease) or 0 (no heart disease). Thirteen other variables describe patient information and possible predictors of heart disease. For this analysis, we will focus on the resting electrocardiographic results (restecg). Our research question is:
# 
# *Do individuals with abnormal readings (restecg = 1) have an increased odds of heart disease compared to individuals with normal readings (restecg = 0)?*
# 
# Let's start by loading the relevant packages and reading in the data.

# In[ ]:


get_ipython().system('pip install --upgrade pandas')
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[ ]:


heart = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
heart.head()


# Before starting any exploratory analysis, let's see if we will need to do any preprocessing to address missing data.

# In[ ]:


# How many records are there
len(heart.index)


# In[ ]:


# Check for the existence of any missing values
heart.count()


# Great! There aren't any missing values (this rarely happens). Now let's get a list of the continuous and categorical variables and assign the correct data type for categoricals.

# In[ ]:


cont = ['age','trestbps','chol','thalach','oldpeak']
cat = [x for x in list(heart.columns) if x not in cont]
cat.remove('target')


# In[ ]:


for val in cat:
    heart[val] = heart[val].astype('category')


# # Descriptive Statistics

# Next let's group the data based on the presence of heart disease and get some statistics for each group, starting with the continuous variables.

# In[ ]:


heart_gp = heart.groupby('target')
stats_cont = heart_gp[cont].agg([np.mean, np.std]).stack().T
stats_cont


# For each measure, let's see if there are any significant differences between those with heart disease and those without heart disease. For continuous variables we can use a t-test, but first let's check that all assumptions for this test are met.
# 
# Let's start with homogeneity of variances. Homogeneity of variances can be checked using Levene's test, available in the scipy package.

# In[ ]:


#Split the dataframe into diseased and not diseased
disease = (heart[heart['target'] == 1]).reset_index()
no_disease = (heart[heart['target'] == 0]).reset_index()

results = {}
for val in cont:
    results[val] = stats.levene(disease[val], no_disease[val])
pd.DataFrame.from_dict(results, orient='Index', columns=['statistic','p-value'])


# It seems that not all variances are equal since some p-values are significant at 0.05 alpha. We can use Welch's t-test to overcome this, but first let's check for normal distribution of residuals. We can use a q-q plot to visually assess.

# In[ ]:


diff = disease['age'] - no_disease['age']
sm.qqplot(diff, line='q')
plt.title('Age Q-Q Plot') 
plt.show()


# More formally, we can use the Shapiro-Wilk test. The following will return a DataFrame with the p-value of the test.

# In[ ]:


shapiro = heart_gp.agg(lambda x:stats.shapiro(x)[1]).T
shapiro.loc[cont] < 0.05


# The data does not appear to be normally distributed. Depending on the sample size, this assumption can sometimes be relaxed as it may still be approximately normal. Alternatively, a non-parametric test such as the Mann Whitney U test can be used.
# 
# Let's plot the histograms and see if the data appear to be approximately normal.

# In[ ]:


fig = plt.figure()

plt.subplot(2, 2, 1)
disease['age'].plot(kind='hist', title='Age - Diseased Group')

plt.subplot(2, 2, 2)
no_disease['age'].plot(kind='hist', title='Age - Non-Diseased Group')


# In[ ]:


plt.subplot(2, 2, 1)
disease['trestbps'].plot(kind='hist', title='trestbps - Diseased Group')

plt.subplot(2, 2, 2)
no_disease['trestbps'].plot(kind='hist', title='trestbps - Non-Diseased Group')


# In[ ]:


plt.subplot(2, 2, 1)
disease['chol'].plot(kind='hist', title='chol - Diseased Group')

plt.subplot(2, 2, 2)
no_disease['chol'].plot(kind='hist', title='chol - Non-Diseased Group')


# In[ ]:


plt.subplot(2, 2, 1)
disease['thalach'].plot(kind='hist', title='thalach - Diseased Group')

plt.subplot(2, 2, 2)
no_disease['thalach'].plot(kind='hist', title='thalach - Non-Diseased Group')


# In[ ]:


plt.subplot(2, 2, 1)
disease['oldpeak'].plot(kind='hist', title='oldpeak - Diseased Group')

plt.subplot(2, 2, 2)
no_disease['oldpeak'].plot(kind='hist', title='oldpeak - Non-Diseased Group')


# Although the data look skewed, they do appear to be approximately normal. The exception is *oldpeak*. We can use the Mann Whitney U test for this value, but first let's calculate a Welch's t-test for the other values.

# In[ ]:


results = {}
for val in list(stats_cont.index):
    results[val] = stats.ttest_ind(disease[val],no_disease[val],equal_var=False)
test = pd.DataFrame.from_dict(results, orient='Index', columns=['statistic', 'p-value'])
test = test.drop('oldpeak', axis=0)
test


# Now let's do the Mann Whitney U test for *oldpeak*. We will also add it to our test results data frame.

# In[ ]:


mannwhit = stats.mannwhitneyu(disease['oldpeak'],no_disease['oldpeak'])
test = test.append(pd.Series({'statistic':mannwhit[0],'p-value':mannwhit[1]},name='oldpeak'))
test  


# All values except for cholesterol (chol) are significantly different between groups. Just for fun, let's add the p-value to our statistics table for continuous variables.

# In[ ]:


pd.concat([stats_cont,test['p-value']],axis=1)


# Now let's check the categorical data using chi-square test for independence. Note that we have small cell counts for *restecg*, *ca* and *thal*. For these variables, we will use Fisher's exact test.

# In[ ]:


heart_gp.sex.value_counts().unstack(0)


# In[ ]:


heart_gp.cp.value_counts().unstack(0)


# In[ ]:


heart_gp.fbs.value_counts().unstack(0)


# In[ ]:


heart_gp.restecg.value_counts().unstack(0)


# In[ ]:


heart_gp.exang.value_counts().unstack(0)


# In[ ]:


heart_gp.slope.value_counts().unstack(0)


# In[ ]:


heart_gp.ca.value_counts().unstack(0)


# In[ ]:


heart_gp.thal.value_counts().unstack(0)


# In[ ]:


results = {}
for val in cat:
    results[val] = stats.chi2_contingency(pd.crosstab(heart[val],heart.target))
test = pd.DataFrame.from_dict(results, orient='Index', columns=['statistic', 'p-value','df','expected'])
test.drop(['df','expected'],axis=1,inplace=True)
test.drop(['restecg','ca','thal'], inplace=True)
test


# Let's run a Fisher's exact test for the remaining three variables. Unfortunately, the contingency tables are all larger than 2x2, and there is no python function for performing the test on tables larger than 2x2. However, we can call R to perform the calculation.

# In[ ]:


# Run the next line if rpy2 is not already installed
get_ipython().system('conda install -c r rpy2 --yes ')


# In[ ]:


import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
statsr = importr('stats')


# In[ ]:


statsr.fisher_test(pd.crosstab(heart.restecg,heart.target).to_numpy())


# The result is a ListVector with information about the calculation performed in R. We are interested in the p-value for the two-sided test.

# In[ ]:


result = statsr.fisher_test(pd.crosstab(heart.ca,heart.target).to_numpy())
print('p-value: ',result[0])


# In[ ]:


result = statsr.fisher_test(pd.crosstab(heart.thal,heart.target).to_numpy())
print('p-value: ',result[0])


# From the results it appears that *fbs* is the only categorical value that is not significantly different between groups.
# 
# To summarize, all variables with the exception of cholestoral (chol) measured in mg/dl and fasting blood sugar (fbs) differ significantly between groups.
# 
# # Crude Odds Ratio
# 
# Let's now get the crude odds ratio of a having ST-T wave abnormality (restecg = 1) vs. a normal result (restecg = 0). Because this variable is not continuous or binary, we will need to create dummy variables for it. We will go ahead and create dummy variables for all categorical variables.

# In[ ]:


heart_dum = pd.get_dummies(heart, drop_first=True)
heart_dum.head()


# Since we will only be using *target* and the *restecg* dummies for calculating the crude OR, we will get a subset of the full data set to only include these values.

# In[ ]:


crude_calc = heart_dum.loc[:,['restecg_1','restecg_2','target']]
crude_calc.head()


# Now let's fit a logistic regression model.

# In[ ]:


y = crude_calc['target']
ind = crude_calc[['restecg_1','restecg_2']]
X = sm.add_constant(ind)
logit_model = sm.Logit(y,X)
result = logit_model.fit()
result.summary2()


# To calculate the crude odds ratio for restecg=abnormal vs. restecg=normal we just need to exponentiate the coefficient for restecg_1.

# In[ ]:


np.exp(result.params[1])


# To calculate the 95% confidence interval we use the standard error of the coefficient.

# In[ ]:


se = 0.2359
lower = result.params[1] - 1.96*se
upper = result.params[1] + 1.96*se
print("95% CI: (",np.exp(lower),",",np.exp(upper),")")


# Without controlling for any other variables, individuals with an abnormal reading seem to have almost twice the odds of heart disease compared to individuals with a normal ecg. The results are statistically significant and we have a 95% confidence that the true value lies somewhere in the range of (1.25, 3.16).

# # Adjusted Odds Ratio
# 
# Finally, let's fit a model that adjusts for other potential confounders. We want the most parsimonious model possible, so we will define a function to do feature selection by backward elimination.
# 
# In backward elimination, all variables are included in the model and then are iteratively eliminated if their p-value is less than a threshold defined by *alpha*. To ensure that we do not eliminate our variable of interest, we include a *keep* parameter to skip over variables regardless of their p-value. The process is repeated until all remaining variables are smaller than the threshold.

# In[ ]:


def backward_elimination(df, dep, ind, alpha, keep):
    y = dep
    x1 = list(ind.columns)
    while len(x1) > 1:
        X = sm.add_constant(df[x1])
        model = sm.Logit(y,X)
        result = model.fit()
        pvalues = pd.Series(result.pvalues.values[1:], index=x1)
        idmax = pvalues.idxmax()
        if idmax in keep:
            pvalues.sort_values(ascending=False, inplace=True)
            pvalues.drop(keep, inplace=True)
            idmax = pvalues.index[0]
        pmax = pvalues.loc[idmax]
        if pmax > alpha:
            x1.remove(idmax)
        else:
            return x1


# In[ ]:


dep = heart_dum['target']
ind = heart_dum[heart_dum.columns.difference(['target'])]
predictors = backward_elimination(heart_dum, dep, ind, 0.05, ['restecg_1', 'restecg_2'])
predictors


# We now fit the model with the selected predictors. All levels of dummy variables will be added even if they were not found to be significant.

# In[ ]:


predictors.extend(['ca_4','cp_1','slope_2','thal_3'])
predictors.sort()


# In[ ]:


x1 = heart_dum[predictors]
X = sm.add_constant(x1)
model = sm.Logit(y,X)
result = model.fit()
result.summary2()


# The p-value for restecg_1 is not significant and would have been eliminated during feature selection if we had not specified it as the *keep* argument. From these results, we know that the adjusted odds ratio will not be significant and we can infer that the significance seen for the crude odds ratio was due to the influence of other variables.

# In[ ]:


print('Adjusted Odds Ration: ', np.exp(result.params['restecg_1']))

# Confidence Interval
se = 0.3828
lower = result.params['restecg_1'] - 1.96*se
upper = result.params['restecg_1'] + 1.96*se
print('95% CI: (',np.exp(lower),',',np.exp(upper),')')


# # Conclusions
# 
# Although it is not appropriate to interpret non-significant odds ratios, because this is only an exercise, I will go ahead and do it anyway. The interpretation is as follows:
# 
# After adjusting for confounders, the odds of heart disease are approximately 75% greater when individuals presented an abnormal ecg compared to individuals with a normal ecg. However, these result did not show statistical significance at an alpha of 0.05.
