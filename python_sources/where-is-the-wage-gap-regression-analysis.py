#!/usr/bin/env python
# coding: utf-8

# The gender gap in wages has been shown [many times][1] in different papers. Quite consistently, researchers find that there's a 20-30% gap between average wages, which is reduced to 5-10% when adjusting for all the factors.
# 
# Can we reproduce these results with the IBM data set?
# 
# ##*Important comment - as it turns out this is a fictional data, the conclusions should not be taken seriously. The analysis process might still be interesting though. This is probably why it seems that the wage gap is tiny and elusive.##
# 
#   [1]: https://www.glassdoor.com/research/studies/gender-pay-gap/

# In[36]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from subprocess import check_output
from collections import OrderedDict
matplotlib.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")


# In[14]:


path = '../input/'
filename = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

df = pd.read_csv(path + filename)

df.head(50)


# Let's explore the income distribution for men and women:

# In[37]:


sns.distplot(df.MonthlyIncome[df.Gender == 'Male'], bins = np.linspace(0,20000,60))
sns.distplot(df.MonthlyIncome[df.Gender == 'Female'], bins = np.linspace(0,20000,60))
plt.legend(['Males','Females'])


# Interestingly, the males histogram does not seem to be more to the right. let's quantify it and see whether the mean and median tell a different story:

# In[16]:


avg_male = np.mean(df.MonthlyIncome[df.Gender == 'Male'])
avg_female = np.mean(df.MonthlyIncome[df.Gender == 'Female'])

med_male = np.median(df.MonthlyIncome[df.Gender == 'Male'])
med_female = np.median(df.MonthlyIncome[df.Gender == 'Female'])

std_male = np.std(df.MonthlyIncome[df.Gender == 'Male'])
std_female = np.std(df.MonthlyIncome[df.Gender == 'Female'])

plt.bar([1,4],[avg_male, med_male])
plt.bar([2,5],[avg_female, med_female], color = 'r')
plt.xticks([2,5],['Average Monthly Income','Median Monthly Income'])
plt.legend(['Males','Females'])

print(avg_female/avg_male)


# Surprisingly, a woman makes 1.04$ for each 1$ a man earns. So where is the wage gap? 
# 
# Let's adjust for the different features. first, see if the monthly rate is different. I was assuming that the rate would be the income divided by the monthly hours. But this does not seem to be case, so I am not sure what is the meaning of this feature. Let's explore it nonetheless:

# In[17]:


sns.distplot(df.MonthlyRate[df.Gender == 'Male'], bins = np.linspace(0,30000,60), kde = False)
sns.distplot(df.MonthlyRate[df.Gender == 'Female'],bins = np.linspace(0,30000,60), kde = False)
plt.legend(['Males','Females'])
plt.show()

avg_male_rate = np.mean(df.MonthlyRate[df.Gender == 'Male'])
avg_female_rate = np.mean(df.MonthlyRate[df.Gender == 'Female'])

med_male_rate = np.median(df.MonthlyRate[df.Gender == 'Male'])
med_female_rate = np.median(df.MonthlyRate[df.Gender == 'Female'])

plt.bar([1,4],[avg_male_rate, med_male_rate])
plt.bar([2,5],[avg_female_rate, med_female_rate], color = 'r')
plt.xticks([2,5],['Average Monthly Rate','Median Monthly Rate'])
plt.legend(['Males','Females'], loc = 2)

print(avg_female_rate/avg_male_rate)


# Just like with the average income, the females rate is higher.  The ratio is identical - the average female rate is 4% higher. So whatever this feature is, it is highly correlated with the income. 
# 
# Let's continue our naive approach of controlling for different factors, before engaging in regressions. 
# 
# Now we will look at the different departments:

# In[18]:


plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.title('Sales')
sns.distplot(df.MonthlyIncome[(df.Department == 'Sales') & (df.Gender == 'Male')], bins = np.linspace(0,20000,60))
sns.distplot(df.MonthlyIncome[(df.Department == 'Sales') & (df.Gender == 'Female')], bins = np.linspace(0,20000,60))
plt.xlabel('')

plt.subplot(3,1,2)
plt.title('R&D')
sns.distplot(df.MonthlyIncome[(df.Department == 'Research & Development') & (df.Gender == 'Male')], bins = np.linspace(0,20000,60))
sns.distplot(df.MonthlyIncome[(df.Department == 'Research & Development') & (df.Gender == 'Female')], bins = np.linspace(0,20000,60))
plt.xlabel('')

plt.subplot(3,1,3)
plt.title('HR')
sns.distplot(df.MonthlyIncome[(df.Department == 'Human Resources') & (df.Gender == 'Male')], bins = np.linspace(0,20000,60))
sns.distplot(df.MonthlyIncome[(df.Department == 'Human Resources') & (df.Gender == 'Female')], bins = np.linspace(0,20000,60))


# It seems that sales is the only department where males earn more on average. Let's see how this quantifies:

# In[19]:


males_sales = np.mean(df.MonthlyIncome[(df.Department == 'Sales') & (df.Gender == 'Male')])
females_sales = np.mean(df.MonthlyIncome[(df.Department == 'Sales') & (df.Gender == 'Female')])

males_rnd = np.mean(df.MonthlyIncome[(df.Department == 'Research & Development') & (df.Gender == 'Male')])
females_rnd = np.mean(df.MonthlyIncome[(df.Department == 'Research & Development') & (df.Gender == 'Female')])

males_HR = np.mean(df.MonthlyIncome[(df.Department == 'Human Resources') & (df.Gender == 'Male')])
females_HR = np.mean(df.MonthlyIncome[(df.Department == 'Human Resources') & (df.Gender == 'Female')])


plt.bar([1,4,7],[males_sales,males_rnd,males_HR])
plt.bar([2,5,8],[females_sales,females_rnd,females_HR])

plt.xticks([1.5,4.5,7.5],['Sales','R&D','HR'])
plt.legend(['Males','Females'])
plt.ylabel('Average Monthly Income')


# The average male at the sales department still earns less than the average female, though the difference is smaller compared to the other departments. Also, looking at the histograms it is clear that the average is not the best metric.
# 
# So let us see if we can use a regression in order to determine the influence of the gender feature on the income (as well as predicting the salary of an employee, if we're already here. A reasonable score of this model is of course also necessary to validate our model it meaningful and not over-fitting)
# 
# ## Regression
# 
# Let's start with a small number of features:

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

model = LinearRegression()
df.Gender[df.Gender == 'Male'] = 0
df.Gender[df.Gender == 'Female'] = 1

from sklearn.model_selection import train_test_split


columns = ['Age', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction','StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

X = df[columns]

df_std = StandardScaler().fit_transform(X)
df_std = pd.DataFrame(df_std)
df_std.columns = columns
df_std['Gender'] = df.Gender
y = df.MonthlyIncome

columns = ['Age', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction','StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager','Gender']

X_train, X_test, y_train, y_test = train_test_split(df_std, y, test_size=0.33, random_state=42)

model.fit(X_train,y_train)

def multiassign(d, keys, values):
    d.update(zip(keys, values))
    
plot_dict = dict()
multiassign(plot_dict,columns,model.coef_)
plot_dict = OrderedDict(sorted(plot_dict.items(), key=lambda x: x[1]))

fig, ax = plt.subplots(figsize = (10,10))
ax.barh(range(len(columns)),list(plot_dict.values()), align='center')
ax.set_yticks(range(len(columns)))
ax.set_yticklabels(plot_dict.keys())

ax.set_title('Regression Coefficients')

print('Test set R^2...',model.score(X_test,y_test))
print('Training set R^2...',model.score(X_train,y_train))


# By far the most important feature for the regression is the total working years, which makes sense. As females were coded as "1" in the binary feature "Gender", we see that the gender coefficient is positive. which means that based on this model, being a female is correlative with higher wages even when controlling for the age, working years, etc.
# 
# Our model, however, has a relatively low R^2 value based on what I would expect. Let's add more features to the pot and see what can we cook out of it:

# In[40]:


model = LinearRegression()


columns = ['Age', 'DailyRate', 
       'DistanceFromHome', 'Education', 
         'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction',
       'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

X = df[columns]

df_std = StandardScaler().fit_transform(X)
df_std = pd.DataFrame(df_std)
df_std.columns = columns
df_std['Gender'] = df.Gender
y = df.MonthlyIncome

columns =  ['Age', 'DailyRate', 
        'DistanceFromHome', 'Education', 
         'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel','JobSatisfaction',
       'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager','Gender']

X_train, X_test, y_train, y_test = train_test_split(df_std, y, test_size=0.33, random_state=42)

model.fit(X_train,y_train)

plot_dict = dict()
multiassign(plot_dict,columns,model.coef_)
plot_dict = OrderedDict(sorted(plot_dict.items(), key=lambda x: x[1]))

fig, ax = plt.subplots(figsize = (10,10))
ax.barh(range(len(columns)),list(plot_dict.values()), align='center')
ax.set_yticks(range(len(columns)))
ax.set_yticklabels(plot_dict.keys())

print('Test set R^2...',model.score(X_test,y_test))
print('Training set R^2...',model.score(X_train,y_train))


# While the model now scores much better, I am quite suspicious about the Job Level feature. This might be considered as a leak, as it probably a very very strong proxy to the income. Let's see how the salary depends on the strongest features:

# In[22]:


plt.figure(figsize = (10,10))
plt.subplot(2,1,1)
plt.plot(df.JobLevel,df.MonthlyIncome,'o', alpha = 0.01)
plt.xlabel('Job Level')
plt.ylabel('Monthly Income')

plt.subplot(2,1,2)
plt.plot(df.TotalWorkingYears + np.random.normal(0,0.5,len(df)),df.MonthlyIncome,'o', alpha = 0.2)
plt.xlabel('Total Working Years')
plt.ylabel('Monthly Income')


# I would not consider the Job Level as a good feature when we try to predict the income based on features which are supposed to be at least somewhat non-correlated with the target (otherwise what is the point). 
# 
# The total working years is more legit in my opinion, though the distribution is a bit odd and seems unnatural (with a very large amount of employees working for 10 years):

# In[41]:


sns.distplot(df.TotalWorkingYears, bins = np.arange(min(df.TotalWorkingYears),max(df.TotalWorkingYears),1), kde =False)
plt.ylabel('Number of Employees')


# Do all the other features, besides JobLevel, even matter? that is, how would a regression model based only on the Job Level feature would perform? 

# In[42]:


model = LinearRegression()

columns = [ 'JobLevel']

X = df[columns]

df_std = StandardScaler().fit_transform(X)
df_std = pd.DataFrame(df_std)
df_std.columns = columns

y = df.MonthlyIncome

columns = [ 'JobLevel']

X_train, X_test, y_train, y_test = train_test_split(df_std, y, test_size=0.33, random_state=42)

model.fit(X_train,y_train)

print('Test set R^2...',model.score(X_test,y_test))
print('Training set R^2...',model.score(X_train,y_train))


# As I suspected - about 90% of the variance can be explained by this feature. Therefor, I would omit it from here on. Let's see how our model performs without that, but using all the other continuous features: 

# In[44]:


model = LinearRegression()


columns = ['Age', 'DailyRate', 
       'DistanceFromHome', 'Education', 
         'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement','JobSatisfaction',
       'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

X = df[columns]

df_std = StandardScaler().fit_transform(X)
df_std = pd.DataFrame(df_std)
df_std.columns = columns
df_std['Gender'] = df.Gender
y = df.MonthlyIncome

columns =  ['Age', 'DailyRate', 
        'DistanceFromHome', 'Education', 
         'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement','JobSatisfaction',
       'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager','Gender']

X_train, X_test, y_train, y_test = train_test_split(df_std, y, test_size=0.33, random_state=42)

model.fit(X_train,y_train)

plot_dict = dict()
multiassign(plot_dict,columns,model.coef_)
plot_dict = OrderedDict(sorted(plot_dict.items(), key=lambda x: x[1]))

fig, ax = plt.subplots(figsize = (10,10))
ax.barh(range(len(columns)),list(plot_dict.values()), align='center')
ax.set_yticks(range(len(columns)))
ax.set_yticklabels(plot_dict.keys())

print('Test set R^2...',model.score(X_test,y_test))
print('Training set R^2...',model.score(X_train,y_train))


#  1. The Gender coefficient is still positive
#  2. The general performance of the model has not improved much compared to our initial benhmark which used only a handful of features 
# 
# Let's see if the categorical features can improve the situation. 
# 
# first. we'll have to code them into binary features for each category (one hot encoding):

# In[45]:


categorical = pd.get_dummies(df[['EducationField','JobRole','MaritalStatus','Department']])
categorical['Gender'] = df.Gender
categorical = categorical.drop(['EducationField_Human Resources','JobRole_Healthcare Representative','Department_Human Resources','MaritalStatus_Divorced'], axis = 1)


# In[46]:


model = LinearRegression()

X = categorical
y = df.MonthlyIncome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model.fit(X_train,y_train)

plot_dict = dict()
multiassign(plot_dict,categorical.columns,model.coef_)
plot_dict = OrderedDict(sorted(plot_dict.items(), key=lambda x: x[1]))

fig, ax = plt.subplots(figsize = (10,10))
ax.barh(range(len(categorical.columns)),list(plot_dict.values()), align='center')
ax.set_yticks(range(len(categorical.columns)))
ax.set_yticklabels(plot_dict.keys())

print('Test set R^2...',model.score(X_test,y_test))
print('Training set R^2...',model.score(X_train,y_train))


# The categorical model performs much better, with the job role being the most important feature (as one can expect). This looks like a better control.
# 
# Also, the gender coefficient is, finally, negative.
# 
# ## Have we found the gender gap? 
# 
# Let's rejoin all the legitimate features to our final linear regression:

# In[47]:


result = pd.concat([df, categorical], axis=1)


# In[48]:


model = LinearRegression()


columns = ['Age', 'DailyRate', 
       'DistanceFromHome', 'Education', 
         'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement','JobSatisfaction',
       'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

X = df[columns]

df_std = StandardScaler().fit_transform(X)
df_std = pd.DataFrame(df_std)
df_std.columns = columns
df_std['Gender'] = df.Gender
df_std = pd.concat([df_std,categorical.drop(['Gender'],axis = 1)],axis = 1)
y = df.MonthlyIncome


X_train, X_test, y_train, y_test = train_test_split(df_std, y, test_size=0.33, random_state=42)

model.fit(X_train,y_train)

plot_dict = dict()
multiassign(plot_dict,df_std.columns,model.coef_)
plot_dict = OrderedDict(sorted(plot_dict.items(), key=lambda x: x[1]))

fig, ax = plt.subplots(figsize = (10,10))
ax.barh(range(len(df_std.columns)),list(plot_dict.values()), align='center')
ax.set_yticks(range(len(df_std.columns)))
ax.set_yticklabels(plot_dict.keys())

print('Test set R^2...',model.score(X_test,y_test))
print('Training set R^2...',model.score(X_train,y_train))


# Using all the features we get a validation score which is almost as high as the validation score using the Job level feature which is a great predictor for the income. and with this model, we see that the gender coefficient is indeed negative - that is, when controlling for all the available features, the contributing of being a female to the wage is negative. 
# 
# Based on this analysis, the wage gap is, sadly, observed in this data set too. although much more elusive than usually. 
# 
# One can of course wonder whether the fact that I was clearly looking for a certain conclusion impaired the statistical inference process. Do tell me your opinion in the comment section!
# 
# ## And again, since it turns out the data is fictional, do not conclude anything too dramatic :) ##
# 
