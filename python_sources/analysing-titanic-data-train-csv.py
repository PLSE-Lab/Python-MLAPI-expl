#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# **Step 1 - General Question:**
# 
# What category of people where most likely to survive?

# **Step 2 - Data Acqisition:**

# In[ ]:


titanic_data = pd.read_csv("../input/train.csv")


# ** Step 3 - Data Cleaning:**

# In[ ]:


head = titanic_data.head()
columns = titanic_data.columns
cases = titanic_data['PassengerId'].count()


# In[ ]:


head


# In[ ]:


columns


# In[ ]:


cases


# In[ ]:


titanic_data['Cabin'].isna().sum()


# **Observations:**
# 1. There are numerous counts of NaN for titanic_data['cabin']
# 2. The dataset has 891 cases with 12 observations in this dataset
# 3. PassengerId and Ticket observations will not be relevant to my analysis swo I will drop the columns

# a. Drop ('PassengerId', 'Cabin', 'Ticket') columns and round off 'Fare' for easier analysis.

# In[ ]:


new_data = titanic_data.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)
new_data['Fare_x'] = new_data['Fare'].round()


# In[ ]:


new_data.head()


# b. Extract titles from names and drop (name, fare) columns

# In[ ]:


new_data['title'] = new_data['Name'].apply(lambda words: words.split()[1])
new_data = new_data.drop(['Name','Fare'], axis = 1)


# In[ ]:


new_data[:2]


# 
# 
# Observation:
# 
# a. There are cases where titles are absent from names.

# In[ ]:


age_na = new_data['Age'].isna().sum()


# b. Age observation has 177 missing values 

# Options: 
# - replace nan with mean
# - interpolate with method 'index'

# In[ ]:


new_data['Age_nan_mean'] = new_data['Age'].fillna(new_data['Age'].mean())
new_data['Age_nan_index'] = new_data['Age'].interpolate(method = 'index')


# In[ ]:


new_data[:5]


# 
# **Step 4 - Data Exploration:**

# A.  Gender

# 1. The proportion of women to men?

# In[ ]:


women_to_men = new_data.groupby('Sex', as_index=False)['Survived'].count()
women_to_men = women_to_men.rename(columns={'Survived':'Count'})
women_to_men


# 2. The survival ratio by sex?

# In[ ]:


survival_by_sex = new_data.groupby('Sex', as_index=False)['Survived'].sum()
survival_by_sex


# 3. The survival ratio by sex in 'Class of Travel' (Pclass)?

# In[ ]:


survival_by_sex_pclass = new_data.groupby(['Pclass','Sex'], as_index=False)['Survived'].sum()
survival_by_sex_pclass


# 
# 4. Survival in class of travel (Pclass) by 'Embarked' and by sex?

# In[ ]:


x = sns.FacetGrid(new_data, row = 'Pclass', col = 'Embarked', hue = 'Sex')
x = x.map(plt.hist, 'Survived')
plt.legend(title='',loc ='right', labels = ['male', 'female'])
plt.show()


# In[ ]:


x = sns.FacetGrid(new_data, row = 'Pclass', col = 'Embarked')
x = x.map(plt.hist, 'Survived')
plt.show()


# B. Age

# 1. Test correlation

# In[ ]:


x = new_data.corr()
sns.heatmap(x).set_title('Correlation Map')
plt.show()


# In[ ]:


Max_age = new_data['Age'].max()
Max_age


# In[ ]:


mean_age = new_data.groupby(['Pclass', 'Sex'], as_index = False)['Age'].mean()
mean_age


# In[ ]:


mean_age_x = new_data.groupby(['Pclass', 'Sex'], as_index = False)['Age_nan_index'].mean()
mean_age_x


# In[ ]:


def quartiles(age):
    quart_1 = np.quantile(age, 0.25)
    quart_2 = np.quantile(age, 0.5)
    quart_3 = np.quantile(age, 0.75)
    quart_4 = np.quantile(age, 1)
    return (quart_1, quart_2, quart_3, quart_4)
quarts = quartiles(new_data['Age_nan_index'])
quarts_male = quartiles(new_data[new_data['Sex']=='male']['Age_nan_index'])
quarts_female = quartiles(new_data[new_data['Sex']=='female']['Age_nan_index'])


# In[ ]:


print('The overall quartiles are:')
print(quarts)
print('The quartiles for men are:')
print(quarts_male)
print('The quartiles for women are:')
print(quarts_female)


# In[ ]:


age_survival = new_data.groupby(['Pclass', 'Sex', 'Survived'], as_index = False)['Age'].mean()
age_survival[:4]


# In[ ]:


age_survival_nan = new_data.groupby(['Pclass', 'Sex', 'Survived'], as_index = False)['Age_nan_index'].mean()
age_survival_nan['combined'] = age_survival_nan['Pclass'].apply(lambda x:np.str(x)) + age_survival_nan['Sex'] + age_survival_nan['Survived'].apply(lambda x:np.str(x))
age_survival_nan[:4]


# In[ ]:


age_survival_male = new_data[new_data['Sex']=='male'].groupby(['Pclass', 'Sex', 'Survived'], as_index = False)['Age_nan_index'].mean()
age_survival_male['combined'] = age_survival_male['Pclass'].apply(lambda x:np.str(x)) + age_survival_male['Sex'] + age_survival_male['Survived'].apply(lambda x:np.str(x))
age_survival_male


# In[ ]:


age_survival_female = new_data[new_data['Sex']=='female'].groupby(['Pclass', 'Sex', 'Survived'], as_index = False)['Age_nan_index'].mean()
age_survival_female


# In[ ]:


age_join = age_survival_nan[age_survival_nan['Sex']=='male'].join(age_survival_male[['combined', 'Age_nan_index']].set_index('combined'), on = 'combined', how='left', lsuffix='_all', rsuffix='_male')
age_join = age_join.reset_index()
age_join


# - Visualization:

# In[ ]:


x = sns.lineplot(x = 'Pclass', y = 'Age', hue = 'Sex', data = mean_age).set_title('Mean age by Pclass')
plt.show()


# In[ ]:


x = sns.lineplot(x = 'Pclass', y = 'Age_nan_index', hue = 'Sex', data = mean_age_x).set_title('Mean age by Pclass for interpolated values')
plt.show()


# In[ ]:


x = sns.lineplot(x = 'Pclass', y = 'Age_nan_index', hue = 'Survived', data = age_survival_female).set_title('Survival mean age for women by Pclass')
plt.show()


# In[ ]:


x = sns.lineplot(x = 'Pclass', y = 'Age_nan_index', hue = 'Survived', data = age_survival_male).set_title('Survival mean age for men')
plt.show()


# In[ ]:


new_data['age_groups'] = pd.qcut(new_data.loc[:,'Age'], [0, .25, .5, .75, 1], labels = ['0 - 20','21 - 28','29 - 38','39 - 80'])
k = sns.catplot(x= 'age_groups', y = 'Survived', data = new_data, hue='Pclass', kind='bar')
plt.show()


# In[ ]:


x = sns.boxplot(x= 'Sex', y = 'Age', data = new_data, notch = True).set_title('Age by sex')
plt.show()


# In[ ]:


x = sns.boxplot(x= 'Sex', y = 'Age_nan_index', data = new_data, notch = True).set_title('Age by sex for interpolated values')
plt.show()


# In[ ]:


y = sns.swarmplot(x= 'Sex', y = 'Age_nan_index', data = new_data, color = '.25')
plt.show()


# Center_Male = 29 years
# 
# Center_Female = 27 years
# 
# Some outlier ages for men between 68 and 80 years
# 
# Spread: 
#  - Inter-quatile range of (age): men - (21 to 39 years) and women - (18 - 37 years) 
# 
# Mean age should be slightly above 29 for men ~ 31 years and ~ 28 years for women

# In[ ]:


new_data['Survived_x'] = new_data['Survived'].map({0:'Died',1:'Survived'})
y = sns.swarmplot(x= 'Sex', y = 'Age_nan_index', data = new_data, hue='Survived_x', color='0.5')
plt.show()


# In[ ]:


x = sns.boxplot(x= 'Pclass', y = 'Age_nan_index',hue = 'Sex', data = new_data).set_title('min, max age by class (age range by class)')
plt.show()


# In[ ]:


x = sns.FacetGrid(new_data[new_data['title'].isin(['Mr.','Master.','Mrs.','Miss.'])], row='Sex', col = 'title')
x = x.map(plt.hist, 'Survived')
plt.show()


# C. Family groups:

# 1. where men or women with siblings/Spouses onboard more or less likely to survive?

# In[ ]:


k = sns.catplot(x= 'Sex', y = 'SibSp', data = new_data, hue='Survived', kind='bar')
plt.show()


# 2. where men or women with parents/children onboard more or less likely to survive?

# In[ ]:


k = sns.catplot(x= 'Sex', y = 'Parch', data = new_data, hue='Survived', kind='bar')
plt.show()


# D.  Fare

# In[ ]:


k = sns.catplot(x= 'Pclass', y = 'Fare_x', data = new_data, hue='Sex', kind='bar')
plt.show()


# Fare was higher for women of all classes especially high for first class women.

# In[ ]:


k = sns.catplot(x= 'Embarked', y = 'Fare_x', data = new_data, hue='Sex', kind='bar')
plt.show()


# Fare was higher for people who embarked @ Cherbourg

# E. Class of Travel

# How were survived men and women distributed across class of travel?

# In[ ]:


k = sns.barplot(x= 'Pclass', y = 'Survived', data = new_data, hue='Sex')
plt.show()


# What was the distribution of age across Pclass?

# What was the ratio of age across Pclass?

# What was the ratio of age across Pclass

# In[ ]:


k = sns.barplot(x= 'Pclass', y = 'Survived', data = new_data, hue='age_groups')
plt.show()

