#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# **------(NOTE)---------**
# 
# *This notebook is not yet finished, as I will update this kernel further*
# 
# **---------------------**
# 
# In this notebook, we will explore the Titanic dataset where we want to find out what were the factors of survining and dying during ship sinking.
# 
# We will do the following steps (**step** - done step):
# 
# 0. **Retrieve the data**
# 1. Explore the data
# 2. Preprocess the data
# 3. Model the data
# 4. Train and test the model
# 5. Evalutaion and presentation
# 
# We will cover all those steps using Python with its packages numpy, pandas, matplotlib and scikit-learn, maybe even use the PyTorch module later on.

# ## Step 0: Retrieve the data and load
# 
# We will retrieve Titanic Dataset from the Kaggle website https://www.kaggle.com/c/titanic, where we will use the simple API provided by the Kaggle that we will retrieve and read the data.
# 
# Assuming we have downloaded the necessary data, it is time to load the data into the pandas dataframe. We will also include the packages right away that are needed for data exploration step.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'notebook')

df = pd.read_csv('../input/train.csv')


# Now that we have loaded the data, we will print the first 10 rows of data to see what we're dealing with. Also, we will test if there are any duplicate rows in the data.

# In[ ]:


df[:10]


# In[ ]:



len(df)


# In[ ]:


df.duplicated().value_counts()


# In[ ]:


df['PassengerId'].is_unique


# As we can see, there are no duplicate rows nor duplicate PassengerId, which means that each row is unique on its own and we don't need to disregard any data. We will test for each column if we don't have any data regarding to it in the next step.

# ## Step 1: Explore the data
# 
# We will do some explorations, analytics and describe some events regarding this dataset. We will go now column by column, describing the data in the proces. First, we want to see how many survivors do we have in the dataset.
# 
# We will do the following analysis of how much impact next columns has on survival rate of passengers:
# 
# - Gender
# - Age group
# - Siblings/Spouse
# - Parents/Children
# - PClass
# - Fare
# - Cabin
# 
# I believe that Name and Ticket are columns that have no impact on survival rate of each passenger. I don't think that certains names don't raise or lowers the chances of surviving a disaster, as well as tickets, since they are, in some ways, another ways of identification. Also, the Port of Embarkation also doesn't have an impact, since I believe that the location where passengers have entered the ship don't have that much significance. Even if we can find the correlation, for example, people who ported in Queenstown have higher chances of surviving, but that doesn't guarantee us for the next ship how will it behave. Instead, we want to analyze the traits of human being and its behaviour and location in the ship: How old is the passenger, does he/she have siblings, parents, children, where is he/she located in the ship, what gender is the passenger, cabin location and more.
# 
# Of course, every next analytics will combine with previous, if needed, for further proving hypothesis'. I will list all analytics in this section and I will provide explanations why I will include or not include certain columns in modeling the data for predicting survival rate of passengers.
# 
# For now, I have decided on next columns that will go in the preprocessing and modeling part of the project:
# 
# - Sex
# - Age (engineered in age group)

# ### Gender analysis
# 
# We want to see if the passengers are uniformly distributed or if there is some kind of correlation with the likelihood of surviving the accident. We will begin our research by seeking how many survivors do we, actually, have.

# In[ ]:


survival = {0: 'Not Survived', 1: 'Survived'}

df['Survived'].value_counts()


# Let's normalize the input and calculate the proportion of those passengers who died and who survived.

# In[ ]:


prop_survived = df['Survived'].value_counts() / len(df)


# In[ ]:


prop_survived = prop_survived.rename(survival)
print(prop_survived)

prop_survived.plot(kind='bar', title='Survival rate')


# As we can see, we have around 61% who died during the Titanic sinking and around 38% survivors in our dataset.
# 
# Our next step in our analytics is to measure the proportion of male and female passangers.

# In[ ]:


gender_data = df['Sex']

gender_data[:10]


# In[ ]:


# We want to see whether we have NaN values for our genders
gender_data.isna().value_counts()


# In[ ]:


prop_gender = gender_data.value_counts() / len(gender_data)


# In[ ]:


print(prop_gender)


# We have around 64% male and around 35% female passengers. Now we want to find out how many males and how many females survived in the sinking of Titanic.

# In[ ]:


male_female_survived = df.groupby(by=['Sex', 'Survived']).size().reset_index().rename(columns={0: 'Count'})

male_female_survived


# In[ ]:


import seaborn as sns

sns.barplot(x='Sex', y='Count', hue='Survived', data=male_female_survived).set_title('Survival rate by Gender')


# In[ ]:


# now we calculate the percentage
prop_male_female_survived = male_female_survived.copy()

prop_male_female_survived['Count'] = prop_male_female_survived['Count'] / prop_male_female_survived['Count'].sum()
prop_male_female_survived.rename(columns={'Count': 'Percentage'}, inplace=True)

print(prop_male_female_survived)


# We can see that the majority of population that have not survived during the Titanic sinking are males, which means that if a person is a male, according to the analysis, the chances of surviving is minimum.
# 
# We will test out how many males have survived, then the females.

# In[ ]:


male_survived = df[df['Sex'] == 'male']

len(male_survived)


# In[ ]:


male_survived['Survived'].value_counts().rename(survival).plot(kind='bar', title='Survival rate of Male Passengers')


# In[ ]:


prop_male_survive = male_survived['Survived'].value_counts().rename(survival) / len(male_survived)
prop_male_survive


# Now we can see, looking only at the males, that we have over 80% casualties. Let's look at the female passengers.

# In[ ]:


female_survived = df[df['Sex'] == 'female']

len(female_survived)


# In[ ]:


female_survived['Survived'].value_counts().rename(survival).plot(kind='bar', title='Survival rate of Female Passengers')


# In[ ]:


prop_female_survive = female_survived['Survived'].value_counts().rename(survival) / len(female_survived)
prop_female_survive


# In contrast, looking at the females, we have just under 75% female passengers that have survived the disaster, meaning that the sex plays a significant role whether a passenger will survive or not.

# ### Age group analysis
# 
# As we have seen, the gender plays a big role of determining the chance of surviving favouring female passengers.
# 
# We will divide age groups with the following:
# 0. minimum_age - 14
# 1. 15 - 24
# 2. 25 - 40
# 3. 41 - 65
# 4. 65 - max_age
# 
# Not every passenger has an age provided, so we will calculate how many passengers do not have the age provided.

# In[ ]:


# dividing into age groups
df['Age'].isna().value_counts()


# In[ ]:


df['Age'].describe()


# We can see that there are a somewhat significant portion of passengers with no provided age, so we will calculate the mean of other ages. We also see that the age is not consistent, so we will try to group them in age groups, as specified before

# In[ ]:


from math import modf

# We want the integer part of the mean to take for fillna()
# We also will not change the df, so we will save it to df_modified

# For now, we will assume they are all in the mean
# Note: Include deviation part?
df_modified = df[['PassengerId', 'Survived', 'Sex', 'Age']].copy()

df_modified['Age'] = df_modified['Age'].fillna(modf(df['Age'][df['Age'].notna()].mean())[1])


# In[ ]:


df_modified[-10:]


# Now that we have filled the gaps in the missing age data, we will now add a new column that will add to the age group. We will define a function that will assign each passenger to it's age group. We have seen that the oldest passenger was 80 years old at the time, so that will make the last age group 65-81.

# In[ ]:


# 0 - [0 -15)
# 1 - [15 - 25)
# 2 - [25 - 40)
# 3 - [40 - 65)
# 4 - [65 - 81)

# Age groups here are left_inclusive

# NOTE: Check to do it with pd.Categorical type

age_groups = {0: (0, 15), 1: (15, 25), 2: (25, 40), 3: (40, 65), 4: (65, 81)}

def which_age_group(x):
    for key, age_group in age_groups.items():
        if x >= age_group[0] and x < age_group[1]:
            return key

df_modified['AgeGroup'] = df_modified['Age'].apply(which_age_group).astype('int64')


# In[ ]:


df_modified[:10]


# Now that we have assigned to each passenger it's AgeGroup, it is time to do some analytics and draw some conclusions. Let see how many have passengers we have for each age group.

# In[ ]:


print(df_modified.groupby('AgeGroup').size().rename(age_groups))

df_modified.groupby('AgeGroup').size().rename(age_groups).plot(kind='bar', title='Number of passengers by Age Group')


# It is remarkably similar to the normal distribution of age groups in the population of given passengers. Let's see how many have survived for each age group and calculate their percentages.

# In[ ]:


# looking at the whole population

age_group_survive = df_modified.groupby(['AgeGroup', 'Survived']).size().reset_index().rename(columns={0: 'Count'})


# In[ ]:


# To get sense of which age group we are talking about, not looking at indices

age_group_survive['AgeGroup'] = age_group_survive['AgeGroup'].apply(lambda x: age_groups[x])


# In[ ]:


age_group_survive


# In[ ]:


age_group_survive['Survived'] = age_group_survive['Survived'].apply(lambda x: survival[x])


# In[ ]:


age_group_survive


# In[ ]:


sns.barplot(x='AgeGroup', y='Count', hue='Survived', data=age_group_survive).set_title('Survival rate by Age Group')


# We can see that there were a lot of casualties in every age group except the children age group (0, 15), which is not surprising. We will extend these analytics by providing the sex as a key.
# 
# First, we will see how many passengers of different sex do we have in each age group

# In[ ]:


age_group_by_sex = df_modified.groupby(['AgeGroup', 'Sex', 'Survived']).size().reset_index().rename(columns={0: 'Count'})


# In[ ]:


age_group_by_sex['AgeGroup'] = age_group_by_sex['AgeGroup'].apply(lambda x: age_groups[x])


# In[ ]:


age_group_by_sex


# In[ ]:


age_group_by_sex['Survived'] = age_group_by_sex['Survived'].apply(lambda x: survival[x])


# In[ ]:


# NOTE: Check how to plot with combining two columns without the need to create new one

age_group_by_sex['AgeGroupSex'] = age_group_by_sex['AgeGroup'].apply(lambda x: str(x)) + " - " + age_group_by_sex['Sex']


# In[ ]:


age_group_by_sex


# In[ ]:


# plotting

a4_dims = (10, 9)
fig, ax = plt.subplots(2, figsize=a4_dims)

survival_by_age_group_males = sns.barplot(ax=ax[0],
                                            x='AgeGroupSex',
                                            y='Count',
                                            hue='Survived',
                                            data=age_group_by_sex[age_group_by_sex['Sex'] == 'male'])

survival_by_age_group_males.set_title('Survival rate by sex and age group - males')

survival_by_age_group_females = sns.barplot(ax=ax[1],
                                            x='AgeGroupSex',
                                            y='Count',
                                            hue='Survived',
                                            data=age_group_by_sex[age_group_by_sex['Sex'] == 'female'])

survival_by_age_group_females.set_title('Survival rate by sex and age group - females')


# We can conclude here that age group also plays a role on providing a chance of survival to certain groups of people. That means that age group as category can tell us more about the significance of surviving.
# 
# We can conclude that mostly the children and females were saved, further proving the significant role a gender has on survival rate.

# ### Sibling - spouses analysis

# Further analysis and feature engineering required for sibling-spouses analytics. The goal is to test if there is any correlation to minimizing the chance of surviving if a passenger has a sibling or spouse.
# 
# We will describe the series to see what we're dealing with.

# In[ ]:


df.head()


# In[ ]:


df['SibSp'].describe()


# We see that, if someone has a sibling or spouse, it will have tipically 1 or 2, let's see if we can count how many of them we have.

# In[ ]:


df['SibSp'].value_counts()


# As suspected, tipically it will have 1 or 2, with small portion of them having more.
# 
# We can use the feature engineering again to convert the SibSp into a boolean, since a small portion of them would not make a significant change in the model.

# In[ ]:


df_siblings = df[['PassengerId', 'Survived', 'Sex', 'SibSp', 'Age']].copy()


# In[ ]:


df_siblings[:6]


# In[ ]:


df_siblings['SibSp'].notna().value_counts()


# We will also use the feature engineering for the age group we have done before, so that we can provide further analysis on the relation siblings/spouse with the certain age group and certain gender.
# 
# Also, the column SibSp has no NaN values, so we don't need to do any missing data procedures.

# In[ ]:


df_siblings['Age'] = df_siblings['Age'].fillna(modf(df['Age'][df['Age'].notna()].mean())[1])
df_siblings['AgeGroup'] = df_siblings['Age'].apply(which_age_group).astype('int64').apply(lambda x: age_groups[x])
df_siblings['HasSibSp'] = df_siblings['SibSp'].apply(lambda x: x > 0)


# In[ ]:


df_siblings[:6]


# In[ ]:


df_siblings['HasSibSp'].value_counts()


# Now that we have the columns HasSibSp, we can see now how many casualties were there regarding the siblings.

# In[ ]:


# using the whole population
hassib_survival_rate = df_siblings.groupby(['HasSibSp', 'Survived']).size().reset_index().rename(columns={0: 'Count'})


# In[ ]:


hassib_survival_rate


# In[ ]:


hassib_survival_rate['Survived'] = hassib_survival_rate['Survived'].apply(lambda x: survival[x])


# In[ ]:


sns.barplot(x='HasSibSp', y='Count', hue='Survived', data=hassib_survival_rate).set_title('Survival rate by having siblings or spouse')


# We can see that there are no drastic changes in comparison. We need to prove that having a sibling will not change the outcome of survival. We will check for sibling/spouse for male and female from the sample of population who had siblings or spouse.

# In[ ]:


passengers_with_siblings = df_siblings[df_siblings['HasSibSp']]
passengers_with_no_siblings = df_siblings[df_siblings['HasSibSp'] == False]


# In[ ]:


survival_rate_by_sex_and_sibsp = passengers_with_siblings.groupby(['Sex', 'Survived']).size().reset_index().rename(columns={0: 'Count'})
survival_rate_by_sex_and_nosibsp = passengers_with_no_siblings.groupby(['Sex', 'Survived']).size().reset_index().rename(columns={0: 'Count'})


# In[ ]:


survival_rate_by_sex_and_sibsp


# In[ ]:


survival_rate_by_sex_and_nosibsp


# In[ ]:


survival_rate_by_sex_and_sibsp['Survived'] = survival_rate_by_sex_and_sibsp['Survived'].apply(lambda x: survival[x])

survival_rate_by_sex_and_sibsp
# siblings_gender_plot = sns.barplot(x='Sex', y='Count', hue='Survived', data=survival_rate_by_sex_and_sibsp)

# siblings_gender_plot.set_title('Survival rate of having siblings by sex ')

# siblings_gender_plot


# In[ ]:


survival_rate_by_sex_and_nosibsp['Survived'] = survival_rate_by_sex_and_nosibsp['Survived'].apply(lambda x: survival[x])

survival_rate_by_sex_and_nosibsp


# In[ ]:


fig, ax = plt.subplots(2, figsize=a4_dims)

survival_has_sib = sns.barplot(ax=ax[0], x='Sex', y='Count', hue='Survived', data=survival_rate_by_sex_and_sibsp)
survival_has_sib.set_title('Survival rate of having siblings by sex')

survival_has_nosib = sns.barplot(ax=ax[1], x='Sex', y='Count', hue='Survived', data=survival_rate_by_sex_and_nosibsp)
survival_has_nosib.set_title('Survival rate of not having siblings by sex')


# We don't see some drastic changes in comparison, as well, by looking at the siblings/spouse, and with calculating percentages

# In[ ]:


survival_rate_by_sex_and_sibsp['Count'] = survival_rate_by_sex_and_sibsp['Count'] / survival_rate_by_sex_and_sibsp['Count'].sum()


# In[ ]:


survival_rate_by_sex_and_nosibsp['Count'] = survival_rate_by_sex_and_nosibsp['Count'] / survival_rate_by_sex_and_nosibsp['Count'].sum()


# In[ ]:


survival_rate_by_sex_and_nosibsp


# In[ ]:


# NOTE: Research impact of sibling/spouse on age groups

survival_rate_by_sex_and_sibsp


# There are no strong impacts of having a sibling. I believe that because there were no more casualties in the has_siblings dataframe, the proportion of them both would tend to some constant *k*, so in the final model, we won't include the sibling/spouse part (bias posibility, unlikely I believe).
# 
# Next, we will do parent/child part of analytics. I believe it will have much more impact than sibling/spouse part.

# ## TODO: Parents/Child analytics

# In[ ]:




