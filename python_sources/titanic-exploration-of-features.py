#!/usr/bin/env python
# coding: utf-8

# This notebook explores the [Kaggle Titanic](https://www.kaggle.com/c/titanic) disaster survival dataset and evaluates which features may be most useful in a predictive model of survival.
# 
# The test dataset test.csv is maintained as a hold-out set for final validation and is not used for any other purpose, including filling missing values in the training data.

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns


# # Basic data exploration

# In[ ]:


# Read in both datasets
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# Print out some basic information about the datasets before we focus
# entirely on the training dataset
print(f'Training data has {train_data.shape[0]} rows, {train_data.shape[1]} columns')
print(f'Test data has {test_data.shape[0]} rows, {test_data.shape[1]} columns')

cols_both = [col for col in train_data.columns if col in test_data.columns]
cols_train_only = [col for col in train_data.columns if col not in test_data.columns]
cols_test_only = [col for col in test_data.columns if col not in train_data.columns]

print(f'Both datasets have columns {cols_both}') if len(cols_both) else None
print(f'Training has columns {cols_train_only}, test does not') if len(cols_train_only) else None
print(f'Test has columns {cols_test_only}, train does not') if len(cols_test_only) else None


# ### Exploratory analysis of the training data
# 
# Now that we know the basic structure of the training and test data, we'll look at the training data to understand what cleanup is needed and what feature engineering might make sense.

# In[ ]:


train_data.profile_report(title='Training data: initial profiling report')


# In[ ]:


train_data.head()


# # Survival with embarkation, Pclass, and gender

# In[ ]:


def explore_generalsurvival(df_train, min_samples=10):
    """Show mean survival for data grouped by gender, embarkation point,
    and Pclass.  Only show subsets with at least <min_samples> samples.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=[14, 5])
    for gender, ax in zip(['male', 'female'], axes):
        ind = (train_data['Sex'] == gender)
        mean_surv = df_train[['Embarked', 'Pclass', 'Survived', 'Sex']][ind]
        mean_surv_vals = (mean_surv.groupby(['Embarked', 'Pclass'])['Survived']
                                   .mean()
                                   .unstack('Pclass'))
        mask = (mean_surv.groupby(['Embarked', 'Pclass'])['Survived']
                         .count()
                         .unstack('Pclass'))
        mask = mask < min_samples
        sns.heatmap(mean_surv_vals, ax=ax, annot=mean_surv_vals, mask=mask,
                    cbar_kws={'label': 'Mean survival'})
        ax.set_title(gender)
        
explore_generalsurvival(train_data)


# Figure: In all categories shown, survivorship was higher for females than males.  For males (left), first class passengers tend to have higher survivorship than second or third class passengers.  Males in third class who embarked at C had higher survivorship than passengers who embarked at Q or S.  For females (right), first and second class passengers had very high survivorship regardless of their embarkation point.  For female third class passengers, survivorship was higher for passengers who embarked at Q or C than at S.
# 
# ** Feature implications: gender and Pclass are clearly important features; embarkation point appears less consequential, but may also be worth including**.

# In[ ]:


def explore_familysamplesize(df, sibsp='SibSp', parch='Parch'):
    """Show the sample size once we group by gender, Parch, and SibSp.  No cases are
    masked in this plot.  sibsp and parch variable names are passed in as we use the
    same function before and after we collapse classes.
    """
    fig, axes = plt.subplots(1, 2, figsize=[15, 5])
    for gender, ax in zip(['male', 'female'], axes):
        mean_surv = df[[sibsp, parch, 'Survived', 'Sex']][train_data['Sex'] == gender]
        mean_surv_cnts = (mean_surv.groupby([sibsp, parch])['Survived']
                                   .count()
                                   .unstack(sibsp))
        sns.heatmap(mean_surv_cnts, ax=ax, annot=mean_surv_cnts,
                    cbar_kws={'label': 'Number of cases'})
        ax.set_title(gender)
        
explore_familysamplesize(train_data)


# Figure: By far, most passengers are travelling alone (410 males and 130 females).  The next most common category is to have SibSp=1 (60 males, 63 females); presumably many of these are married couples without children.  There are also reasonable numbers of passengers, especially females, travelling with Parch=1 or 2.  Other classes are too small to provide robust model training.
# 
# **Feature implication: We should collapse SibSp and Parch into a smaller number of better-balanced categories.**

# In[ ]:


def explore_familysurvival(df, min_samples=6, sibsp='SibSp', parch='Parch'):
    """Show mean survival for different parent child and sibling classes, masking out
    any classes with less than <min_samples> samples.  The variable name is passed in
    for sibsp and parch because we use the same function to plot before and after we
    collapse classes."""
    
    fig, axes = plt.subplots(1, 2, figsize=[15, 5])
    for gender, ax in zip(['male', 'female'], axes):
        mean_surv = df[[sibsp, parch, 'Survived', 'Sex']][train_data['Sex'] == gender]
        mean_surv_vals = (mean_surv.groupby([sibsp, parch])['Survived']
                                   .mean()
                                   .unstack(sibsp))
        mask = (mean_surv.groupby([sibsp, parch])['Survived']
                         .count()
                         .unstack(sibsp))
        mask = mask < min_samples
        sns.heatmap(mean_surv_vals, ax=ax, annot=mean_surv_vals, mask=mask,
                    cbar_kws={'label': 'Mean survival'})
        ax.set_title(gender)
        
explore_familysurvival(train_data)


# Figure: Average female survivorship varies only somewhat with the number of family connections, with average ratings ranging from 73% to 88%.  Average male survivorship for males travelling alone is 16%, but is substantially higher for males who have at least one family member travelling with them.  Male survivorship is especially elevated for males who have at least two family members travelling with them -- e.g. males with Parch=1 and SibSp=1 have 45% survivorship and males with Parch=2 have 56% survival if SibSp=0, and 50% if SibSp=1.
# 
# ** Feature implication: Parch and SibSp may be useful features, especially for males.  Given that the effect of
# Parch is not the same as SibSp, adding them together into a family size feature may cause loss of useful signal.**

# In[ ]:


def add_binary_family_features(df_train):
    """Create new variables has_parch and has_sibsp where values of Parch and SibSp
    greater than 1 are collapsed to 1.
    """

    new_names = {'Parch': 'has_parch',
                 'SibSp': 'has_sibsp'}
    for var in ['Parch', 'SibSp']:
        new_var = df_train[var].copy(deep=True)
        new_var[new_var > 1] = 1
        df_train[new_names[var]] = new_var
    return df_train

train_data = add_binary_family_features(train_data)


# In[ ]:


explore_familysamplesize(train_data, sibsp='has_sibsp', parch='has_parch')


# In[ ]:


explore_familysurvival(train_data, min_samples=6, sibsp='has_sibsp', parch='has_parch')


# # Survival with age and other factors

# In[ ]:


g = sns.catplot(x="Pclass", y="Age", hue="Survived", col="Sex",
                data=train_data[['Pclass', 'Survived', 'Sex', 'Age']],
                kind="violin", split=True, cut=0, scale='count')


# Figure: Distribution of ages for males (left plot) and females (right plot), those who survived (orange) and those who didn't (blue), and Pclass (x-axis).  Violinplot areas have been scaled by the counts of data within each group, and the distributions have been rigidly cut at the min and max values for each distribution.
# 
# Males: In first and second class, younger males tended to have a higher survivorship than older ones; indeed in those classes it looks like all male children survived.  For third class males, age does not appear to have an impact on survival.
# 
# Females: In first and second class we again see that females nearly all survived; in third class, age  doesn't seem to be a strong predictor of survival, similar to the behavior we saw for males.
# 
# ** Feature implications: Age looks like a useful feature, especially for males.  If ages are binned into a smaller number of categories, it will be important to maintain a child-specific age range (possibly 0 to 15).**

# # Passenger with fares of 0

# In[ ]:


train_data[train_data['Fare'] == 0]


# In[ ]:


train_data[train_data['Fare'] == 0].groupby('Pclass')['Fare'].count()


# Fare handling: Passengers who paid zero in fare are all male, none survived except one,
# all embarked at S, and none of them had any siblings, parents, or children on board.  They are relatively evenly distributed across Pclass categories, and range in age from 19 to 49.  There's nothing obvious here to tell us how to manage the zero fares, so we elect to leave them as is.

# # Survival with Fare and Pclass

# In[ ]:


# First we look at the distribution in fares for all three classes
for pclass in [1, 2, 3]:
    df_ind = (train_data['Pclass'] == pclass)
    sns.distplot(train_data['Fare'][df_ind], hist=False, label=pclass)


# In[ ]:


# Next we see if removing the passengers who paid zero and looking only at
# the low end of the distribution clarifies
df_sub = train_data[train_data['Fare'] != 0]
fig, axes = plt.subplots(nrows=3, figsize=(7, 7), sharex=True)
for ind, pclass in enumerate([1, 2, 3]):
    df_ind = (df_sub['Pclass'] == pclass)
    sns.distplot(df_sub['Fare'][df_ind], hist=True, 
                 bins=np.arange(0, 110, 2), label=pclass,
                 ax=axes[ind], )
    _ = axes[ind].set_xlim([-10, 100])
    axes[ind].set_ylabel(f'pclass={pclass}')


# Figure: Looks like there is some overlap in what fare was paid between all three classes.  First class has a much longer tail.  Removing the passengers who paid zero in fare doesn't make much difference in the appearance of the distributions.

# In[ ]:


g = sns.catplot(x="Pclass", y="Fare", hue="Survived", col="Sex",
                data=train_data[['Fare', 'Pclass', 'Survived', 'Sex']],
                kind="violin", split=True)


# Figure: In general, there's not an obvious relationship between Fare paid and survivorship for most Pclass categories for either gender.   The strongest relationship that appears in this view is that first class female passengers who paid more appear to be less likely to have survived.
# 
# It's also interesting that the separation in the distribution of fares paid by different Pclass categories seems stronger for females than for males.

# # Summary of findings
# 
# The following table provides key information about each column, some of which originated in the above profile report.  These thoughts will guide our data cleanup and feature engineering.
# 
# |Column name | Description | Proposed plan|
# |---|---|---|
# |**PassengerId** | Numeric values are 100% unique and vary from 1 to 891.| Use as **index** column. |
# |**Pclass** | Passengers are each assigned a numeric class: first (1), second (2), or third (3) class.  There are no missing values.  Allocation to classes is 55% in 3rd, 21% in 2nd, and 24% in 1st.| **Use as a feature**, could be a powerful proxy for socio-economic class and/or cabin location.  Watch out for class imbalance.|
# |**Name** | Passenger names, 100% unique.  One could consider some derived features based on names. | **Not promising** as a first area of focus, ignore for now. |
# |**Sex** | Categorical variable with values 'male' and 'female'.  There are no missing values.  Allocation to genders is 65% male and 35% female. | **Transform to boolean column is_male and use as feature**.  Watch out for class imbalance. |
# | **Age** | Numeric values varying from 0.42 (a baby under one year old) to 80 (a senior).  There are 177 missing values (20% missing). | **Fill missing values** and **test as a feature** given potential that age could be a strong predictor of survival. |
# |**Ticket** | Categorial variable, no missing values, 76% of ticket numbers are unique.  Ticket numbers vary from 3 to 18 characters in length with a mean length of about 7.  Values can include characters and digits (e.g. 'CA 2144'), just digits (e.g. '382652'), and probably many other forms. | **Not promising** as a first area of focus, ignore for now. |
# |**Fare** | Numeric value from 0 to 512 with a mean of 32; no missing values, though 15 values are zero.  Stronger density at low fares, long tail at higher fares (exponential distribution appearance). Could be a potential proxy for socio-economic class (along with Pclass) or proximity to exits (assuming perhaps lower socio-economic class passengers might be placed at lower decks / further from exits).| **1. Investigate 15 passengers with zero fares and 2. Test fare as a feature**. |
# |**Cabin** | Categorical variable, 77% of cabin numbers are missing and there are 148 unique cabin numbers amongst the remainder.  Value is usually a letter followed by a number (e.g. 'E101', 'G6', 'B22'), though sometimes two or more cabins are listed for a given passenger (e.g. 'B96 B98', 'C23 C25 C27').  One might glean something about the passenger survivorship from this data (e.g. the deck number as an indicator of proximity to exits / life boats). | Given the large number of missing values, ** not promising** as a first area of focus, ignore for now. |
# |**Embarked** | Categorical variable giving the port of embarkation ('C' for Cherbourg, 'Q' for Queenstown, 'S' for Southampton).  72% of values are 'S', 19% are 'C', 9% are 'Q', and 0.2% are missing (2 missing values). | Worth **evaluating as a feature**, after filling missing values with S (the most common value).  Watch out for class imbalance. |
# |**SibSp** | Numeric value giving the number of siblings and/or spouses aboard the Titanic. Category includes brother, sister, stepbrother, stepsister, husband, and/or wife.  None are missing, mean value is 0.52, and values range from 0 to 8.  68% of passengers have 0, 24% have 1, 3% have 2, and decreasing numbers have from 3 to 8. | **Reduce to binary has_sibsp variable and test as a feature**, be aware of poor class balance even after reducing categories. |
# |**Parch** | Numeric value giving the number of parents and/or children aboard.  Category includes mother, father, daughter, son, stepdaughter, and stepson.  0% are missing, mean value is 0.38, and values range from 0 to 6.  76% have 0, 13% have 1, 9% have 2, 0.6% have 3, and decreasing numbers have from 4 to 6. | **Reduce to binary has_parch variable and test as a feature**, be aware of poor class balance even after reducing categories.|
