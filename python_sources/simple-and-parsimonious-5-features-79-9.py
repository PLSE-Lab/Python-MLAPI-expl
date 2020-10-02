#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Having gone over multiple kernels and read other statisticians' takes (outside of Kaggle) on the Titanic dataset, I wanted to make a model that is as simple as possible while still keeping prediction accuracy (leaderboard, etc.) in mind.
# 
# Most of work for this kernel was performed while on a quest for the coveted 80%, without use of clear overfitting to the leaderboard, or ensemble-based models.
# 
# ### Goals:
# 
# * Remove "noise" features that add little to no predictive value.
# * Reduce collinearity by removing as many "overlaps" between features as possible.
# * Focus on model simplicity and interpretability.
# * Avoid "fitting to the leaderboard" while still maintaining a good score.
# 
# ### Table of Contents:
# 
# * [Exploratory Analysis](#Exploratory-Analysis)
#     * [Import](#Import)
#     * [Rich Ladies First!](#Rich-Ladies-First!)
#     * [What's in a Name?](#What's-in-a-Name?)
#     * [Age Is but a Number](#Age-Is-but-a-Number)
#     * [Thicker Than Water](#Thicker-Than-Water)
#     * [All Hands on Deck!](#All-Hands-on-Deck!)
#     * [Fare Thee Well](#Fare-Thee-Well)
#     * [All That Matters Is Where You're Going](#All-That-Matters-Is-Where-You're-Going)
# * [Model Selection and Validation](#Model-Selection-and-Validation)
# * [Afterword](#Afterword)
# * [References](#References)

# ## Exploratory Analysis
# 
# Much of the analysis is sourced from, or motivated by, the excellent kernels which I *highly recommend* working through.
# 
# * https://www.kaggle.com/headsortails/pytanic
# * https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic
# * https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# * https://www.kaggle.com/pliptor/divide-and-conquer-0-82297
# 
# ### Import
# After importing the data, we create a combined DataFrame containing both the train and test sets. This allows us to perform feature transformation and/or selection in a single line.
# 
# During the exploratory analysis, we will be mutating both `train` and `df`, since we will be visualizing data and computing statistics on both DataFrames. However, if you are performing feature transformations for a final model, you will only need to mutate `df`.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
import numpy as np

train = pd.read_csv('../input/train.csv', header=0, index_col='PassengerId')
test = pd.read_csv('../input/test.csv', header=0, index_col='PassengerId')

# take out the 'Survived' column from the training data
X_train = train.drop('Survived', axis=1)

# length of the training data; used to recombine the sets
tr_len = len(X_train)

# combined data
df = X_train.append(test)


# ### Rich Ladies First!
# 
# As seen in the graph below (and as you have probably heard over twenty times, by now), being a woman dramatically increases odds of survival. This is due to the (in)famous '[Women and Children First](https://en.wikipedia.org/wiki/Women_and_children_first)' rule used when boarding passengers onto lifeboats. Call it fair, call it not, but using this rule alone nets you a 0.76555 on the leaderboard, serving as a baseline model.
# 
# The other main factor is passenger class. Though a strong predictor, it turns out passenger class is nowhere near as strong as gender:

# In[ ]:


plt.figure(1, figsize=(8, 4))
plt.subplot(121)
sns.barplot(x='Sex', y='Survived', data=train)
plt.subplot(122)
sns.barplot(x='Pclass', y='Survived', data=train)
plt.show()


# Henceforth, we will refer to this rule as 'RWC': When we explore most of the other variables, we will first sort by gender and passenger class.

# ### What's in a Name?
# First up, the `Name` feature. Since names are fairly unique, any analysis using them (or even last name alone) is likely to result in huge overfitting. However, looking at the first few names...

# In[ ]:


df['Name'].head()


# ... we see that they all seem to have a format of "(Last Name), (Title). (First / Middle Names)" - also, some married women may be referred to by their husband's name, with their actual name in parentheses after.
# 
# The simplest and "standard" trick is to extract only the Title, since this provides a good (and great, as we will see later) way to bin the passengers. We can do this with a single line of code:

# In[ ]:


df['Title'] = df['Name'].str.extract('\,\s(.*?)[.]', expand=False)
train['Title'] = df.loc[:tr_len, 'Title']


# For those who aren't familiar with regular expression, there is a great guide on [Wikipedia](https://en.wikipedia.org/wiki/Regular_expression). Notice you need the ? in there, as there is a single name with a second dot (.) character following the dot after the title.

# In[ ]:


print('Unique titles in the training set only:\n{}\n'.format(train['Title'].unique()))
print('Unique titles in both sets:\n{}'.format(df['Title'].unique()))


# The title 'Dona' is found only in the test set, suggesting we might want to combine some titles. To further motivate this, let's take a look at the frequencies of each title, by gender:

# In[ ]:


train_f = train[train['Sex'] == 'female']
train_m = train[train['Sex'] == 'male']
plt.figure(2, figsize=(16, 6))
plt.subplot(121)
sns.countplot(train_f['Title'])
plt.subplot(122)
sns.countplot(train_m['Title'])
plt.show()


# The first thing we can do is obvious: 'Mme' is simply French for 'Mrs' while 'Mlle' is French for 'Miss'.
# 
# Also, since the 'Rare' titles are a good predictor of survival for females, we can combine them all into a 'FRare' title.
# 
# Finally, there is one last female title: 'Ms'. This can be either 'Mrs' or 'Miss', so let's take a look at the data:
# 
# 

# In[ ]:


df[df['Title'] == 'Ms']


# Ms. Reynaldo is 28 and traveling alone in second class, while Ms. O'Donoghue has an unknown age and is traveling alone in third class. Neither have a husband listed. As Ms. Reynaldo is somewhat young, and Ms. O'Donoghue is alone in third class (suggesting youth), we will group them both in with 'Miss'.
# 
# For the male titles, things are more interesting. There are several approaches:
# 
# * Combine all 'Rare' titles into 'MRare'
# * Separate by category: 'Noble' for 'Don', 'Sir' and 'Jonkheer'; 'Mil' for 'Major', 'Col' and 'Capt'.
# * Separate out 'Rev'. Why? Perhaps men who are reverends are more likely to be selfless in this time of crisis.
# 
# With so many approaches, let us start with the following:
# 
# 1. Combine 'Mme' and 'Mlle' into the English equivalents, and bin the rare female titles. Combine 'Ms' into 'Miss'.
# 2. Bin the military titles and the male noble titles (we can put 'Dr' with the nobles). Leave 'Rev' separate for now.
# 3. Compare survival rates for titles, by gender and class.

# In[ ]:


df['Title'].replace('Mme', 'Mrs', inplace=True)
df['Title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
df['Title'].replace(['Lady', 'the Countess', 'Dona'], 'FRare', inplace=True)
df['Title'].replace(['Sir', 'Jonkheer', 'Don'], 'MRare', inplace=True)
df['Title'].replace(['Col', 'Capt', 'Major'], 'Mil', inplace=True)

# female and male doctors
df.loc[(df['Title'] == 'Dr') & (df['Sex'] == 'female'), 'Title'] = 'FRare'
df.loc[(df['Title'] == 'Dr') & (df['Sex'] == 'male'), 'Title'] = 'MRare'

# mutate the training DataFrame, for exploration
train['Title'] = df.loc[:tr_len, 'Title']

# plot titles by gender and class
train_f = train[train['Sex'] == 'female']
train_m = train[train['Sex'] == 'male']
plt.figure(3, figsize=(12, 8))
plt.subplot(231)
sns.barplot(x='Title', y='Survived',data=train_f[train_f['Pclass'] == 1])
plt.xlabel('Females, 1st Class')
plt.subplot(234)
sns.barplot(x='Title', y='Survived', data=train_m[train_m['Pclass'] == 1])
plt.xlabel('Males, 1st Class')
plt.subplot(232)
sns.barplot(x='Title', y='Survived', data=train_f[train_f['Pclass'] == 2])
plt.xlabel('Females, 2nd Class')
plt.subplot(235)
sns.barplot(x='Title', y='Survived', data=train_m[train_m['Pclass'] == 2])
plt.xlabel('Males, 2nd Class')
plt.subplot(233)
sns.barplot(x='Title', y='Survived', data=train_f[train_f['Pclass'] == 3])
plt.xlabel('Females, 3rd Class')
plt.subplot(236)
sns.barplot(x='Title', y='Survived', data=train_m[train_m['Pclass'] == 3])
plt.xlabel('Males, 3rd Class')
plt.show()


# When separating by class, something becomes quite apparent: The titles (other than 'Master') don't seem to say much at all! Due to the very small sample sizes for the 'Rare' titles, we will combine them into 'Mrs' and 'Mr' to avoid overfitting. We'll leave the distinction between 'Miss' and 'Mrs' for now.
# 
# Finally, our feature engineering will allow `Title` to fully replace `Sex`, since the titles are no longer shared between genders.

# In[ ]:


df['Title'].replace('FRare', 'Mrs', inplace=True)
df['Title'].replace('MRare', 'Mr', inplace=True)
df['Title'].replace('Mil', 'Mr', inplace=True)
df['Title'].replace('Rev', 'Mr', inplace=True)
train['Title'] = df.loc[:tr_len, 'Title']


# ### Age Is but a Number
# 
# Next, let's take a look at `Age`. While it is one of the more popular predictors, we will eventually remove it completely! But to see why, let's first begin with a histogram of survivors and non-survivors.
# 
# Keeping RWC in mind, we will create a different graph for each gender - since there are many missing values, we won't separate this feature by passenger class.

# In[ ]:


fs_ages = train.loc[(train['Survived'] == 1) & (train['Sex'] == 'female'), 'Age'].dropna()
fd_ages = train.loc[(train['Survived'] == 0) & (train['Sex'] == 'female'), 'Age'].dropna()
ms_ages = train.loc[(train['Survived'] == 1) & (train['Sex'] == 'male'), 'Age'].dropna()
md_ages = train.loc[(train['Survived'] == 0) & (train['Sex'] == 'male'), 'Age'].dropna()

plt.figure(4, figsize=(8, 8))
plt.subplot(211)
sns.distplot(fs_ages, bins=range(81), kde=False, color='C1', label='Survived')
sns.distplot(fd_ages, bins=range(81), kde=False, color='C0', label='Died', axlabel='Female Age')
plt.legend()
plt.subplot(212)
sns.distplot(ms_ages, bins=range(81), kde=False, color='C1', label='Survived')
sns.distplot(md_ages, bins=range(81), kde=False, color='C0', label='Died', axlabel='Male Age')
plt.legend()
plt.show()


# Notice the gap around age 12-13. This will be a convenient binning point!
# 
# For a way to visualize overall survival *rate* by age, I used somewhat hacky method to create a "moving survival rate" window. For every x, the "survival rate" is the survival rate of people aged between x - 3 and x + 3 years, inclusive.

# In[ ]:


rate_by_age_m = np.zeros(80)
rate_by_age_f = np.zeros(80)
for i in range(80):
    ages = train[(train['Age'] >= i - 2) & (train['Age'] <= i + 4)]
    rate_by_age_m[i] = ages.loc[ages['Sex'] == 'male', 'Survived'].mean()
    rate_by_age_f[i] = ages.loc[ages['Sex'] == 'female', 'Survived'].mean()
plt.figure(5)
plt.plot(rate_by_age_m, label='Males')
plt.plot(rate_by_age_f, label='Females')
plt.xlabel('6-Age Window')
plt.ylabel('Survival Rate')
plt.legend()
plt.show()


# The biggest effect of age on survival is noticed for male children, where survival drops dramatically after about age 12. For females, there is a big dip in survival between ages 5-11, but babies and teenage girls seem to have normal survival rates. This may or may not be a random effect.
# 
# There isn't much data to support creating a new bin for the elderly (we see that such a bin will have a very low feature importance anyway). No men that were between 63 and 79 survived, though this may not be surprising given the small sample size (12) and the overall low survival rate of males. There is a single survivor at age 80 (causing the anomalous increase at the end). All women above age 57 survived, but again, there is a very small sample size (7) and females had an overall high survival rate.
# 
# The rest of the graph is less interesting. Men, surprisingly, seem to have a common survival rate. While it may be worthwhile to create an 'Elder' feature for ensemble-based learners, we will stick to identifying children.
# 
# For male children, this is surprisingly easy. Remember that `Title` feature from earlier? Let's have a look...

# In[ ]:


df.loc[df['Title'] == 'Master', 'Age'].describe()


# Well then. Let's see how many boys up to 14.5 years old have another title:

# In[ ]:


df.loc[(df['Title'] != 'Master') & (df['Sex'] == 'male') & (df['Age'] <= 14.5)]


# For the most part, it seems that 'Mr' and 'Master' capture the boundary between "children" and "men" quite well. To make the boundary clearer, we'll fix the 'Title' feature:
# 
# 

# In[ ]:


# boys below age 12 are 'Master'
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 12), 'Title'] = 'Master'
# teenage boys above age 12 are 'Mister'
df.loc[(df['Title'] == 'Master') & (df['Age'] > 12), 'Title'] = 'Mr'

train['Title'] = df.loc[:tr_len, 'Title']


# Finally, it seems that girls have a worse survival rate than women, but not by much. Recall from earlier, though, that 'Miss' had a lower survival rate than 'Mrs', but again, not by much. Could this explain the difference between girls and women?

# In[ ]:


print('Female Survival Rate: {}'.format(train_f['Survived'].mean()))
print('Miss Survival Rate: {}'.format(train.loc[train['Title'] == 'Miss', 'Survived'].mean()))
print('Girl Survival Rate: {}'.format(train_f.loc[train_f['Age'] <= 12, 'Survived'].mean()))


# Since have a slightly lower survival rate, we will create a `Child` feature, using age 12 as the cutoff. Based on the histogram, this seems to be the 'border' between where the differences in survival between children and adults manifest.
# 
# Also, out of the missing values, we will try to get at least the males that have a 'Master' title, through imputation via title.

# In[ ]:


df['Age'] = df.groupby('Title')['Age'].apply(lambda x: x.fillna(x.median()))
train['Age'] = df.loc[:tr_len, 'Age']
df['Child'] = df['Age'] <= 12
train['Child'] = df.loc[:tr_len, 'Child']


# ### Thicker Than Water
# 
# We have two features that are somewhat related: `SibSp` and `Parch`. A common tactic is to group the features into one, representing family size. Let's create this feature, and take a look at its distribution:

# In[ ]:


df['Family'] = df['SibSp'] + df['Parch']
train['Family'] = df.loc[:tr_len, 'Family']

plt.figure(6, figsize=(12, 8))
train_f = train[train['Sex'] == 'female']
train_m = train[train['Sex'] == 'male']
plt.subplot(231)
sns.countplot(train_f.loc[train_f['Pclass'] == 1, 'Family'])
plt.xlabel('Females, 1st Class')
plt.subplot(234)
sns.countplot(train_m.loc[train_m['Pclass'] == 1, 'Family'])
plt.xlabel('Males, 1st Class')
plt.subplot(232)
sns.countplot(train_f.loc[train_f['Pclass'] == 2, 'Family'])
plt.xlabel('Females, 2nd Class')
plt.subplot(235)
sns.countplot(train_m.loc[train_m['Pclass'] == 2, 'Family'])
plt.xlabel('Males, 2nd Class')
plt.subplot(233)
sns.countplot(train_f.loc[train_f['Pclass'] == 3, 'Family'])
plt.xlabel('Females, 3rd Class')
plt.subplot(236)
sns.countplot(train_m.loc[train_m['Pclass'] == 3, 'Family'])
plt.xlabel('Males, 3rd Class')
plt.show()


# We see that most travelers, especially males and those in the lower classes, traveled alone. There is then a group of men and women traveling with 1-2 family members; and fewer with 3 or more companions.
# 
# How did this impact survival?

# In[ ]:


plt.figure(7, figsize=(12, 8))
plt.subplot(231)
sns.barplot(x='Family', y='Survived',data=train_f[train_f['Pclass'] == 1])
plt.xlabel('Females, 1st Class')
plt.subplot(234)
sns.barplot(x='Family', y='Survived', data=train_m[train_m['Pclass'] == 1])
plt.xlabel('Males, 1st Class')
plt.subplot(232)
sns.barplot(x='Family', y='Survived', data=train_f[train_f['Pclass'] == 2])
plt.xlabel('Females, 2nd Class')
plt.subplot(235)
sns.barplot(x='Family', y='Survived', data=train_m[train_m['Pclass'] == 2])
plt.xlabel('Males, 2nd Class')
plt.subplot(233)
sns.barplot(x='Family', y='Survived', data=train_f[train_f['Pclass'] == 3])
plt.xlabel('Females, 3rd Class')
plt.subplot(236)
sns.barplot(x='Family', y='Survived', data=train_m[train_m['Pclass'] == 3])
plt.xlabel('Males, 3rd Class')
plt.show()


# We can draw the following conclusions:
# 
# * In first and second class:
#     * Women seem to have roughly the same survival chance, independent of family size.
#     * Men with larger family sizes seem to have relatively higher chances of survival. There is a spike for 3 companions for 
#     men and first class, but the sample size tells us it's extremely likely this association is due to chance.
# * In third class:
#     * Women and men seem to have relatively higher chances of survival, up to 3 companions.
#     * Women and men with 4 or more companions had drastically lower odds of survival.
# 
# To reduce the noise, we will create three bins for family size:
# 
# * Alone
# * 1-3 Companions
# * 4 or more companions

# In[ ]:


df['FamSize'] = (df['Family'] >= 4).astype(int) + (df['Family'] > 0).astype(int)
train['FamSize'] = df.loc[:tr_len, 'FamSize']


# ### All Hands on Deck!
# Next, let's take a look and see what we can extract from the `Cabin` feature.

# In[ ]:


train['Cabin'].dropna().head()


# Cabin numbers are in the format (letter)(number), the letter ostensibly indicating type of deck. Since it's possible certain decks may have had an advantage, we will use this feature. While there may be differences between cabin numbers, with the small number of cabins as is, it's likely this will simply identify specific passengers, rather than trends.

# In[ ]:


df['Deck'] = df['Cabin'].str[0]
df['Deck'].fillna('U', inplace=True)
train['Deck'] = df.loc[:tr_len, 'Deck']

train_f = train[train['Sex'] == 'female']
train_m = train[train['Sex'] == 'male']
plt.figure(8, figsize=(8, 8))
plt.subplot(221)
sns.barplot(x='Deck', y='Survived', data=train_f[train_f['Pclass'] == 1])
plt.xlabel('Females, 1st Class')
plt.ylabel('Survival Rate')
plt.subplot(222)
sns.barplot(x='Deck', y='Survived', data=train_m[train_m['Pclass'] == 1])
plt.xlabel('Males, 1st Class')
plt.ylabel('Survival Rate')
plt.subplot(223)
sns.barplot(x='Deck', y='Survived', data=train_f[train_f['Pclass'] == 2])
plt.xlabel('Females, 2nd Class')
plt.ylabel('Survival Rate')
plt.subplot(224)
sns.barplot(x='Deck', y='Survived', data=train_m[train_m['Pclass'] == 2])
plt.xlabel('Males, 2nd Class')
plt.ylabel('Survival Rate')
plt.show()


# Among females, the deck makes little difference - but, first and second class females were highly likely to survive at any rate! Among males (the gender of more interest), we see that cabins ranging from A-F had better odds of survival than an unknown cabin, or cabin T. Thus, we'll go with marking the decks A through F as 'good':

# In[ ]:


df['GoodDeck'] = df['Deck'].isin(['A', 'B', 'C', 'D', 'E', 'F'])
train['GoodDeck'] = df.loc[:tr_len, 'GoodDeck']


# ### Fare Thee Well
# 
# [Here](https://www.kaggle.io/svf/1331670/27182181000626b0a013fac5a5d6a0ba/__results__.html#Fare_eff_cat) is an excellent explanation of the theory that `Fare` represents the *total fare* paid by an entire group. Hence, `Fare` should actually be divided by ticket group size! Doing so will remove a lot of the skew in the feature:

# In[ ]:


df['TicketSize'] = df['Ticket'].value_counts()[df['Ticket']].values
df['AdjFare'] = df['Fare'].div(df['TicketSize'])
train['AdjFare'] = df.loc[:tr_len, 'AdjFare']
plt.figure(9)
sns.boxplot(x='Pclass', y='AdjFare', data=df[df['AdjFare'] > 0])
plt.show()


# The fares appear much more stratified by class. Let's instead replace `Fare` with a measure of overpayment or underpayment. We will group fares by passenger class, and then compute the Z-score.
# 
# Note that there is a single missing fare; we'll fill it in with the median (by class).

# In[ ]:


df['ScFare'] = df.groupby('Pclass')['AdjFare'].apply(lambda x: x.sub(x.median()).div(x.std())).fillna(0)
train['ScFare'] = df.loc[:tr_len, 'ScFare']


# Let's see the distribution of these scores:

# In[ ]:


plt.figure(10)
sns.distplot(df['ScFare'])
plt.show()


# From the KDE, it seemse that almost all of the fares are within two standard deviations of the mean. There is a slight bump two sd's below the mean, but not above. There are a very small amount of outliers, up to 6 sd's below or 8 sd's above the mean. Let's see how this score relates to survival chances:

# In[ ]:


center = np.zeros(136)
rate_by_fare_m = np.zeros(136)
rate_by_fare_f = np.zeros(136)
for i in range(136):
    center[i] = 0.1 * i - 5.5
    fares = train[(train['ScFare'] >= 0.1 * i - 5.7) & (train['ScFare'] <= 0.1 * i - 5.3)]
    rate_by_fare_m[i] = fares.loc[fares['Sex'] == 'male', 'Survived'].mean()
    rate_by_fare_f[i] = fares.loc[fares['Sex'] == 'female', 'Survived'].mean()
plt.figure(11)
plt.plot(center, rate_by_fare_m, label='Males')
plt.plot(center, rate_by_fare_f, label='Females')
plt.xlabel('0.4-Z-Score Window')
plt.ylabel('Survival Rate')
plt.legend()
plt.show()


# While there are some results for the outliers, again, with the small sample size, taking these into account simply fits the model for those outliers. Would someone who overpaid by 6 standard deviations naturally have a higher survival chance? Or did the single family that did simply happen to survive?
# 
# For these reasons, we'll exclude `Fare` entirely from our analysis.

# ### All That Matters Is Where You're Going
# 
# There are two missing values for `Embarked`, with several methods of imputation. One approach is ticket number:

# In[ ]:


df[df['Embarked'].isnull()]


# Since the ticket number starts with '113', we will check all such tickets:

# In[ ]:


df.loc[df['Ticket'].str.startswith('113'), 'Embarked'].value_counts()


# Since the vast majority of tickets starting with '113' boarded at Southampton, it's likely that this pair of tickets was meant to board at Southampton as well. So we can impute a value of 'S'.
# 
# However, it's unclear how `Embarked` actually influences survival other than acting as a marker for other already-present features. We will see that the feature importance of `Embarked` is very low, so it will ultimately not be used in the final model.

# In[ ]:


df['Embarked'].fillna('S', inplace=True)
train['Embarked'] = df.loc[:tr_len, 'Embarked']


# ## Model Selection and Validation
# 
# Now we are ready to give feature selection a go! We have the following features:

# In[ ]:


df.info()


# We now combine the dataframes and strip the following features:
# 
# * `Name` and `Sex` (replaced by `Title`)
# * `Age` (replaced by `Child` and `Title` to a lesser degree)
# * `SibSp`, `Parch` and `Family` (replaced by `FamSize`)
# * `Cabin` and `Deck` (replaced by `GoodDeck`)
# * `Fare`, `AdjFare` and `ScFare` (captured in `Pclass`)
# * `Ticket` and `TicketSize`
# * `Embarked`
# 
# This leaves us with the following features:
# 
# * `Title` - taking values `Mr`, `Mrs`, `Miss` and `Master`.
# * `Child` - `1` if `Age <= 12` and `0` otherwise.
# * `FamSize` - `0` if alone, `1` if with 1-3 family members, `2` if with 4 or more family members.
# * `GoodDeck` - `1` if staying in a cabin in Decks A, B, C, D, E or F.
# * `Pclass` - passenger class.

# In[1]:


dft = df.drop(['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Family', 'Cabin', 'Deck', 'Fare',
               'AdjFare', 'ScFare', 'Ticket', 'TicketSize', 'Embarked'], axis=1)


# Finally, we are ready to select our model! We'll use a random forest classifier, to get a list of feature importances.
# 
# First, we will encode our categorical variables with `LabelEncoder()`. Since we are using a random forest, we won't need to create dummy features. Then, we'll split our data back into the train and test DataFrames.
# 
# Random forests require setting several parameters, two main ones being `max_features` and `max_depth` which control model flexibility. We'll test several values for both using the `GridSearchCV()` function from scikit-learn:

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# encode categorical variables
le = LabelEncoder()
dft['Title'] = le.fit_transform(dft['Title'])

# recreate the training and test sets
Xp_train = dft[:tr_len]
p_test = dft[tr_len:]
p_train = Xp_train.join(train[['Survived']])

# split the new training set into X and y.
X = p_train.drop('Survived', axis=1)
y = p_train['Survived']

# select parameters
rf = RandomForestClassifier(n_estimators=500)
depths = [4, 5, 6, 7]
features = [2, 3]
rf_params = {'max_depth': depths, 'max_features': features}
grid = GridSearchCV(rf, param_grid=rf_params, cv=5).fit(X, y)
print('Parameter Scores:\n{}\n'.format(pd.DataFrame(
    grid.cv_results_['mean_test_score'].reshape(len(depths), len(features)),
    index=depths, columns=features)))
print('Feature Importances:\n{}'.format(pd.Series(
    grid.best_estimator_.feature_importances_, index=X.columns)))


# And finally, we have a list of feature importances as well as the best parameters to use for the classifier! Note that `Child` is the weakest feature, so taking it out (and having `Title` supersede it completely) may be an option.
# 
# We can either use the random forest classifier we just trained to make our test set predictions, or we can use another classifier such as logistic regression or SVM's. Using only 5 features, all of these simple methods should be able to obtain as high as 79.9% accuracy on the public test set! We save our predictions as follows:

# In[ ]:


predicted = np.column_stack((p_test.index.values, grid.best_estimator_.predict(p_test)))
np.savetxt('prediction.csv', predicted.astype(int), fmt='%d', delimiter=',',
           header='PassengerId,Survived', comments='')


# ## Afterword
# 
# Using only 5 features, we were able to obtain a model obtaining 0.79904 on the public leaderboard! More importantly, our chosen features were well reasoned, and many "noisy" features were removed. The model is highly interpretable.
# 
# However, it's still not quite satisfying. Among the issues I have encountered:
# 
# * The `Child` feature is relatively weak in feature importance. However, I have not found a way to remove the feature 
# without worsening the leaderboard performance.
# * Dropping features can remove interactions. In particular, would there be a better way to bin `Age`? How about `Family`?
# * I've more or less hit a brick wall in trying to hit the coveted 0.80, without resorting to [tailoring my model around the  LB][1]. By this, I mean: I've read (and can obviously reproduce) kernels which have higher than a 0.80. Ultimately, though, I've used a rule where if I can't justify a feature transformation, I won't use it. Otherwise, of course, I could just create different features all day, until I randomly hit 0.80 out of luck.
# * That said, "not fitting to the leaderboard" is easier said than done. First, the LB is only based on 209 entries, which makes even unintentional fitting quite easy. The second problem is the small training set size, which makes cross-validation unreliable. In other words: What better metric do we have besides the public LB for how well our result will truly generalize? Who knows.
# 
# Ultimately, I've concluded that the best way forward from this simple, model is to create more features, and use ensemble methods (which have the power to weight the weaknesses of each model). In the end, this "competition" is just a learning example! This model and exercise has done well for me; hopefully, it will for you too.
# 
# As this is my first kernel I've fully completed for this site, I welcome any suggestions, comments, or corrections - as someone who has been a long-time lurker and reader, I'm excited to begin contributing back to the community as well. To those who have already published kernels, many of which I have used, thank you. To those reading this kernel, thank you as well!
# 
# 
#   [1]: https://stats.stackexchange.com/questions/291827/variable-transformation-on-kaggle-titanic-problem
