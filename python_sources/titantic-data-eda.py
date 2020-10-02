#!/usr/bin/env python
# coding: utf-8

# # Initial Exploratory Data Analysis
# This is a first pass analysis of the Titanic Data set. 
# 
# When I first came to kaggle to start learning and practicing machine learning, I got stuck badly on this data set. I could not beat the naive approach of labeling all females as survivors. It was so frustrating that I dropped maching learning for a long period after my defeat. 
# 
# That said, I couldn't come back to Kaggle without taking another swing at this data set. The goal of this analysis is to determine strong indicating features for predicting survival that I'll use later to guide my implimentation of a classifier. My hope is that this time I can beat the naive approach from the tutorials.
# 
# Also, something that I failed to recognize durring my previous attemps is the benefit of seeking feedback. If you happen to read thought this and notice places where I could improve my analysis, 

# In[1]:


from IPython.display import display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Getting a Feel for the Data Set
# 
# Here I am just taking a peak at the columns and various summary stats to get a feel for what the data looks like. I didn't really record any of my thoughts as I did this. I might come back and try to fill this part in with notes later..

# In[2]:


df = pd.read_csv('../input/train.csv')
display(df.head())
display(df.describe())


# In[3]:


df['Embarked'].value_counts()


# In[4]:


df['SibSp'].value_counts()


# In[5]:


df['Ticket'].value_counts()[:10]


# ## Gender and Class
# The low hanging fruit here is gender and class. These categorical variables are already well cleaned and represent a binary grouping of passengers. Let's take a look at what the survival rates look across these two groups:

# In[42]:


_ = sns.barplot(data=df, x='Sex', y='Survived')


# The survival rate for women is almost 4x that of men! The first thing that comes to mind is the addage "ladies first". Let's take a look at the counts across male and female passangers to get an idea of where we want to go from here:

# In[55]:


_ = sns.factorplot(data=df, x='Sex', kind='count', col='Survived')


# From count alone we can see that if we predicted survival ased on gender there would only be ~200 misclassifications in the 891 training examples. Which already isn't too bad. To improve on this, we would need to determine factors that could be used to identify male survivors and / or factors that increase the risk of perishing for female passengers.
# 
# Another feature that is fairly complete and clean is the passenger class, `Pclass`. It's reasonable to expect that the class of passenger would have some effect on survival rate. Passenger's of high class would might have been given preferential treatement or more immediate access to liveboats (cabins closer to the deck).
# 
# Let's take a look at surival rate by `Sex` and `Pclass`:

# In[6]:


_ = sns.barplot(data=df, x='Sex', y='Survived', hue='Pclass')


# As expected, passangers of high class have a higher survival rate. Female passangers in 3rd class have a considerably lower surival rate. Male survival rates are generally lower but, there is a large spike in male survivors in the 1st class. 
# 
# What other factors could effect suvival rates and specifically: what factors can help us determine which male and 3rd class, female passangers survived?
# 
# Some ideas that come to mind immediately are:
# - Age: Young male passanges may be more likely to survive. As it is likely that children would get some priority treatment in emergencies.
# - Family / Relatives: Children would likely get high priority. I would expect that members of families might have a better survival rate as well. `'SibSp'` and `'Parch'` can help create a metric of family for passengers.
# - Relative Location: Cabins can give us some sense of what part of the ship a passanger might have been when the Titanic began to sink. People on lower decks would have a harder time making it to the rafts , reducing their chances of survival.
# - Social Standing: Passanger class hints at this a bit, but a stronger underlying factor might be a person's status/wealth/title. This might be something we can determine based on passanger's fare and name field. 
# 
# Features that I don't expect a lot from:
# - Embarked: This feature doesn't scream helpful. Alone this feature doesn't seem like something that would be closely tied to survival rate. It could still be potentially useful in filling in missing values but, even that is a stretch...
# - Ticket: This feature looks really messy. Though, there are some recurring values which may indicate families traveling together
# 

# ## Age
# Let's start by taking a look at the age distribution among passangers

# In[7]:


df.loc[: ,'Age'].hist(bins=40)
plt.title('Age Distribution of Passengers')
_ = plt.show()


# From the histogram above, we can see that children make up a small amount of the passangers aboard the Titanic. The largest group of passangers is young adults and adults. The next largest being middle aged.
# 
# Let's take a look at these distributions by survival to get a general idea of the strength of age as a potential indicator.

# In[8]:


df.loc[df['Survived'] == 1, 'Age'].hist(alpha=0.5, bins=40, label='Survived')
df.loc[df['Survived'] == 0, 'Age'].hist(alpha=0.5, bins=40, label='Perished')
plt.title('Age Distributions')
plt.legend()
_ = plt.show()


# We can see that a more than half of chidren passengers survived. The number of young children that surived is especially high. 
# 
# Survivor counts among adults are lower relative to population but, there are still a large number of surviving adults. This might partly be due to female passangers having such strong survival rates. Let's take another look at these distributions for male passangers:

# In[9]:


males = df[df['Sex'] == 'male']
males.loc[males['Survived'] == 1, 'Age'].hist(alpha=0.5, bins=40, label='Survived')
males.loc[males['Survived'] == 0, 'Age'].hist(alpha=0.5, bins=40, label='Perished')
plt.title('Age Distributions of Male Passengers')
plt.legend()
_ = plt.show()


# Now we are starting to see a much more compelling case for age's predictive power. While all other age groups have very low survival counts, male children are more likely to survive than perish. Compain this distribution to the previous one, it looks like gender has little indicative power when it comes to children.
# 
# Let's bin up our ages and look at some barcharts to get a feel of how age look in terms of average survival rates. For the bins, we could go big with 40 bins (like our histograms) or conservatively try to break our bins up into meaningful age groups. The former could allow our model to make decissions about which age bins are valuable indicators, but gives up some interperability of our model. For exploration purposes, I'll go with the later.  When it comes time to create a model, I might choose to go with a greater bin number.
# 
# Selecting my bins partly arbitrarily while staring at the distributions above, I ended up with the following:
# 
# - Young Children: Ages `[0,5)`
# - Children: Ages `[5, 10)`
# - Teens: Ages `[10,15)`
#   - It seems worth distinguishing these three groups based on the above distributions
# - Young Adults: Ages `[15,25)` I feel this choice deserves and explanation here is some of my reasoning:
#   - The age of adulthood was lower before the 1900s
#   - Many traditional rites of passage occur before age 15
#   - By age 15, people are very autonomous
#   - In an emergency I don't think they would be expected to need as much assistance
# - Adults: Ages `[25,45)`
# - Seniors: Ages `[45,80)`

# In[10]:


df['age_bin'] = pd.cut(df['Age'], bins=[0, 5, 10, 15, 25, 45, 80])
females = df[df['Sex'] == 'female']
males= df[df['Sex'] == 'male']

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches((20, 4))

sns.barplot(data=df, x='age_bin', y='Survived', ax=ax1)
ax1.set_title('Survival Rates by Age Group')
[tick.set_rotation('vertical') for tick in ax1.get_xticklabels()]

sns.barplot(data=males, x='age_bin', y='Survived', ax=ax2)
ax2.set_title('Male Survival Rates by Age Group')
[tick.set_rotation('vertical') for tick in ax2.get_xticklabels()]

sns.barplot(data=females, x='age_bin', y='Survived', ax=ax3)
ax3.set_title('Female Survival Rates by Age Group')
[tick.set_rotation('vertical') for tick in ax3.get_xticklabels()]

_ = plt.show()


# I'm pretty happy with this binning, It sucessfully captures the higher survival rate of children while preserving some nuance within the younger groups. Ages $[15, 80)$  don't vary as much but, that was expected from looking at the distributions earlier. I just didn't feel right grouping them all into a single bin. 

# ## Titles
# If we look at the names, we can see the middle word is an honorific. All names apear to follow the pattern `<family name>, <honorific>. <given name>`.

# In[11]:


import random
random.sample(list(df['Name'].values), 5)


# Let's break this honorific value out and take a look at is as it's own feature. We'll start by appling a conditional split for the titles and look at the distribution count of the values:

# In[12]:


df['title'] = titles = df['Name'].apply(
    lambda x: x.split(',')[1].split('.')[0] if ',' in x else x)
df['title'].value_counts()


# For safe measure, let's take a peak at our test set to make sure that we have all our honorifics accounted for.

# In[13]:


test_df = pd.read_csv('../input/test.csv')
test_df['title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
test_df['title'].value_counts()


# My initial impression is that the spread here isn't great. The majority of our values are in `'Mr.'`, `'Miss.'` and `'Mrs.'` with the remaining values spread thinly over a large number of titles. I want to try binning some of these categories up to make a new feature. Before I do that though, I want to get a feel for any trends in the existing titles.

# In[14]:


sns.barplot(data=df, x='title', y='Survived')
_ = plt.xticks(rotation='vertical')


# We can see from the plot above that that titles that indicate status, such as `'Mme'`, `'Lady'` and `'Sir'`, have higher survival survival rate. Titles that indicate military servie all have similar survival rates (however, this is based on a very small population).
# 
#  Let's try mapping these titles into categories based on status. To do this, I've defined a function calledd `map_title` that taes a title places it in one of 4 categories:
#  - `untited`: general titles
#  - `titled`: titles that indicate status
#  - `service`: titles that indicate military service
#  - `cloth`: a seperate category to distinguish `Rev` from `service`
#  - If all else fails, return the title given

# In[15]:


def map_title(title):
    title = title.strip()
    if title in ['Mr','Mrs', 'Mme', 'Miss', 'Ms', 'Mlle']:
        return 'untitled'
    if title in ['Lady', 'Sir', 'Jonkheer', 'Master', 'the Countess', 'Don', 'Dona']:
        return 'titled'
    if title in ['Dr', 'Major', 'Col', 'Capt']:
        return 'service'
    if title in ['Rev']:
        return 'cloth'
    return title


# Let's look at the counts we produce with these new categories:

# In[16]:


titles.apply(map_title).value_counts()


# These results are look pretty skewed. However, ignoring `untitled` (which is essentially the `null` condition), we have 45, 12 and 6 which is a much better distribution of values. Maybe I am overselling my work here, but at the very least we no longer have one offs like `the Countess` and `Jankheer` which were unlikely to provide any predictive value to new observations. 
# 
# Let's take a look at our survival rate for these titles. In this plot, I'll also bisect the data by sex (infact I'll do this a lot) because sex is such a strong indicating feature. If I did not look at each sex individually, I would not be able to say if category was a strong idicator of survival rate or just predominantly female.

# In[17]:


df['title_'] = df['title'].apply(map_title)
_ = sns.barplot(data=df, x='title_', y='Survived', hue='Sex')


# From the chart, we can see that `titled` passanges have a higher survival rate among men and women, `untitled` survivors are exceptionally low among men, `cloth` has no survivors which was partly expected and `service` is a bit of a mixed bag.  `titled` stands out as a strong indicator of male survival.
# 
# ## Marital Status
# 
# Another idea that occured to me while working through this is to create a feature that indicates marital status using the title data. To do this I start with `map_marraige` which simply translates female titles to `married` and `unmarried`. For now, we will dump the other titles into a bin called `unknown`.

# In[19]:


def map_marriage(title):
    title = title.strip()
    if title in ['Mrs', 'Mme', 'the Countess']:
        return 'married'
    if title in ['Ms', 'Miss', 'Mlle', 'Lady', 'Dona', 'Rev']:
        return 'unmarried'
    return 'unknown'


# In[20]:


titles.apply(map_marriage).value_counts()


# That's more unknowns than I was hoping for, tbh. Let's see if we can make an educated guess about the rest of the passangers based on other features:
# 
# From the data dictionary we know the `SibSp` indicates the combined count of siblings and spouses. But, it is hard to say someone is traveling with a sibling or a spouse. One possible answer is Age. We can look at the Average age of marriage for the 1910s and that value as a threshold: If a passanger is above the average age of marriage with a `SibSp > 0` it is more likely that they are married than traveling with a sibling.
# 
# A little investigation tells me that the [median age of marriage](https://www.infoplease.com/us/marital-status/median-age-first-marriage-1890-2010) for men was 25 in the 1910s. We'll use this value to guess the remaining marital statuses. Let's give it a shot and see what we get:

# In[21]:


df['married'] = df['title'].map(map_marriage)
married = ((df['Age'] >= 25) & (df['SibSp'] > 0)).apply(
    lambda x: 'married' if x else 'unmarried')
mask = df['married'] == 'unknown'
df.loc[mask, 'married'] = married[mask]

df['married'].value_counts()


# It looks like this got us an extra 5 married people and the rest went to unmarried. It is a bit hard to gauge how sucessfull this approach was by just counting the resulting values. I'm not really sure what the best evaluation metric for this would be. I'll try revising this later... Some ideas:
# - Looking at age distribution among married and unmarried groups
# - Looking at SibSp distribution
# - Experimenting with additional threshold values
# 
# Putting that all aside for now, let's take a look at marital status as an indicator of survival rate:

# In[22]:


df = df.sort_values(by='married')
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches((15, 4))

titles = ['Marriage Status and Survival Rate',
          'Marriage Status and Survival Rate by Sex']
sns.barplot(data=df, x='married', y='Survived', ax=ax1)
ax1.set_title(titles[0])
sns.barplot(data=df, x='Sex', y='Survived', hue='married')
ax2.set_title(titles[1])
_ = plt.show()


# This doesn't look great... Generally being married looks like a strong indicator of survival, but the sex break down looks pretty weak. In both sexes marriage only provides a small `0.1` to `0.2` boost in survival rate over not being married. 
# 
# Why is this? It's likely that the passengers labels as married are predominantly female. Remember, we had to guess for men and even then we didn't identify a lot of men as married. If we take a look at  the count of married and unmarried passangers by gener, we can see how skewed the `marriage` feature is:

# In[29]:


_ = sns.factorplot(data=df, x='married', hue='Sex', kind='count')


# ## Location
# One idea was to break out the letter prefix from the cabin as the deck and use that as a feature. This look good on paper, but there aren't very many observations...

# In[31]:


df['deck'] = df['Cabin'].apply(lambda x: x if pd.isna(x) else x[0])
print("value counts:")
print(df['deck'].value_counts())
print("finite count:")
print(df['deck'].value_counts().sum())
print("nan count:")
print(df['deck'].isna().sum())


# This is pretty sparse. Let's see if we can gain any insights from plotting survival rates:

# In[32]:


_ = sns.barplot(data=df, x='deck', y='Survived', hue='Sex')


# Nothing really outstanding here. If there was a noticable trend here there might have been some value in trying to fill the missing deck values.
# 
# As a quick check for how easy filling the missing values would be, I decided to plot the `fare` against `deck`. 

# In[36]:


_ = sns.barplot(data=df, x='deck', y='Fare')


# A lot of the decks have clear non-overlapping ranges of price.  However, `deck` didn't look great to begin with. Let's shift our focus to the `fare` instead.

# ## Fare
# We've already seen that the fare paid is a strong indicator of the deck that a passanger is on.  Let's see how `fare` fares as an indicator of survival.
# 
# First, we'll bin up fare by intervals of 10 and take a look at the distribution of prices paid:

# In[37]:


df['fare_bin'] = pd.cut(df['Fare'], bins=list(range(0, 150, 10)))
_ = df['fare_bin'].value_counts().sort_index().plot(kind='bar')


# This is pretty skewed but, that is partly to be expected. There were likely fare more 3rd class cabins than 1st and 2nd class. 
# 
# Let's take a look at the survial rate by fare bin:

# In[39]:


sns.barplot(data=df, x='fare_bin', y='Survived')
_ = plt.xticks(rotation='vertical')


# This looks promising, the fare paid increases, passanger survival rate also increases. We need to be mindful thought that past the `(80, 90]` range we have a very small number of observations. We also haven't ruled out `Sex` as an underlying factor. Let's take a look at the same chart broken up by passanger sex:

# In[40]:


sns.barplot(data=df, x='fare_bin', y='Survived', hue='Sex')
plt.gcf().set_size_inches((10, 4))
_ = plt.xticks(rotation='vertical')


# This looks pretty good. Looking at male survival rates we still see a similar trend. Among female passangers the trend is less clear. But, I'm not too concerned since I am not as interested in features that indicate female survival since female surival is generally already very high.
# 
# ## Thoughts for Further Analysis and Modeling
# 
# A quick summary of findings so far:
# - `Sex` is a very strong indicator of survival rate. Female passangers were much more likely to survive than male passangers. 
# - `Pclass` has some potential predictive strength and might help us get a stronger prediction on survival:
#   - 1st class male passengers are more likely to suvive than other male passangers by a moderate margin.
#   - 3rd class female passangers are much less likely to surivive that 1st and 2nd class.
# - Age has the potential to be a strong predictor of survival rate, although children make up a small part of the passenger sample, they have a generally high survival rate regardless of sex.
# - We are able to split out honorific titles from passengers names and use them to categorize our passengers:
#   - Binning these titles by societal roles is very promising.
# - The fare that passangers paid also seems to have some potential predictive power for male passengers. It is likely that this ties into customer class and cabin location.
# - Features that didn't appear promising:
#   - `Embarked`: This feature doesn't seem to have any strong correlation with survial rate.
#   - Marital Status: Although I explored this a bit, it was generally hard to identify married males. This results in the feature being skewed toward women passangers.
#   - Deck / `Cabin` location: This field was too sparse to make use of.
#   
# With all this information, I think we can begin to train a model for predicting survival rates among passangers. In my next notebook, I'll try to build and train a classifier that can perform better than predicting all women survive and all men. In my previous dives into machine learning, that was about where I threw up my arms and quit. I'm hoping that won't be the case this time around... Wish me luck! 
# 
# I'll update this kernel to link the next one if it is a sucess. Also, any feed back and suggestions would be greatly appreciated.
