#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel will dive deeply into the data exploration phase of the Titanic Machine Learning Competition and will conclude with the beginnings of a framework that makes it easy to run various models using multiple feature subsets, leveraging Backward Feature Elimination and Cross-Validation for parameter tuning.
# 
# I did my original analysis in Spyder using multiple module files so converting the code into a format for a Jupyter Notebook has been educational to say the least. Hopefully the fusion of the two approaches has resulted in code that is streamlined and easy to follow.
# 
# First, the standard library imports.

# In[ ]:


import pandas as pd
import numpy as np
import math
import re
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

sns.set(font_scale = 1.4)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

g_colors = ['red', 'green']
g_sex = ['male', 'female']


# # Basic Data Exploration Phase
# 
# In approaching a new data project, it can be tempting to dive in and start exploring / meandering down a road of summaries and quick plots.
# 
# I find that this approach can turn into a series of tweaks and additions that leaves me questioning whether I'm made progress or simply done a bunch of aimless work. And it inevitably leads to a cleanup phase where I have to reorganize everything I've done to make it more intelligible.
# 
# To mitigate this situation, I try to put some structure around my efforts and attempt to create a reasonable definition of "Done" by laying out a list of specific questions to focus on.
# 
# The first phase of this kernel will be an initial survey of the raw dataset aimed at becoming familiar with the data at a very basic level. So I'll focus on the following questions:
# 1. What is the basic description of the data fields?
# 2. How is the training set's response data distributed?  
# 3. How are the raw feature fields distributed?  
# 4. What data is missing?  
# 
# After I've answered these questions, I feel confident that I'm ready to move onto the next task.

# ## 1) What Is The Basic Description Of The Data Fields?
# Below are the columns from the training data.
# 
# [Notice that I lowercase all of the column names right off the bat. I generally avoid all capitalization in my [python/R] code as it's one less detail to worry about.]

# In[ ]:


train_raw = pd.read_csv('../input/titanic/train.csv')
train_raw.columns = [col.lower() for col in train_raw.columns]
train_raw.columns


# **survived** is the *response variable*, the value we will be trying to predict.
# 
# **passengerid** is a rowid and will basically be a throwaway field.
# 
# So our raw dataset is comprised of 5 strings and 5 numerics.

# In[ ]:


train_raw.iloc[:, 2:].dtypes


# In[ ]:


train_raw.iloc[:, 2:].describe()


# In[ ]:


train_raw.iloc[:, 2:].head(10)


# I see some NaNs, **pclass** looks interesting, I have some questions about **sibsp** and **parch**, I'm tempted to start munging but let's continue with the high level exploration.

# ## 2) How Is The Response Data Distributed?
# We don't want to dig into the full training set too much before splitting into the train/test sets as even visually examining the test data could lead us to "mentally overfit" with ideas for feature generationthat we shouldn't have. But a basic survey of the full dataset should be ok and can even be helpful to anticipate data quality problems that we will need to address down the road.
# 
# First the survival rate.

# In[ ]:


pd.DataFrame(dict(counts = train_raw['survived'].value_counts(), 
                  percent = train_raw['survived'].value_counts(normalize=True)))


# 891 records in our data set, ~1/3 survived, 2/3 died.  [Wikipedia](https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic#Casualties_and_survivors) indicates ~68% of people died on the titanic.  
# 
# Based on our training data, if we predict that everyone from the test set dies, we can expect to achieve **~62-68% accuracy.**  
# 
# So our initial goal will be to exceed 68%.

# ## 3) How Are The Raw Features Distributed?
# Let's look at how the feature sets are distributed in our dataset.  First, we will visualize features that have fewer than 10 unique values.  
# Note that the following graphs use a shared y-axis of percentages.

# In[ ]:


fig, axes = plt.subplots(1, 6, sharey = True, figsize = (15, 3))

# Because we are skipping certain columns, we cannot use enumerate for a counter
k = 0

for col in train_raw.columns.values:
    if len(train_raw[col].unique()) < 10:
        ax_cur = axes.ravel()[k]
        sns.barplot(x = train_raw[col].value_counts(normalize = True).index,
                    y = train_raw[col].value_counts(normalize = True),
                    palette = "Dark2",
                    ax = ax_cur)
        ax_cur.set_title(col)
        ax_cur.set_ylabel('')
        
        k += 1


# The chart displays the **survived** ratio of 1/3 vs 2/3 that we calculated before.  
# We see a similar ratio between men and women.  
# Based on **pclass**, over 50% of the passengers were in third class.  
# Based on **sibsp** and **parch**, the majority of people were sailing alone or with one person.  
# Most people **embarked** the Titanic at Southampton.  

# It's a fair assumption that **age** and **fare** will be important features in our models. Let's take a look at their overall distribution.
# 
# Note: Any assignment to "eat" is done to suppress the output of the unattractive / uninformative plotting object. I use this convention throughout the notebook

# In[ ]:


fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 3))

sns.distplot(train_raw.loc[train_raw.age.notnull(), 'age'],
             ax = axes[0],
             color = '#632de9') #purple blue

sns.distplot(train_raw.loc[train_raw.fare.notnull(), 'fare'],
             ax = axes[1],
             color = '#76cd26') #apple green

axes[0].set_title('Age Distribution')

eat = axes[1].set_title('Fare Distribution')


# A majority of 20-somethings on the Titanic with a surprisingly large group of very young passengers.
# 
# ***fare** is highly skewed so let's zoom in for a better look.  
# 
# *Note: Unlike the previous charts, the following charts do NOT share the same y-axis.*

# In[ ]:


fare_boundary = 110
fig, axes = plt.subplots(1, 2, figsize = (15, 3))

sns.distplot(train_raw.loc[train_raw.fare < fare_boundary, 'fare'],
             ax = axes[0],
             color = '#0cff0c') #neon green

sns.distplot(train_raw.loc[train_raw.fare >= fare_boundary, 'fare'],
             ax = axes[1],
             color = 'green')

axes[0].set_title('Fare < $' + str(fare_boundary))
eat = axes[1].set_title('Fare >= $' + str(fare_boundary))


# The vast majority of people paid ~10-20 dollars per trip.  (Don't be fooled by the non-matching y-axis.)
# 
# Looks like a few paid ~500 dollars for their tickets. Out of curiosity, let's take a look at them:

# In[ ]:


train_raw[train_raw.fare > 400]


# 3 people at $512. Not surprisingly, we see that each of them survived. Definitely a hint at where the analysis will lead us.

# ## 4) What Data Is Missing?
# One last bit of peaking at the full dataset before we separate out the train/test sets. Below are the columns that have NaN values. 

# In[ ]:


nans = [(col, sum(train_raw[col].isnull())) for col in train_raw.columns.values if sum(train_raw[col].isnull()) > 0]
pd.DataFrame(nans, columns = ['Column', 'NaNs'])


# A few missing **embarked** values, **cabin** is largely empty and **age** has a significant number of missing values.  
# Each of these fields will need to be addressed in the data munging phase.

# # Deep Dive Into The Training Set
# Now that we have a general understanding of the data we're working with, it's time to split off the training / test set so we can do a deeper analysis of the data to determine which features to include / create for the modeling effort.
# 
# Again, let's start with a list of questions to guide our efforts:
# 1. How is survival distributed against the various raw features?
# 2. How are the various features correlated?
# 3. What pairs of features have interesting survival relationships?
# 4. What additional features appear to be worth engineering?

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train_raw, train_raw.survived, random_state = 1)


# ## A Note About Data Objects and Naming Conventions
# Data munging can lead to a lot of overwriting / modifications of objects and if done haphazardly, it can lead to confusion. I like to set up clearly delineated data flows and naming conventions to keep things segregated and clean. 
# 
# This project has datasets explicitly named "train.csv" and "test.csv." And we need to split "train.csv" into train and test sets and then later act on the imported test.csv file. Ugh
# 
# To avoid this confusion, I refer to the "test.csv" dataset as "eval."
# 
# Below is the full list of data objects I came up with.
# 
# * **train_raw** -> train.csv
# * **eval_raw** -> test.csv
# * **x_train, x_test, y_train, y_test** -> Resulting datasets from train_test_split()
# * **x_train_tidy, x_test_tidy, x_eval_tidy** -> These are the transformations of the x data, essentially the results of my data munging. They include all cleaned up data and engineered features.
# * **x_train_thin, x_test_thin, x_eval_thin** -> There are the subset of features from the tidy objects that will be used for modeling.
# 
# This is a non-descructive data flow that allows easy reference to source objects and it's easy to remember while I code. At no point do I need to go back and run half of my script to see what the data originally looked like or to answer a question I might have about some data's previous state. 

# ## 1) How Is Survival Distributed Against The Raw Features?
# Some simple barplots will give us a first look into how survival is affected by the various features we have. Let's look first at the categorical feature set.

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize = (15, 8))
plt.subplots_adjust(hspace = .3)

for k, col in enumerate(['pclass', 'sex', 'embarked', 'sibsp', 'parch']):
    pd.crosstab(x_train['survived'], x_train[col]).T.plot(kind = 'bar',
                                                          color = g_colors,
                                                          ax = axes.ravel()[k],
                                                          rot = 0)
    
plt.suptitle('Survival Counts By Categorical Feature', fontsize = 20)
fig.delaxes(axes[1][2])


# In[ ]:


fig, axes = plt.subplots(2, 3, sharey = True, figsize = (15, 8))
plt.subplots_adjust(hspace = .3)

for k, col in enumerate(['pclass', 'sex', 'embarked', 'sibsp', 'parch']):

    d = pd.crosstab(train_raw['survived'], train_raw[col])

    colors = np.array([g_colors[c] for c in (d.iloc[0] < d.iloc[1]) * 1])

    axes.ravel()[k].bar(x = d.columns,
                        height = (d.iloc[1] - d.iloc[0]) / (d.iloc[1] + d.iloc[0]),
                        color = colors)
    
    axes.ravel()[k].set_title(col)
    axes.ravel()[k].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    
plt.suptitle('Survival Percentages By Categorical Feature', fontsize = 20)
fig.delaxes(axes[1][2])


# Looks like sailing third class does not bode well for survival.  
# 
# Women do far better than men as well.  
# 
# Being alone or part of a large family seems to work against you. It's worth noting, however, the low number of passengers having **sibsp** or **parch** > 2.
# 
# Having **embarked** from Southampton, where most people boarded, also has a low survival ratio compared to Cherbourg.
# 
# Let's look at the two continuous features we have: **age** and **fare**

# ### **Age**

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize = (15, 4))
axes.set_xticks(range(0, 100, 10))

for i in range(0, 2):
    sns.distplot(x_train.loc[(x_train.age.notnull()) & (x_train.survived == i), 'age'],
                 color = g_colors[i],
                 hist = False,
                 ax = axes)

axes.set_title('Survival By Age')
z = axes.set_xlim(x_train.age.min(), x_train.age.max())


# I cut off the density chart at 0 to avoid the visual illusion that even more young children survived than actually did. But clearly, being under the age of ~12 increases one's chances of surviving. Meanwhile, older teenagers and 20-somethings seem to have been asked to fend for themselves and had a much lower survival rate.
# 
# From about the age of 35 and older, the survival rate seems pretty even.
# 
# With the **sex** / **survived** chart from above and the **age** chart here, we see that the "women and children first" policy seems to have been followed.
# 
# Let's look at **age** and **sex** together.

# In[ ]:


fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 4))
axes[0].set_xticks(range(0, 100, 10))

for i in range(0, 4):
    var_survived = i % 2
    var_sex = math.floor(i / 2)
    
    sns.distplot(x_train.loc[(x_train.age.notnull()) & (x_train.sex == g_sex[var_sex]) & (x_train.survived == var_survived), 'age'],
                 color = g_colors[var_survived],
                 hist = False,
                 ax = axes[var_sex])
    
    axes[var_sex].set_title(g_sex[var_sex].title() + ' Survival By Age')
    
    axes[var_sex].set_xlim(x_train.loc[x_train.sex == g_sex[var_sex], 'age'].min(), x_train.loc[x_train.sex == g_sex[var_sex], 'age'].max())


# Hmm, younger women under 30 had a slightly worse survival rate for all ages but it looks like it was offset by a higher survival rate for women almost to the age of 30.
# 
# Young boys appear to have been taken care of the best.

# ### **Fare**

# In[ ]:


fig, axes = plt.subplots(1, 1, sharex = True, sharey = True, figsize = (15, 5))

for i in range(0, 2):
    sns.distplot(x_train.loc[(x_train.fare.notnull()) & (x_train.survived == i), 'fare'],
                 color = g_colors[i],
                 hist = False,
                 ax = axes)

axes.set_title('Survival By Fare')
z = axes.set_xlim(0, x_train.fare.max())


# Yea, that's not going to work. Let's try two different approaches. First we'll try zooming in on different ranges of **fare**. The second approach will use a log transformation of **fare**

# In[ ]:


fare_lt = x_train[(x_train.fare.notnull()) & (x_train.fare < 50)]

fig, axes = plt.subplots(1, 1, figsize = (15, 5))

for i in range(0,2):
    sns.distplot(fare_lt.loc[(fare_lt.survived == i), 'fare'],
                 color = g_colors[i],
                 hist = False)

axes.set_xlim(fare_lt.fare.min(), fare_lt.fare.max())
z = axes.set_title('Survival for Fares Less Than $50')


# In[ ]:


fare_gt = x_train[(x_train.fare.notnull()) & (x_train.fare >= 50)]

fig, axes = plt.subplots(1, 1, figsize = (15, 5))

for i in range(0, 2):
    sns.distplot(fare_gt.loc[(fare_gt.survived == i), 'fare'],
                 color = g_colors[i],
                 hist = False)

axes.set_xlim(fare_gt.fare.min(), fare_gt.fare.max())
z = axes.set_title('Survival for Fares Greater Than $50')


# We need to be mindful of the differing y-axis scales as there are far fewer people in the high-priced group. But they certainly seemed to have better survival rates.
# 
# People who paid less than ~12 dollars had a noticeable reduction in their survival rate.
# 
# Now let's try using a log transformation on **fare** to view the highly skewed data.

# In[ ]:


fares = x_train[(x_train.fare.notnull()) & (x_train.fare > 0)].copy()

fares['fare_log'] = np.log(fares.fare)

fig, axes = plt.subplots(1, 1, figsize = (15, 5))

for i in range(0, 2):
    sns.distplot(fares.loc[(fares.survived == i), 'fare_log'],
                 color = g_colors[i],
                 hist = False)

z = axes.set_title('Survival By Log Transformation Of Fare')


# Now we can look at the full set of customers and see how **fare** influenced survival. As we saw before, low paying passengers fared worse than those paying more for their tickets.
# 
# Two columns we haven't looked at are **name** and **cabin** but I'm going to defer those until later because both are going to require some data munging.

# ## 2) How Are The Various Features Correlated?
# The standard first step for looking at correlation is to generate a pairs plot.

# In[ ]:


fig = plt.figure(figsize = (15, 8))
cols = ['pclass', 'age', 'fare', 'survived']
colors = g_colors
pd.plotting.scatter_matrix(train_raw[cols],
                           figsize=[13,13],
                           alpha=0.2,
                           c = train_raw.survived.apply(lambda x:colors[x]))

plt.tight_layout()


# ## 3) What Pairs Of Features Have Interesting Survival Relations?
# First let's look at how various features look when compared to **age**.

# In[ ]:


def show_dot_plot(p_x, p_y, p_ax):
    
    sns.stripplot(x = p_x,
                  y = p_y,
                  data = x_train,
                  jitter = True,
                  alpha = .7,
                  hue = 'survived',
                  palette = g_colors,
                  ax = p_ax)
    
    return

fig, axes = plt.subplots(2, 3, figsize=(15,8))

cols = ['pclass', 'sex', 'embarked', 'sibsp', 'parch']

for k, col in enumerate(cols):
    show_dot_plot(p_x = col,
                  p_y = 'age',
                  p_ax = axes.ravel()[k])
    
fig.delaxes(axes[1][2])
z = plt.suptitle('Features By Age', fontsize = 24)


# We saw this in a previous chart but there appears to have been few large families on the Titanic and they did not have a high survival rate.
# 
# How about when compared to fare? Remembering that the data was highly skewed let's just look at fares below $50 dollars.

# In[ ]:


def show_dot_plot(p_x, p_y, p_ax):
    
    sns.stripplot(x = p_x,
                  y = p_y,
                  data = x_train[x_train.fare < 100],
                  jitter = True,
                  alpha = .7,
                  hue = 'survived',
                  palette = g_colors,
                  ax = p_ax)
    
    return

fig, axes = plt.subplots(2, 3, figsize=(15,8))

cols = ['pclass', 'sex', 'embarked', 'sibsp', 'parch']

for k, col in enumerate(cols):
    show_dot_plot(p_x = col,
                  p_y = 'fare',
                  p_ax = axes.ravel()[k])

fig.delaxes(axes[1][2])
    
z = plt.suptitle('Features By Fare (Less Than $100)', fontsize = 24)


# Even without the higher fared datapoints, it's clear that red sinks to the bottom of these charts, green rises to the top. Higher fares made a difference.
# 
# A very interesting item that I didn't notice while doing my original analysis: 1st class does not necessarily imply a super high fare. There are a number of 1st class passengers who paid only $30.

# One final comparison: Survival by **sex** compared against **pclass** and **embarked**.

# In[ ]:


eat = sns.factorplot(x = "pclass",
                     y = "survived",
                     data = x_train,
                     hue = 'sex',
                     col = 'embarked',
                     palette = [sns.xkcd_rgb['medium blue'], sns.xkcd_rgb['carmine']])


# I really like this chart and it shows a distinct difference in third class female survival for those boarding at Southampton. I'm not sure what to make of that but it clearly shows the relevance of including **embarked** in the final analysis.

# ## 4) What Additional Features Appear To Be Worth Engineering?
# It's easy to get lost in data exploration but it's time to move onto the feature engineering phase. We'll do additional visualizations as new features are created.
# 
# For now, what candidates do we have for new features?
# * We deferred tackling **name** and **cabin** until later. Both fields have some information that we can extract.
# * **sibsp** and **parch** could be combined into an overall family size value.
# * Based on what we saw in the **age** / **survived** distribution, perhaps dummy variables for each of the noteworthy age ranges would be worth creating.
# * Speaking of dummy variables, each of the categorical variables will be turned into dummy variables. This includes **pclass**, **sex** and **embarked**. Perhaps **sibsp** and **parch**.

# # Feature Engineering / Data Munging
# Time to start putting together our working dataset. Based on the exploratory phase, we have a long list of items to address:
# 1. Derive feature(s) out of the **name** field.
# 1. Derive feature(s) out of the **cabin** field.
# 1. Derive feature(s) out of the **ticket** field.
# 1. Derive family size field.
# 1. Fill in missing **embarked** values.
# 1. Create dummy variables where appropriate.
# 1. Estimate missing **age** values.
# 1. Generate bins for **age** ranges based on chart insights.
# 1. Apply scaling where appropriate.
# 1. Estimate missing **fare** values.

# ## A Note On My Personal Workflow
# In an effort to be as DRY (Don't Repeat Yourself) as possible, I try to encapsulate as much functionality as I can into reusable functions. Feature generation is a perfect candidate for this because it includes a ton of logic and whatever we decide to do with x_train, we will need to perform the exact same operations to x_test and x_eval.
# 
# I find the easiest way to address this is to create a single function called tidy_data() function to do all data transformations.

# In[ ]:


def tidy_data(df):
    data = df.copy()
    
    # Perform all data cleaning / feature engineering against data
    
    return data


# Once complete, a simple function call is all that will be required to prepare a dataset for modeling regardless of what phase of data I am working with.
# 
# For the sake of brevity and readability, I will walk through the data munging phase in this kernel using straight python manipulation on "x_train_tidy."

# In[ ]:


x_train_tidy = x_train.copy()


# I will finish by providing a final working version of the tidy_data() function.

# ## 1) Derive Features From Name
# A quick and dirty feature we can consider for **name** is its length.

# In[ ]:


x_train_tidy['name_length'] = x_train_tidy.name.str.len()

fig, axes = plt.subplots(1, 1, figsize = (15, 4))
axes.set_xticks(range(0, 100, 10))

for i in range(0, 2):
    sns.distplot(x_train_tidy.loc[(x_train.survived == i), 'name_length'],
                 color = g_colors[i],
                 hist = False,
                 ax = axes)

axes.set_title('Survival By Name Length')
eat = axes.set_xlim(x_train_tidy.name_length.min(), x_train_tidy.name_length.max())


# Definitely appears to be some insight hidden in that seemingly superficial piece of data. I'll be honest, this is one of the new discoveries I made while writing this kernel. I did not initially consider name length as a new feature but it looks like it will be very informative.
# 
# Now let's turn to the actual content of **name**. The piece of information that stands out is the title that many names have. So let's extract a list of them.

# In[ ]:


x_train_tidy['title'] = x_train_tidy.name.str.extract(' ([A-Za-z]+)\.').str.lower()
pd.DataFrame(x_train_tidy.title.value_counts())


# We'll convert the french pronouns into Mrs and Miss respectively and let's see what happens if we group the infrequent male titles into a "rare" category.
# 
# And remember, we don't know what data could be lurking in x_test or x_eval. So after these conversions have been made, let's blank out any title that is not Mr/Mrs/Miss/Master just to be safe.

# In[ ]:


mrs = ['mme', 'lady', 'countess']
miss = ['mlle']
rare = ['dr', 'rev', 'col', 'capt', 'major', 'don']

x_train_tidy.loc[x_train_tidy.title.isin(mrs), 'title'] = 'mrs'
x_train_tidy.loc[x_train_tidy.title.isin(miss), 'title'] = 'miss'
x_train_tidy.loc[x_train_tidy.title.isin(rare), 'title'] = 'rare'

x_train_tidy.loc[~x_train_tidy.title.isin(['mr', 'mrs', 'miss', 'master', 'rare']), 'title'] = ''

pd.DataFrame(x_train_tidy.title.value_counts())


# Let's take a look at how the title groupings look.

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(12,6))

g = sns.stripplot(x = 'title',
                  y = 'age',
                  data = x_train_tidy,
                  jitter = True,
                  alpha = .7,
                  hue = 'survived',
                  palette = g_colors)

eat = g.set_title('Age By Title')


# I'm genuinely surprised that the rare titles didn't fare better given the assumption of their elevated social status. Maybe the title implied a higher level of character where they sacrificed themselves in trying to help others.
# 
# Perhaps any correlation between title and **survived** will be captured by the **sex** column. But title should be useful for estimating missing **age** values.

# ## 2) Derive Features From Cabin
# It looks like we can extract the cabin floor from the **cabin** column. In the case where **cabin** has multiple entries, I'm just going to take the first letter.
# 
# With some look-ahead insight, I'm going to fill in empty cabins with a dummy value of 'z.'
# 
# And remember to lower case.

# In[ ]:


x_train_tidy['cabin_floor'] = x_train_tidy.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()
x_train_tidy.loc[x_train_tidy.cabin_floor.isnull(), 'cabin_floor'] = 'z'

fig, axes = plt.subplots(1, 1, figsize=(13,6))

g = sns.stripplot(x = 'cabin_floor',
                  y = 'fare',
                  data = x_train_tidy,
                  jitter = True,
                  alpha = .7,
                  hue = 'survived',
                  palette = g_colors)

g.set_ylim(0, 300)

eat = g.set_title('Fare By Cabin Floor (High Fare Outliers Excluded)')


# We can see the value of the dummy variable as "no cabin" is definitely correlated with a lower fare and lower survival.  
# All valid cabin floors have a lot of green, aside from cabin_floor c, and we've seen that higher fares = higher survival.

# ## 3) Derive Features From Ticket
# We can derive two features from the **ticket** field: The alpha part and the numeric part.  

# In[ ]:


x_train_tidy['ticket_alpha'] = x_train_tidy.ticket.str.extract('([A-Za-z\.\/]+)').str.replace('\.', '').str.lower()
x_train_tidy['ticket_num'] = x_train_tidy.ticket.str.extract('([0-9\.\/]+)').str.replace('\.', '')
pd.DataFrame(x_train_tidy.ticket_alpha.value_counts())


# I'll be frank: I ignored this field in my initial analysis and looking at the results now, I'm still not sure how helpful it will be. Let's do a quick plot against fare and survival.

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(13,6))

top_tickets = x_train_tidy.ticket_alpha.value_counts().index.values[:10]

g = sns.stripplot(x = 'ticket_alpha',
                  y = 'fare',
                  data = x_train_tidy.loc[x_train_tidy.ticket_alpha.isin(top_tickets)],
                  jitter = True,
                  alpha = .7,
                  hue = 'survived',
                  palette = g_colors)

g.set_ylim(0, 300)

eat = g.set_title('Fare By Ticket (High fare outliers excluded)')


# Ok, perhaps there is some value in some of the ticket classes. Let's get a bit more exact and look at the correlation with survival.

# In[ ]:


encode = pd.get_dummies(x_train_tidy['ticket_alpha'], prefix = 'ticket')
test = pd.concat([x_train_tidy, encode], axis = 1)
test_corr = test.corr().survived
pd.DataFrame(test_corr[[x for x in test_corr.index.values if 'ticket_' in x]].sort_values())


# None of the correlations seem very high so I am inclined to ignore the ticket fields. (I will still create them, though, in the tidy_data() function in case I have a change of heart.)

# ## 4) Derive Family Size Column
# This will be an easy one. We're just going to add up **sibsp** and **parch** and add one for the person.
# 
# In addition, we're going to create a column representing a large family, where fam_size is 4 or higher.

# In[ ]:


x_train_tidy['fam_size'] = x_train_tidy.sibsp + x_train_tidy.parch + 1
x_train_tidy['fam_size_large'] = (x_train_tidy.fam_size > 3) * 1


# ## 5) Fill In Missing Embarked Column
# **embarked** had only 2 empty values so we can just insert the most common port of Southampton.  
# Also, make **embarked** all lower case as it will be transformed into dummy variables.

# In[ ]:


x_train_tidy.loc[x_train_tidy.embarked.isnull(), 'embarked'] = 'S'
x_train_tidy['embarked'] = x_train_tidy.embarked.str.lower()


# ## 6) Create Dummy Variables
# At this point, we have most of our feature creation done. Below is the full list of columns that we have so far:

# In[ ]:


x_train_tidy.columns


# Now it's time to start putting the resulting data into the best format for modeling. The first step is to create dummy variables for our categorical features.

# ## Intermediate tidy_data()
# 
# Creating dummy variables is going to add a whole bunch of a columns and if we make any error, it's going to be a hassle to debug. The challenge with the Jupyter notebook format (at least on Kaggle) is that you can either run 1 cell or all cells. You can't just run up to the current cell. So getting back to the state where the bug occurred can be a challenge. That's another motivation for encapsulating all data munging work into a function.
# 
# Below is an intermediate version of tidy_data() that includes all of the transformations up until now, including dummy variable generation.

# In[ ]:


def tidy_data(df):
    data = df.copy()
    
    data['name_length'] = data.name.str.len()
    data['title'] = data.name.str.extract(' ([A-Za-z]+)\.')
    mr = ['Rev', 'Dr', 'Col', 'Capt', 'Don', 'Major']
    mrs = ['Mme', 'Countess', 'Lady']
    miss = ['Mlle']

    data.loc[data.title.isin(mr), 'title'] = 'mr'
    data.loc[data.title.isin(mrs), 'title'] = 'mrs'
    data.loc[data.title.isin(miss), 'title'] = 'miss'
    data.loc[data.title=='Master', 'title'] = 'master'
    data.loc[~data.title.isin(['mr', 'mrs', 'miss', 'master']), 'title'] = ''

    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()
    data.loc[data.cabin_floor.isnull(), 'cabin_floor'] = 'z'

    data['ticket_alpha'] = data.ticket.str.extract('([A-Za-z\.\/]+)').str.replace('\.', '').str.lower()
    data['ticket_num'] = data.ticket.str.extract('([0-9\.\/]+)').str.replace('\.', '')

    data['fam_size'] = data.sibsp + data.parch + 1

    data.loc[data.embarked.isnull(), 'embarked'] = 'S'
    data['embarked'] = data.embarked.str.lower()

    for col in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'title', 'cabin_floor', 'fam_size']:
        encode = pd.get_dummies(data[col], prefix = col)
        data = pd.concat([data, encode], axis = 1)
    
    return data


# Now debugging or adding new transformations will be a lot more convenient than running an entire notebook. And no additional work will be required to prepare the test and eval data for modeling.
# 
# Below we can see the expanded list of columns.

# In[ ]:


x_train_tidy = tidy_data(x_train)
x_train_tidy.columns


# ## 7) Estimate Missing Age Values
# We've seen that **age** is highly correlated with **survived** but we have 177 missing **age** values. So we need to estimate these missing values and the more effectively we can model **age** the more accurate our final results will be.
# 
# Let's look at the correlation results for age against our full tidy dataset.

# In[ ]:


age_corr = x_train_tidy.corr().age
vals = age_corr[[x for x in age_corr.index.values if x != 'age']]
pd.DataFrame(vals[abs(vals)>.25].sort_values())


# Let's try including just the dummy variables in a linear regression. I'll create a function to simplify the use of it.

# In[ ]:


def model_age(df):
    formula = 'age ~ title_master + pclass_3 + parch_2 + parch_0 + pclass_1'

    model = sm.ols(formula, data = x_train_tidy[x_train_tidy.age.notnull()]).fit()

    age_pred = model.predict(df)
    
    return np.ceil(age_pred)


# Looks promising but we have a very serious problem. The model must be built using data from the x_train split. But the goal is to create a tidy_data() function that works against train, test and eval data. So we have a bit of a circular dependency here as we cannot use a model that references features created in x_train's tidy_data result because it may not exist.
# 
# So we have two options:
# 1. We can avoid using the dummy variables that we determined have the highest correlation and just use the raw data columns, i.e. **parch** and **pclass**.
# 1. We can write a function that is independent of the tidy_data result, essentially recreate the specific dummy variables in the model_age() function.
# 
# After taking a look at the resulting differences between the raw vs dummy columns, I think it's worth duplicating some code in the age model function.

# In[ ]:


def model_age_correct(df):
    dt = x_train.copy()
    
    dt['title_master'] = dt.name.str.contains('Master') * 1
    for col in ['pclass', 'parch']:
        encode = pd.get_dummies(dt[col], prefix = col)
        dt = pd.concat([dt, encode], axis = 1)

    formula = 'age ~ title_master + pclass_3 + parch_2 + parch_0 + pclass_1'

    model = sm.ols(formula, data = dt[dt.age.notnull()]).fit()

    age_pred = model.predict(df)
    
    return np.ceil(age_pred)


# Not too much work and now the age_model is constrained to the x_train dataset.
# 
# This situation really underscores the value of coming up with a clearly delineated data flow that maintains prior state. It may only be relevant once or twice during a project but it avoids a lot of backtracking when you run into a situation like this.
# 
# Let's use our new model function to fill in missing **age** values.

# In[ ]:


x_train_tidy['age_gen'] = x_train_tidy.age
x_train_tidy.loc[x_train_tidy.age_gen.isnull(), 'age_gen'] = model_age_correct(x_train_tidy[x_train_tidy.age_gen.isnull()])


# Notice that I preserve the original **age** column. As always, I prefer additive processes at all times to make it easy to review / doublecheck my work.

# ## 8) Create Age-Based Bins
# Remember back when we visualized the survival rates by age? There were clear ranges where survival was more or less likely. Let's use that chart to come up with bins corresponding to those ranges. To avoid scrolling back, below is the chart in question.
# 
# [And just out of curiosity, let's plot both the age and the age_gen columns.]

# In[ ]:


fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 4))
axes[0].set_xticks(range(0, 100, 10))

for i in range(0, 2):
    var_survived = i % 2
    
    sns.distplot(x_train_tidy.loc[(x_train_tidy.age.notnull()) & (x_train_tidy.survived == var_survived), 'age'], 
                 color = g_colors[var_survived], 
                 hist = False, 
                 ax = axes[0])

    sns.distplot(x_train_tidy.loc[(x_train_tidy.survived == var_survived), 'age_gen'], 
                 color = g_colors[var_survived], 
                 hist = False, 
                 ax = axes[1])

axes[0].set_title('Survival By Age')
axes[1].set_title('Survival By Age (Derived)')
axes[0].set_xlim(x_train_tidy.age.min(), x_train_tidy.age.max())
eat = axes[1].set_xlim(x_train_tidy.age_gen.min(), x_train_tidy.age_gen.max())


# We definitely impacted the density plot by adding a lot of people in the 20-something **age** range. That's the reality of imputing missing values. Revisiting our **age** estimation may be justified later on.
# 
# In regards to creating **age** bins, both charts support the idea of creating bins for the ranges 0 - 14 and 14 - 32. So we'll create 3 bins.

# In[ ]:


bins = [0, 14, 32, 99]

x_train_tidy['age_bin'] = pd.cut(x_train_tidy['age_gen'], bins, labels = ['age_0_14', 'age_14_32', 'age_32_99'])
enc_age = pd.get_dummies(x_train_tidy.age_bin)
x_train_tidy = pd.concat([x_train_tidy, enc_age], axis = 1)


# ## 9) Estimate Missing Fare Values
# Ok, we technically have zero missing **fare** values in our training set but I know that we have a few missing **fare** values in the eval data. So to simplify this kernel, let's address this now by creating a model_fare() function. 

# In[ ]:


age_corr = x_train_tidy.corr().fare
vals = age_corr[[x for x in age_corr.index.values if x != 'fare']]
pd.DataFrame(vals[abs(vals) > .2].sort_values())


# In[ ]:


def model_fare(df):
    data = x_train.copy()
    
    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()
    data.loc[data.cabin_floor.isnull(), 'cabin_floor'] = 'z'
    
    data['is_alone'] = ((data.sibsp + data.parch) == 0) * 1
    
    for col in ['pclass', 'cabin_floor', 'parch']:
        encode = pd.get_dummies(data[col], prefix = col)
        data = pd.concat([data, encode], axis = 1)

    formula = 'fare ~ pclass_1 + pclass_3 + is_alone + cabin_floor_b + cabin_floor_c + cabin_floor_z'

    model = sm.ols(formula, data = data[data.fare.notnull()]).fit()

    fare_pred = model.predict(df)
    
    return round(fare_pred, 2)


# This may be overkill given the small number of missing **fare** values. I probably could have just used the median value.

# ## 10) Apply Scaling Where Appropriate
# Many machine learning algorithms prefer scaled continuous data so let's create scaled versions of the three main continuous features that we have: **age**, **fare** and name_length.
# 
# Given this is the final transformation, I will include the code in the final version of the tidy_data function.

# In[ ]:


def tidy_data(df):
    data = df.copy()
    
    data['name_length'] = data.name.str.len()
    data['title'] = data.name.str.extract(' ([A-Za-z]+)\.')
    mr = ['Rev', 'Dr', 'Col', 'Capt', 'Don', 'Major']
    mrs = ['Mme', 'Countess', 'Lady']
    miss = ['Mlle']

    data.loc[data.title.isin(mr), 'title'] = 'mr'
    data.loc[data.title.isin(mrs), 'title'] = 'mrs'
    data.loc[data.title.isin(miss), 'title'] = 'miss'
    data.loc[data.title=='Master', 'title'] = 'master'
    data.loc[~data.title.isin(['mr', 'mrs', 'miss', 'master']), 'title'] = ''

    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()
    data.loc[data.cabin_floor.isnull(), 'cabin_floor'] = 'z'

    data['ticket_alpha'] = data.ticket.str.extract('([A-Za-z\.\/]+)').str.replace('\.', '').str.lower()
    data['ticket_num'] = data.ticket.str.extract('([0-9\.\/]+)').str.replace('\.', '')

    data['fam_size'] = data.sibsp + data.parch
    data['is_alone'] = ((data.sibsp + data.parch) == 0) * 1

    data.loc[data.embarked.isnull(), 'embarked'] = 'S'
    data['embarked'] = data.embarked.str.lower()

    for col in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'title', 'cabin_floor', 'fam_size']:
        encode = pd.get_dummies(data[col], prefix = col)
        data = pd.concat([data, encode], axis = 1)

    data['fare_gen'] = data.fare
    data.loc[(data.fare == 0) | (data.fare.isnull()), 'fare_gen'] = model_fare(data[(data.fare == 0) | (data.fare.isnull())])

    data['age_gen'] = data.age
    data.loc[data.age_gen.isnull(), 'age_gen'] = model_age_correct(data[data.age_gen.isnull()])

    bins = [0, 14, 32, 99]

    data['age_bin'] = pd.cut(data['age_gen'], bins, labels = ['age_0_14', 'age_14_32', 'age_32_99'])
    enc_age = pd.get_dummies(data.age_bin)
    data = pd.concat([data, enc_age], axis = 1)

    std_scale = StandardScaler().fit(data[['age_gen', 'fare_gen', 'name_length']])
    data['age_scaled'] = 0
    data['fare_scaled'] = 0
    data['name_scaled'] = 0
    data[['age_scaled', 'fare_scaled', 'name_scaled']] = std_scale.transform(data[['age_gen', 'fare_gen', 'name_length']])

    return data


# That was a lot of work but now we have a lot of features to choose from as we start modeling.

# In[ ]:


x_train_tidy = tidy_data(x_train)
x_test_tidy = tidy_data(x_test)
x_train_tidy.columns


# # Modeling
# A whole lot of work and exploration has resulted in ~70 features to be used for modeling. Rather than rattle through an exhaustive list of models looking for lightning in a bottle, I'm going to choose three and focus more on feature selection and hyperparameter tuning.
# 
# The three models will be:
# * Logistic Regression
# * Random Forest Classifier
# * Gradient Boost Machines
# 
# Using these three models, below are the remaining steps we're going to take before we finally submit our results.
# 1. Feature Management
# 1. Feature Synchronization
# 1. Initial Test Run Of Models
# 1. Backward Feature Elimination
# 1. Cross-Validation
# 
# This section is a lot more function-heavy than the data exploration phase. I worked hard to make the code as flexible and reusable as possible to allow easy execution of new models with differing feature sets and hyper parameters. In addition, I rolled my own Backward Feature Elimination and Cross-Validation functions even though I'm sure that sklearn has versions of both.

# # 1) Feature Management
# We have a large number of features and I've purposefully kept them around because data modeling is not a straight line. We're going to want to try out various ideas, different sets of features. If we just drop columns from our dataset along the way, trying to unwind those actions can be a real nuisance.
# 
# Subsetting just the columns we want to use is very easy. But it's worth putting some thought into how best to do it.
# 
# The convention I initially used was to create a constant called COLUMNS_TO_MODEL and selectively comment out what I want to exclude.

# In[ ]:


COLUMNS_TO_MODEL = [#'passengerid', 'survived', 
    #'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked',
    #'name_length',
    #'title',
    #'cabin_floor',
    #'ticket_alpha',
    #'ticket_num',
    #'fam_size',
    #'fam_size_large',
    #'is_alone',
    'pclass_1', 'pclass_2', 'pclass_3',
    'sex_female',
    'sex_male',
    #'sibsp_0', 'sibsp_1', 'sibsp_2', 'sibsp_3', 'sibsp_4', 'sibsp_5', 'sibsp_8',
    #'parch_0', 'parch_1', 'parch_2', 'parch_3', 'parch_4', 'parch_5', 'parch_6',
    'embarked_c', #'embarked_q', 
    'embarked_s',
    #'title_',
    'title_master', 'title_miss', 'title_mr', 'title_mrs',
    #'cabin_floor_a',
    'cabin_floor_b',
    'cabin_floor_c',
    'cabin_floor_d',
    'cabin_floor_e',
    #'cabin_floor_f',
    #'cabin_floor_g',
    'cabin_floor_z',
    'fam_size_0', 'fam_size_1', 'fam_size_2', 'fam_size_3', #'fam_size_4', 'fam_size_5', 'fam_size_6', 'fam_size_7', 'fam_size_10', 
    #'fare_gen',
    #'age_gen',
    #'age_bin',
    'age_0_14', 'age_14_32', #'age_32_99',
    #'age_scaled',
    'fare_scaled',
    'name_scaled']


# Yea, it's a long, tall list. (I use vertical monitors so I don't mind...) But this approach consolidates feature selection into one single location and really simplifies the process. You can just comment out what you want to exclude.
# 
# This approach is convenient and flexible and works really well in a script-based environment like Spyder, but to be frank, it doesn't scale very well. In the context of a Jupyter notebook, this approach requires making new copies everytime one wants to make an adjustment. Or go back and modify/run a prior cell. That's awkward. The other downside of this approach, even when working within Spyder, is it's very easy to forget what features were used on a given model that just so happened to be that high scoring submission. (Guilty as charged.)
# 
# A better approach is to create a csv file to track all of the different iterations of feature selection. 
# 
# I've gone ahead and uploaded features.csv to this kernel which includes two feature sets:
# * all (Excluding obvious columns that should not be present like passengerid or columns that are problematic)
# * baseline (An initial set of manually selected features.)
# 
# I intended to include more sets but Kaggle's dataset versioning is a little wonky so adding new versions don't immediately appear available inside a kernel. TBD
# 
# The following function simplifies the retrieval of feature sets.

# In[ ]:


def get_feature_set(name):
    features = pd.read_csv('../input/titanic-features/features.csv')
    
    return features.loc[features[name]==1, 'feature'].values

get_feature_set('baseline')


# The "baseline" set of features was chosen based on what we've learned from our data exploration phase as well as the following correlation metrics.

# In[ ]:


corr = x_train_tidy.corr().survived
pd.DataFrame(corr[abs(corr) > .1])


# The following is a summary of the decisions that went into selecting the baseline feature set.
# * Remove any raw features that have been turned into dummy features.
# * Remove columns that have been scaled.
# * Remove the **sibsp** and **parch** dummy variables in favor of fam_size [0-3].
# * Remove **ticket**-based features.
# * Remove is_alone since it should be captured by fam_size_0.
# * Remove embarked_q, cabin_floor_a/f/g, age_scaled and age_32_99 based on low correlation.

# # 2) Feature Synchronization
# We have one last issue to address: Whatever features we use for modeling must exist in the test and eval datasets or we will generate an error. Dummy variables can cause column mismatches if certain records/values don't appear in the test or eval set.
# 
# For example, the x_test set could very well not include a record for a cabin on floor B. This can happen simply by the random nature of the train_test_split function or legitimate differences in the eval data.
# 
# Therefore, we should resolve these discrepancies before modeling begins. We have two options:
# * We can filter down the datasets to the intersection of columns between all of the datasets.
# * We can add the missing features into the test or eval sets as needed.
# 
# The first approach requires removing what might be a useful feature so I sort of prefer the second approach as shown below.

# In[ ]:


def add_missing_columns(df, features):
    missing = [col for col in features if col not in df.columns]
    
    for x in missing:
        df[x] = 0
    
    return


# Now we can be confident that the train, test and eval datasets will share the same set of columns.
# 
# To streamline feature subsetting and synchronization, let's create a function to take care of everything required to subset the tidy versions of our data (whether train, test or eval).

# In[ ]:


def get_thin_data(df, features):

    add_missing_columns(df, features)

    df_thin = df[features]

    df_thin = df_thin[df_thin.columns[~(df_thin.dtypes == 'object')]]
    
    return df_thin


# In[ ]:


get_thin_data(x_train_tidy, get_feature_set('all')).columns


# Now we can be confident that the train, test and eval datasets will share the same set of columns.

# # 3) Modeling
# Now we're ready to start modeling. First we'll create two helper functions:
# 1. Run the model and capture the results
# 1. Display and visualize the results.

# In[ ]:


model_results = []

def run_model(mod, features, feature_set = '', back_elim = 'N', cross_val = 'N'):
    
    x_train_thin = get_thin_data(x_train_tidy, features)
    x_test_thin = get_thin_data(x_test_tidy, features)
        
    mod.fit(x_train_thin, y_train)

    y_train_pred = mod.predict(x_train_thin)
    y_test_pred = mod.predict(x_test_thin)
    
    output = (mod.__class__.__name__, 
              accuracy_score(y_train, y_train_pred), 
              accuracy_score(y_test, y_test_pred),
              feature_set,
              back_elim,
              cross_val)
    
    model_results.append(output)
    
    return


# In[ ]:


def show_model_results():
    df = pd.DataFrame(model_results).reset_index(drop = True)
    df.columns = ['model', 'training', 'test', 'feature_set', 'back_elim', 'cross_val']
    #df = df.sort_values(['test'], ascending = False)
    
    fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 5))

    for i in range(0, 2):
        sns.barplot(y = df.feature_set + '|' + df.back_elim + '|' + df.cross_val + ' | '+ df.model,
                    x = df[['training', 'test'][i]],
                    ax = axes[i], 
                    palette = sns.color_palette("deep", 20))

        axes[i].set_ylabel('')
        axes[i].set_xlabel('Accuracy')
        axes[i].set_title(['training', 'test'][i].title() + ' Set Results')
        
    axes[0].set_xlim((.70, 1))
    axes[1].set_xlim((.70, .85))
    
    return df


# Let's start off with Logistic Regression and Random Forests against the full feature set using default parameters.

# In[ ]:


lr = LogisticRegression(random_state = 1)
run_model(lr, get_feature_set('all'), 'all')

rf = RandomForestClassifier(random_state = 1)
run_model(rf, get_feature_set('all'), 'all')


# In[ ]:


show_model_results()


# Even though we have some significant overfitting with Random Forests, the results for the test set are between 75% and 80% which is above the baseline of **68% accuracy** that we identified at the start of this kernel.
# 
# We have a lot of opportunity to improve. First let's try to reduce our feature set.

# # 4) Backward Feature Elimination
# Backward Feature Elimination works by running a model with the full set of features. It then removes the least important feature(s) and reruns the model until a minimum number of features is reached. An accuracy metric is captured for every one of these runs and the final output includes the set of features that corresponds to the run with the highest accuracy.
# 
# I've created two functions:
# 1. The first runs the Backward Feature Elimination process to determine a reduced subset of features.
# 1. The other wraps the first function, collects the final feature set and runs the model with it.
# 
# This results in a single line of code being able to run any model against any feature subset from the features.csv file.

# In[ ]:


def backward_feature_elimination(model, x, y):
    
    bwd_results = []
    
    features = x.columns

    while len(features) > 4:
        model.fit(x[features], y)

        score = accuracy_score(y, model.predict(x[features]))
        
        #   Retrieve feature importance and store values
        values = dict(zip(features, model.feature_importances_))
        values['score'] = score
        bwd_results.append(values)
        
        #   Eliminate feature
        low_import = min(values.values())
        
        features = [k for k, v in values.items() if (v > low_import) & (k != 'score')]

    bwd_features = pd.DataFrame(bwd_results)
    
    # Identify the row with the highest accuracy but reverse array order
    # so the smaller feature set is returned in the case of a tie
    best = bwd_features[bwd_features.index == np.argmax(bwd_features.score[::-1])]
    
    best_result = best.T.reset_index()
    best_result.columns = ['feature', 'importance']
    best_result = best_result[(best_result.importance.notnull()) & (best_result.feature != 'score')].sort_values(['importance'], ascending = False)

    return best.columns[(best.notnull().values.ravel()) & (best.columns != 'score')].values, best_result, bwd_features


# In[ ]:


def run_backward_elimination(model, feature_set):
    x_train_thin = get_thin_data(x_train_tidy, get_feature_set(feature_set))
    x_test_thin = get_thin_data(x_test_tidy, get_feature_set(feature_set))

    feat, res, results = backward_feature_elimination(model, x_train_thin, y_train)
    
    fig, axes = plt.subplots(1, 1, figsize = (12, math.ceil(len(res) / 3.5)))

    eat = sns.barplot(y = res.feature,
                      x = res.importance,
                      color = 'blue',
                      ax = axes)
    
    run_model(model, feat, feature_set, back_elim = 'Y')
    
    return feat


# Let's see how Backward Feature Elimination works for Random Forests against the full set of features.

# In[ ]:


features_all_back_rf = run_backward_elimination(rf, 'all')


# We've gone from 62 features to 27. The top listed features all match the insights we gained during the data exploration phase. Notice the previously ignored name_length is second most important.

# In[ ]:


show_model_results()


# Hmm, the model actually performs worse on both the training and test set. This is a great illustration of how less is more and we shouldn't throw all of our features into a model.
# 
# Let's try again using the hand-selected baseline feature set.

# In[ ]:


run_model(lr, get_feature_set('baseline'), 'baseline')

run_model(rf, get_feature_set('baseline'), 'baseline')
features_base_back_rf = run_backward_elimination(rf, 'baseline')


# Backward Feature Elimination reduces the baseline feature set from 24 features to 17.

# In[ ]:


show_model_results()


# The smaller set of features has helped to regularize the Random Forest Classifier. It's still below the performance of Logistic Regression for the test set, though.
# 
# Let's see what we can get when we use Gradient Boosting Machines.

# In[ ]:


gbm = GradientBoostingClassifier(random_state = 1)

run_model(gbm, get_feature_set('all'), 'all')
run_model(gbm, get_feature_set('baseline'), 'baseline')

features_all_back_gbm = run_backward_elimination(gbm, 'all')
features_base_back_gbm = run_backward_elimination(gbm, 'baseline')


# In[ ]:


show_model_results()


# Some mixed results with BGMs highlights the importance of selected features carefully. I'm not going to go further down this road although the framework is well positioned for new models to be used and new feature subsets to be retrieved from features.csv.
# 
# Next, I'm going to dig into how we can use different model parameters to improve our performance.

# # 5) Cross-Validation
# Each of the models have various parameters that allow us to apply regularization, essentially sacrificing training-set accuracy in favor of test-set accuracy. While we can try to guess which parameters and values to use, a better approach is to just try a whole bunch and select which ones work the best.
# 
# Again, we have two functions: One loops through a given parameter and its values, the second function is the driver that manages multiple parameters and captures the results.
# 

# In[ ]:


def cross_validate(model, x, y, param, n_range, metric = 'accuracy'):
    
    all_scores = []
    
    for val in n_range:
        
        setattr(model, param, val)
        scores = cross_val_score(model, x, y, cv = 5, scoring = metric)
        all_scores.append((param, val, scores.mean(), scores.std(), scores.min(), scores.max()))
    
    return pd.DataFrame(all_scores, columns = ['param', 'value', 'mean', 'std', 'min', 'max'])

def cross_all(model, x, y, params, metric = 'accuracy'):
    
    while len(params) > 0:
        
        cv_best = pd.DataFrame()
        cv_results = {}
    
        for k in params:
            results = cross_validate(model, x, y, k, params[k])
            
            cv_results[k] = results
            
            cv_best = cv_best.append(results[results.index==results['mean'].idxmax()])
        
        cv_best.reset_index(inplace = True)
        
        #   Select best param and set in model. Remove from params
        param, value = cv_best.loc[cv_best.index == cv_best['mean'].idxmax(), ['param', 'value']].values.ravel()
        
        if param in ('max_features', 'n_estimators'):
            value = int(value)
            
        print(cv_best)
        print(param)
        print(value)
        
        setattr(model, param, value)
        
        del params[param]
        
    return model


# I'm going to use GBM Cross Validation against two sets of features: Our baseline set and the reduced set obtained from Backward Feature Elimination.

# In[ ]:


mod_gbm = GradientBoostingClassifier()

cross_thin = get_thin_data(x_train_tidy, features_base_back_gbm)

params = {'n_estimators': range(20, 150, 10),
          'learning_rate': np.arange(.02, .1, .01),
          'subsample': np.arange(7, 10.1, .5) / 10,
          'max_features': range(4, len(cross_thin.columns)),
          'max_depth': range(2, 10)}

mod_gbm = cross_all(mod_gbm, cross_thin, y_train, params)

run_model(mod_gbm, features_base_back_gbm, 'baseline', back_elim = 'Y', cross_val = 'Y')


# In[ ]:


mod_gbm = GradientBoostingClassifier()

cross_thin = get_thin_data(x_train_tidy, get_feature_set('baseline'))

params = {'n_estimators': range(20, 150, 10),
          'learning_rate': np.arange(.02, .1, .01),
          'subsample': np.arange(7, 10.1, .5) / 10,
          'max_features': range(4, len(cross_thin.columns)),
          'max_depth': range(2, 10)}

mod_gbm = cross_all(mod_gbm, cross_thin, y_train, params)

run_model(mod_gbm, get_feature_set('baseline'), 'baseline', back_elim = 'N', cross_val = 'Y')


# In[ ]:


show_model_results()


# Interestingly, cross-validation doesn't seem to have benefited GBM. Let's try Random Forests.

# In[ ]:


mod_rf = RandomForestClassifier()

cross_thin = get_thin_data(x_train_tidy, features_base_back_rf)

params = {'n_estimators': range(20, 150, 10),
          'max_features': range(4, len(cross_thin.columns))}

mod_rf = cross_all(mod_rf, cross_thin, y_train, params)

run_model(mod_rf, features_base_back_rf, 'baseline', back_elim = 'Y', cross_val = 'Y')


# In[ ]:


mod_rf = RandomForestClassifier()

cross_thin = get_thin_data(x_train_tidy, get_feature_set('baseline'))

params = {'n_estimators': range(20, 150, 10),
          'max_features': range(4, len(cross_thin.columns))}

mod_rf = cross_all(mod_rf, cross_thin, y_train, params)

run_model(mod_rf, get_feature_set('baseline'), 'baseline', back_elim = 'N', cross_val = 'Y')


# In[ ]:


show_model_results()


# At first glance, it looks like cross-validation was beneficial to Random Forests but look closer and we can see that Random Forests worked best against the baseline feature subset without cross-validation. Having said that, the smaller set of features very well may perform better against the evaluation set.

# # A Note On Performance Tuning
# In the process of trying different models and features and hyperparameters, It is very easy to get lost. There are simply too many permutations involved. That's why the code I've written attempts to be as self-tracking as possible.
# 
# It is 100% worth the investment to set up some kind of framework to capture the details of everything you try so you will be able to re-execute your model using the same features and parameters.
# 
# That is one strength of the features.csv I created. While some feature selection will be done algorthmically, it's inevitable that one will try a manually selected feature set. Ideally, the feature set should either be captured prior to running the model or the model execution function should output the feature set along with other details from the model run.

# # Next Steps
# This has been a long first kernel. But we have a pretty exhaustive exploration of the Titanic dataset and the beginnings of a pretty flexible model execution framework. In the end, I believe that feature subsetting deserves additional scrutiny as well as trying out some additonal models.
# 
# Perhaps I will start a second kernel to drill more deeply into evaluating different models and feature subsets to come up with an optimal combination. But for now, I will end this kernel here.
# 
# Thanks for reading.
