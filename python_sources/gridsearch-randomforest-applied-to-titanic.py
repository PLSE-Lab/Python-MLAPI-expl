#!/usr/bin/env python
# coding: utf-8

# # Predicting Survival on the Titanic
# 
# This kernel combines a gridsearch with a RandomForest model to predict Titanic survivors. I try to take a quantitative approach in the selection of the features that go into the modeling.

# In[177]:


# %% module imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% read in data and modify data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Data Preparation

# ## Data Overview

# In[178]:


train.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], include='all')


# ## Data Completeness

# In[179]:


train.info()


# Training sample:
# * `Age` incomplete: feature missing for a significant fraction of sample
# * `Cabin` highly incomplete: feature missing for the majority of the sample
# * `Embarked` mostly complete: feature missing for an insignificant fraction of the sample
# * `Cabin` mostly incomplete

# In[180]:


test.info()


# Test sample:
# * `Age` incomplete: feature missing for a significant fraction of the sample
# * `Fare` mostly complete: feature missing for a single data point
# * `Cabin` highly incomplete: feature missing for the majority of the sample

# ## Pre-Processing
# 
# The following features are non-numerical, requiring encoding or data extraction:
# * `Name`
# * `Sex`
# * `Ticket`
# * `Cabin`
# * `Embarked`

# ### `Name`

# In[181]:


train.Name.head()


# Finding: every single person has a title attached to their name. This title might come in handy in the classification. We compile a set of all titles that occur either in the `train` or the `test` sample:

# In[182]:


def extract_title(name):
    name = name.split()
    for w in name:
        if '.' in w:
            return w
    return None

all_titles = pd.concat([train, test], sort=False).Name.apply(lambda name: extract_title(name))
all_titles.unique()


# Do these titles carry information on survival?

# In[183]:


train['title'] = train.Name.apply(lambda name: extract_title(name))
test['title'] = test.Name.apply(lambda name: extract_title(name))

train.groupby('title').Survived.agg(['count', 'mean'])


# Yes, some titles (with significant counts) clearly  show a preference for survival, while others are biased against. We replace some of the rare titles and check again:

# In[184]:


titlemap = {'Don.':'noble', 'Rev.':'noble', 'Dr.':'noble', 'Mme.':'Mrs.',
            'Ms.': 'Miss.', 'Major.':'noble', 'Lady.':'noble', 'Sir.':'noble', 'Mlle.':'Miss.', 
            'Col.':'noble', 'Capt.':'noble', 'Countess.':'noble', 'Jonkheer.':'noble', 'Dona.': 'noble'}
train.title.replace(titlemap, inplace=True)
test.title.replace(titlemap, inplace=True)


# In[185]:


train.groupby('title').Survived.agg(['count', 'mean'])


# We encode the different titles:

# In[186]:


from sklearn.preprocessing import LabelEncoder

title_encoder = LabelEncoder().fit(train.title)
train.title = title_encoder.transform(train.title)
test.title = title_encoder.transform(test.title)


# ### `Sex`

# In[187]:


train.Sex.head()


# In[188]:


sex_encoder = LabelEncoder().fit(train.Sex)

train.Sex = sex_encoder.transform(train.Sex)
test.Sex = sex_encoder.transform(test.Sex)


# ### `Age`
# 
# The passenger's age is a potentially important feature - but unfortunately, it is incomplete. However, we can take advantage of `title` and fill in missing ages with the median ages of passengers carrying the same title.

# In[189]:


# combine data from training and test sample 
age_translator = pd.concat([train, test], sort=False).groupby('title').Age.median().to_dict()

# add missing ages to `train` sample
train['completeAge'] = train.apply(lambda x: age_translator[x.title], axis=1)
train.loc[train.Age.notnull(), 'completeAge'] = train.loc[train.Age.notnull(), 'Age']
#train.completeAge, agebins = pd.cut(train.completeAge, bins=np.linspace(0, 90, 15), retbins=True, labels=False)

# add missing ages to `test` sample
test['completeAge'] = test.apply(lambda x: age_translator[x.title], axis=1)
test.loc[test.Age.notna(), 'completeAge'] = test.loc[test.Age.notna(), 'Age']
#test.completeAge = pd.cut(test.completeAge, bins=agebins, labels=False)


# ### `Parch` + `SibSp` = `familysize` + `hasfamily`
# 
# We combine the features `Parch` and `SibSp` into `familysize` as they describe very similar characteristics. In addtion to that, `hasfamily` describes whether an individual has any family on board (1) or is all by themselves (0).

# In[190]:


train['familysize'] = train.Parch + train.SibSp + 1
test['familysize'] = test.Parch + test.SibSp + 1
train.drop(['Parch', 'SibSp'], inplace=True, axis=1)

train['hasfamily'] = 0
train.loc[train.familysize == 1, 'hasfamily'] = 1

test['hasfamily'] = 0
test.loc[test.familysize == 1, 'hasfamily'] = 1


# ### `Ticket`

# In[191]:


train.Ticket.head()


# Strip characters and extract numbers:

# In[192]:


train['ticketno'] = train.Ticket.apply(lambda x: x.split()[-1])
train.ticketno.replace('LINE', "0", inplace=True)  # assign 0 to single non-numeric value
train.ticketno = train['ticketno'].astype(np.int)

test['ticketno'] = test.Ticket.apply(lambda x: x.split()[-1])
test.ticketno = test['ticketno'].astype(np.int)


# ### `Cabin`

# In[193]:


train.Cabin.head()


# Where available, the cabin number provides information on which deck the corresponding person's cabin was located. We extract this information and encode it in the `decklevel` feature:

# In[194]:


train['decklevel'] = train.Cabin.dropna().apply(lambda x: str(x)[0])
test['decklevel'] = test.Cabin.dropna().apply(lambda x: str(x)[0])

decklevel_encoder = LabelEncoder().fit(train.decklevel.dropna())
train.loc[train.decklevel.notna(), 'decklevel'] = decklevel_encoder.transform(train.decklevel.dropna())
test.loc[test.decklevel.notna(), 'decklevel'] = decklevel_encoder.transform(test.decklevel.dropna())


# ### `Embarked`

# In[195]:


# fill missing value with most common label
train.Embarked.fillna('S', inplace=True)

embarked_encoder = LabelEncoder().fit(train.Embarked)

train.Embarked = embarked_encoder.transform(train.Embarked)
test.Embarked = embarked_encoder.transform(test.Embarked)


# In[196]:


## Completing `Fare` in `test`

test.loc[test.Fare.isna(), 'Fare'] = train.Fare.median()


# ### New feature `rich_small_families`

# In[197]:


f, ax = plt.subplots(1, 2, figsize=(10, 5))

train.groupby('Survived').Pclass.plot.hist(alpha=0.5, legend=True, ax=ax[0])
train.groupby('Survived').familysize.plot.hist(alpha=0.5, legend=True, ax=ax[1])


# While first-class passengers have a significantly higher survival rate than third-class passengers, families of size 2-4 people seem to be more likely to survive. We can combine this knowledge into a new feature that I call `rich_small_families`: the value of this feature is low for low-`Pclass` passengers and passenger are part of a family with ~3 members; both groups are likely to survive:

# In[198]:


train['rich_small_families'] = 3*train.Pclass+np.abs(train.familysize-3)
test['rich_small_families'] = 3*test.Pclass+np.abs(test.familysize-3)

train.groupby('Survived').rich_small_families.plot.hist(alpha=0.5, legend=True)


# ## Correlations with `Survived`
# 
# We investigate correlations of all continuous and ordinal features with `Survived` based on the coefficient of determination:

# In[199]:


train.drop(['Survived', 'Embarked', 'title', 'Sex'], axis=1).corrwith(train.Survived).agg('square').plot.bar(title='Coefficient of Determination')


# `PassengerId` plays no role in survival as do the different flavors of `Age`. The latter is somewhat surprising. We will investigate this with a histogram below. The most important features are `Pclass`, `Fare`, `rich_small_families`.
# 
# For the sake of completeness, we plot a heatmap of the pair-wise coefficients of determination for all features.

# In[200]:


import seaborn as sns

f, ax = plt.subplots(figsize=(15,10))
sns.heatmap(train.corr(method='spearman')**2, annot=True, fmt='.2f', cmap='viridis', ax=ax)


# ### Trends among ordinal and categorical features
# 
# We investigate trends between the categorial features and `Survived`:

# In[201]:


train.groupby('Survived').title.plot.hist(alpha=0.5, legend=True)


# In[202]:


train.groupby('Survived').Embarked.plot.hist(alpha=0.5, legend=True)


# In[203]:


train.groupby('Survived').Sex.plot.hist(alpha=0.5, legend=True)


# In[204]:


train.groupby('Survived').completeAge.plot.hist(alpha=0.5, legend=True)


# In[205]:


train.groupby('Survived').hasfamily.plot.hist(alpha=0.5, legend=True)


# All of the plotted features seem to provide some information that is relevant for the survival of an individual passenger.
# 
# This last histogram explains the mystery of the missing correlation between `Age` and `Survival`: the survival rate is highest among young children and lowest among 20-30 yr olds. However, there is no linear trend with age. Hence, the seeming lack of correlation.  

# # Modeling

# In[206]:


featurelist = ['Pclass', 'Fare', 'title', 'Sex', 'Embarked', 'rich_small_families', 'completeAge', 'hasfamily']


# In[207]:


train[featurelist].info()


# In[209]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [40, 50, 60, 70], 
              'max_depth': [7, 10, 13], 
              'max_features': range(3, len(featurelist)),
             'min_samples_leaf': [1, 2, 3],
             'min_samples_split': [7, 10, 12]}

model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, param_grid=parameters, cv=5, n_jobs=-1, scoring='accuracy')
grid.fit(train[featurelist], train.Survived)


# In[210]:


grid.best_params_


# In[211]:


grid.best_score_


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(grid.predict(train[featurelist]), train['Survived']))


# In[ ]:


sns.heatmap(confusion_matrix(grid.predict(train[featurelist]), train['Survived']), annot=True, fmt='d', cmap='viridis')


# # Predicting the `test` Sample

# In[ ]:


result = test
result.loc[:,'Survived'] = grid.predict(test[featurelist])
result.head()


# In[ ]:


result[['PassengerId', 'Survived']].to_csv('submission.csv', header=True, index=False)


# # Changelog
# 
# * v1: using `featurelist = ['Pclass', 'sexcode', 'completeAge', 'decklevelcode', 'ticketno', 'embarkedcode']` resulting in `train` sample  accuracy of 0.85 and public score of 0.76
# * v2: using `featurelist = ['Pclass', 'sexcode', 'SibSp', 'Parch', 'completeAge', 'Fare', 'embarkedcode', 'decklevelcode']` resulting in `train` sample accuracy of 0.84 and public score of 0.75
# * v3: some minor fixes
# * v4: a major rework with a hopefully more quantitative analysis and feature selection: `featurelist = ['Pclass', 'Fare', 'title', 'Sex', 'Embarked', 'rich_small_families', 'completeAge']` resulting in `train` sample accuracy of 0.84 and public score of 0.78.
# * v5: minor changes
# * v6: added polynomial and interaction features: 0.78 `train` sample accuracy
# * v7: removed the polynomial and interaction features again as they don't have a beneficial effect on Decision Tree-based models; fixed some code; added `hasfamily` feature; resulting in `train` sample accuracy of 0.85 

# In[ ]:




