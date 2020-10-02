#!/usr/bin/env python
# coding: utf-8

# This is the Exploratory Data Analysis that I performed on my take on the Titanic data-set and competition. You can find my other notebooks here:
# * [Titanic - Data Treatment](https://www.kaggle.com/mcromao/titanic-data-treatment)
# * [Titanic Cross Validation Scores](https://www.kaggle.com/mcromao/titanic-cross-validation-scores)
# * [Titanic DNN CV Parameter Scan](https://www.kaggle.com/mcromao/titanic-dnn-cv-parameter-scan)
# * [Titanic Model CV Analyser](https://www.kaggle.com/mcromao/titanic-model-cv-analyser)
# * [Titanic - Submission preparation with Voting](https://www.kaggle.com/mcromao/titanic-submission-preparation-with-voting)

# # Packages and Data Imports

# In[4]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.stats as stats

train_full= pd.read_csv('../input/train.csv',index_col='PassengerId')

quantitative = [f for f in train_full.columns if train_full.dtypes[f] != 'object']
quantitative.remove('Survived')#Survived is target label
qualitative = [f for f in train_full.columns if train_full.dtypes[f] == 'object']


# # 1. Exploratory Data Analysis

# In[5]:


train_full.head()


# In[6]:


train_full.info()


# ## 1.1 Preamble

# There are a few things we expect from the data. For example, we know that higher class tickets were sorted to higher level decks, with easier access to life boats, therefore we expect first class to be a relevant feature to assess survival chance. Then, we expect children and women to take priority, as it is customary in popular culture that "women and children first" in emergency cases. This also means that there should some level of relation between survival chance and number of family members on board, as families would have taken prioriy in principal. Obviously, these are subjective expectations, and we will see if they hold.

# In[7]:


sns.distplot(train_full['Survived'],kde=False)


# In[8]:


train_full['Survived'].describe()


# In[9]:


f = pd.melt(train_full, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  sharex=False, sharey=False)
g = g.map(sns.distplot, "value")


# ## 1.2 Target Class Analysis

# We want to see how the target label/class is influenced by the features

# In[10]:


sns.barplot(y='Survived',x='Pclass',data=train_full)


# In[11]:


sns.barplot(y='Survived',x='Sex',data=train_full)


# In[12]:


sns.barplot(y='Survived',x='Sex',hue='Pclass',data=train_full)


# In[13]:


sns.barplot(y='Survived',x='Embarked',data=train_full)


# In[14]:


sns.barplot(y='Survived',x='SibSp',data=train_full)


# In[15]:


sns.barplot(y='Survived',x='Parch',data=train_full)


# In[16]:


sns.boxplot(x='Survived',y='Age',hue='Sex',data=train_full)


# In[17]:


sns.boxplot(x='Survived',y='Fare',hue='Sex',data=train_full)


# From the above we can outline some trends on the segments more likely to survive:
# * First Class
# * Women
# * Age mostly similar, with a slight shift to younger people being more likely to have survived
# * In proportion, people embarking in S were more likely to not have survived
# * Higher fares
# * In  proportion, people with family members seemd to have been more likely to have survived
# 

# With the above plot we confirm some of the previous points. Namely, First class males were at least twice as likely to have survived in comparisson with other classes, but women in second class enjoied a greater change of survivel. We also note that thrid class women were more likely to have survived than first class man.

# In[18]:


plt.figure(figsize=(20, 7))
plt.subplot(1,2,1)
sns.barplot(x="Pclass", y="Survived", hue="SibSp", data=train_full)
plt.subplot(1,2,2)
sns.barplot(x="Pclass", y="Survived", hue="Parch", data=train_full)


# The two plots above show yet another tendency, which was to favour people with family on board . Notice that for each class, the chance of being saved if having family on board is greater than without.This might indicate that surnames might be relevant!

# In[19]:


def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['Survived'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    
features = quantitative
spearman(train_full, features)


# In[20]:


def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():#for each level of the category c
            s = frame[frame[c] == cls]['Survived'].values
            samples.append(s)#Get all the values of SalePrice for the level cls of the categorical variable c, and append
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(train_full)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)


# ## 1.3 Feature Relations Analysis

# In[21]:


sns.boxplot(x='Pclass',y='Age',hue='Sex',data=train_full)


# In[22]:


sns.boxplot(x='Pclass',y='Fare',hue='Sex',data=train_full)


# In[23]:


sns.heatmap(train_full.drop('Survived',axis=1).corr(),cmap=sns.diverging_palette(220, 20, n=10),annot=True,linewidths=0.1)


# Some variables are fairly correlated:
# * SibSp with Parch, possibly indicating presence of families.
# * Pclass and Fare (negative), indicating first class was more expensive, as expected
# * Age (negatively) correlated with Pclass (older people tended to be in first class), SibSp and Parch, meaning the older people seemed to not being with their families

# In[ ]:




