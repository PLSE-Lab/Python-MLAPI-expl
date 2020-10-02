#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Titanic" data-toc-modified-id="Titanic-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Titanic</a></span><ul class="toc-item"><li><span><a href="#Imports-and-configurations" data-toc-modified-id="Imports-and-configurations-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Imports and configurations</a></span></li><li><span><a href="#Data-preprocessing" data-toc-modified-id="Data-preprocessing-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Data preprocessing</a></span><ul class="toc-item"><li><span><a href="#Let's-take-an-overall-look-at-the-data." data-toc-modified-id="Let's-take-an-overall-look-at-the-data.-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Let's take an overall look at the data.</a></span></li><li><span><a href="#PassengerId---Ordinal" data-toc-modified-id="PassengerId---Ordinal-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>PassengerId - Ordinal</a></span></li><li><span><a href="#Pclass---Ordinal" data-toc-modified-id="Pclass---Ordinal-1.2.3"><span class="toc-item-num">1.2.3&nbsp;&nbsp;</span>Pclass - Ordinal</a></span></li><li><span><a href="#Name-and-Title---Categorical" data-toc-modified-id="Name-and-Title---Categorical-1.2.4"><span class="toc-item-num">1.2.4&nbsp;&nbsp;</span>Name and Title - Categorical</a></span></li><li><span><a href="#Sex---Categorical" data-toc-modified-id="Sex---Categorical-1.2.5"><span class="toc-item-num">1.2.5&nbsp;&nbsp;</span>Sex - Categorical</a></span></li><li><span><a href="#Age---Numerical" data-toc-modified-id="Age---Numerical-1.2.6"><span class="toc-item-num">1.2.6&nbsp;&nbsp;</span>Age - Numerical</a></span></li><li><span><a href="#SibSp-and-Parch---Ordinal" data-toc-modified-id="SibSp-and-Parch---Ordinal-1.2.7"><span class="toc-item-num">1.2.7&nbsp;&nbsp;</span>SibSp and Parch - Ordinal</a></span></li><li><span><a href="#Ticket---Categorical" data-toc-modified-id="Ticket---Categorical-1.2.8"><span class="toc-item-num">1.2.8&nbsp;&nbsp;</span>Ticket - Categorical</a></span></li><li><span><a href="#Cabin---Categorical" data-toc-modified-id="Cabin---Categorical-1.2.9"><span class="toc-item-num">1.2.9&nbsp;&nbsp;</span>Cabin - Categorical</a></span></li><li><span><a href="#Fare---Numerical" data-toc-modified-id="Fare---Numerical-1.2.10"><span class="toc-item-num">1.2.10&nbsp;&nbsp;</span>Fare - Numerical</a></span></li><li><span><a href="#Embarked---Categorical" data-toc-modified-id="Embarked---Categorical-1.2.11"><span class="toc-item-num">1.2.11&nbsp;&nbsp;</span>Embarked - Categorical</a></span></li><li><span><a href="#Dealing-with-NaN" data-toc-modified-id="Dealing-with-NaN-1.2.12"><span class="toc-item-num">1.2.12&nbsp;&nbsp;</span>Dealing with NaN</a></span></li><li><span><a href="#Missing-Embarked---Train" data-toc-modified-id="Missing-Embarked---Train-1.2.13"><span class="toc-item-num">1.2.13&nbsp;&nbsp;</span>Missing Embarked - Train</a></span></li><li><span><a href="#Missing-Age---Train-and-Test" data-toc-modified-id="Missing-Age---Train-and-Test-1.2.14"><span class="toc-item-num">1.2.14&nbsp;&nbsp;</span>Missing Age - Train and Test</a></span></li><li><span><a href="#Missing-Fare---Test" data-toc-modified-id="Missing-Fare---Test-1.2.15"><span class="toc-item-num">1.2.15&nbsp;&nbsp;</span>Missing Fare - Test</a></span></li><li><span><a href="#Missing-Cabin---Train-and-Test" data-toc-modified-id="Missing-Cabin---Train-and-Test-1.2.16"><span class="toc-item-num">1.2.16&nbsp;&nbsp;</span>Missing Cabin - Train and Test</a></span></li></ul></li><li><span><a href="#Feature-enginnering" data-toc-modified-id="Feature-enginnering-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Feature enginnering</a></span><ul class="toc-item"><li><span><a href="#Creating-new-label-for-missing-age" data-toc-modified-id="Creating-new-label-for-missing-age-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Creating new label for missing age</a></span></li><li><span><a href="#Create-new-label-for-the-title" data-toc-modified-id="Create-new-label-for-the-title-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Create new label for the title</a></span></li><li><span><a href="#One-hot-encoding-for-Sex" data-toc-modified-id="One-hot-encoding-for-Sex-1.3.3"><span class="toc-item-num">1.3.3&nbsp;&nbsp;</span>One-hot encoding for Sex</a></span></li><li><span><a href="#One-hot-encoding-for-embarked" data-toc-modified-id="One-hot-encoding-for-embarked-1.3.4"><span class="toc-item-num">1.3.4&nbsp;&nbsp;</span>One-hot encoding for embarked</a></span></li><li><span><a href="#Creating-new-label-for-missing-Cabin" data-toc-modified-id="Creating-new-label-for-missing-Cabin-1.3.5"><span class="toc-item-num">1.3.5&nbsp;&nbsp;</span>Creating new label for missing Cabin</a></span></li></ul></li><li><span><a href="#Neural-net-+-KFold-for-the-win" data-toc-modified-id="Neural-net-+-KFold-for-the-win-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Neural net + KFold for the win</a></span><ul class="toc-item"><li><span><a href="#Building-the-model" data-toc-modified-id="Building-the-model-1.4.1"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>Building the model</a></span></li><li><span><a href="#Evaluating-of-the-model" data-toc-modified-id="Evaluating-of-the-model-1.4.2"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Evaluating of the model</a></span></li></ul></li><li><span><a href="#Submitting-to-kaggle" data-toc-modified-id="Submitting-to-kaggle-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Submitting to kaggle</a></span></li><li><span><a href="#The-End" data-toc-modified-id="The-End-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>The End</a></span></li></ul></li></ul></div>

# # Titanic

# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

# Is there any correlation between the features and the fact that the person survived? 

# ## Imports and configurations

# In[ ]:


import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import set_matplotlib_formats, display, HTML

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import roc_curve, classification_report
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBClassifier

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
defaultcolor = '#4c72b0'
set_matplotlib_formats('pdf', 'png')
pd.options.display.float_format = '{:.2f}'.format
rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,   'xtick.labelsize': 16, 'ytick.labelsize': 16}

plt.rcParams['image.cmap'] = 'Blues'

sns.set(style='darkgrid',rc=rc)

data_dir = '../input/' #directory where the dataset is located


# In[ ]:


train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))


# ## Data preprocessing
# <a id='preprocess'></a>

# ### Let's take an overall look at the data.

# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train['Survived'].describe()


# In[ ]:


fig, ax = plt.subplots()
train['Survived'].value_counts().plot.bar(ax=ax)

plt.xticks(rotation=0);

plt.xticks(ticks=[0,1], labels=['Didn\'t Survived', 'Survived']);

bars = ax.get_children()[:2]

bars[0].set_color('r')

ax.hlines(bars[0].get_height(), bars[0].get_x(), bars[1].get_x()+bars[1].get_width(), linestyles='--')

plt.arrow(bars[1].get_x()+bars[1].get_width()/2, bars[0].get_height(), 0, 
          bars[1].get_height()-bars[0].get_height()+10, color='black', width=0.005, head_length=10)

plt.text(bars[1].get_x()+bars[1].get_width()/2+0.01, (bars[0].get_height()+bars[1].get_height())/2, 
         train[train['Survived']==0].shape[0]-train[train['Survived']==1].shape[0]);


# ### PassengerId - Ordinal

# Is just a individual number to order the passenger and does not have any correlation with the fact that the person survived.

# ### Pclass - Ordinal

# Ticket class of the passanger, the first that come to our minds is that the greater the Pclass the worse the chance of surviving, let's explore this correlation.

# In[ ]:


df=train.pivot_table(index='Pclass', values='Survived', aggfunc=np.mean)


# In[ ]:


fig, ax = plt.subplots(figsize=[15,5])
df=100*train.pivot_table(index='Pclass', values='Survived', aggfunc=np.mean)
df.plot.bar(ax=ax, color=defaultcolor)
for i,p in enumerate(ax.patches):
    ax.annotate('{:.2f}%'.format(df['Survived'][1+i]), (p.get_x()+0.18, p.get_height()+1)).set_fontsize(15)
ax.set(title='Percentage of survivors by Pclass', xlabel='Pclass', ylabel='Percentage', ylim=[0,100]);
plt.xticks(rotation=0);


# As it can be seen by the above plot, Pclass=1 had a greater rate of survivors. 

# In[ ]:


fig, ax = plt.subplots(figsize=[15,5])
train['Pclass'].value_counts().sort_index().plot.bar(ax=ax, color=defaultcolor)
ax.set(title='Number of passengers by Pclass', xlabel='Pclass', ylabel='# of passangers');
plt.xticks(rotation=0);


# In[ ]:


fig, ax = plt.subplots()

ax = sns.boxplot(x='Pclass', y='Age', data=train, ax=ax, color=defaultcolor);

ax = sns.pointplot(x='Pclass', y='Age', data=train.groupby('Pclass', as_index=False).mean(), ax=ax, color='g', 
              linestyles='--')

plt.legend((ax.get_children()[22:23]), ['Mean'])


# ### Name and Title - Categorical

# In a first analysis it may look like we cannot get any useful information from the name label but, after analysis the names, we can see that it contains a very interesting insight: the Title the person had at the time, let's investigate that a bit further.

# In[ ]:


train['Title']=train['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])
test['Title']=test['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])


# In[ ]:


from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots()
pd.concat([train['Title'].value_counts(), test['Title'].value_counts()], axis=1, sort=False).plot.bar(stacked=True, ax=ax);

tmp_sr = ((train['Title'].value_counts() + test['Title'].value_counts()).fillna(0)/(train.shape[0]+test.shape[0])).copy()

tmp_sr = tmp_sr.sort_values(ascending=False).cumsum()*100

ax.set_ylabel('Number of passengers')
ax.set_xlabel('Titles');


ax2 = ax.twinx()
ax2.plot(tmp_sr.index, tmp_sr, color="g", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax2.tick_params(axis="y", colors="g")
ax2.set_ylim([0,105])

ax2.grid(False)

ax.legend(['Train', 'Test'], loc='lower right');


# We will only use the Mr, Miss and Mrs titles for our analysis since these are the only ones we have enough data to train our model properly. Let's look at the survival rate of each to see if we can get a quick insight.

# In[ ]:


temp_series = 100*(train.groupby('Title').count()['PassengerId']/train.shape[0]).copy()

temp_series = temp_series.rename('% of passangers')
with pd.option_context('display.float_format', '{:.3f}'.format):
    display(HTML(pd.concat([train.groupby('Title')['Survived'].mean(), temp_series], axis=1)
                 .sort_values(by='% of passangers', ascending=False).to_html()))


# It looks like the passanger with the title Mr had a smaller chance of surviving compared to Mrs and Master.

# - We can also search for the family name

# ### Sex - Categorical

# In[ ]:


fig, ax = plt.subplots()
dfSex = 100*pd.pivot_table(index='Sex', values='Survived', aggfunc=np.mean, data=train)
dfSex.plot.bar(ax=ax, color=defaultcolor)
plt.xticks(rotation=0)
plt.ylim([0,100])
ax.set(title='Percentage of survivors by sex', xlabel='Sex', ylabel='Percentage');
sex=['female', 'male']
for i,p in enumerate(ax.patches):
    ax.annotate('{:.2f}%'.format(dfSex['Survived'][sex[i]]), (p.get_x()+0.14, p.get_height()+1.0)).set_fontsize(20)


# As it was already expected, women stood a better chance of surviving compared to males.

# ### Age - Numerical

# In[ ]:


fig, ax = plt.subplots(figsize=[15,5])
sns.kdeplot(train['Age'].sort_values().dropna(), ax=ax, legend=False, color='r')
sns.kdeplot(train[train['Survived']==1]['Age'].sort_values().dropna(), ax=ax, legend=False, color='g')
plt.legend(['Total', 'Survivors'])
plt.grid(True)
plt.title('Distribution of the passangers by age');
ax.set_yticklabels(['0%', '0.5%', '1.0%', '1.5%', '2.0%', '2.5%', '3.0%']);


# Strangely, it looks like age didn't affect much the chance of surviving since the distribution of all passengers and the one of those who survived looks pretty similar.

# In[ ]:


print("Percentage of missing Ages: {:.2f}%".format(train['Age'].isna().mean()*100))
print("Survival rate of passengers with not missing Age: {:.2f}%".format(train[train['Age'].apply(lambda x: not pd.isna(x))]['Survived'].mean()*100))
print("Survival rate of passengers with missing Age: {:.2f}%".format(train[train['Age'].apply(lambda x: pd.isna(x))]['Survived'].mean()*100))


# We can see something interesting here: passengers missing the age data have a survival rate almost 10% lower than the one from those that have the age information. This can probably be explained by the fact that the age of those who survived was easier to retrive.

# ### SibSp and Parch - Ordinal

# SibSp: # of siblings / spouses aboard the Titanic
# 
# Parch: # of parents / children aboard the Titanic

# In[ ]:


matrix = np.zeros([9,7])
for i in range(8):
    for j in range(6):
        matrix[i][j]=train[(train['SibSp']==i)&(train['Parch']==j)]['Survived'].mean()
fig, ax = plt.subplots();
sns.heatmap(matrix, annot=True, ax=ax, cbar=False)
ax.set_ylim([0, 9])
ax.set_ylabel('Parch')
ax.set_xlabel('SibSp')
ax.set(title='Survivor rate by SibSp and Parch');


# In[ ]:


train.groupby(['SibSp', 'Parch']).count()['PassengerId']


# We can see that more than half of the passanger had the configuration 0 and 0! Let's see the age distribution of those.

# In[ ]:


train[train['SibSp']==0][train['Parch']==0]['Age'].plot.hist(bins=30)


# In[ ]:


train[train['SibSp']==0][train['Parch']==0]['Age'].value_counts().sort_index().head(10)


# As it was expected most of them are adults or older pepole, we can also see that we have some younger pepole with age 5, 11 and 13, which is quite strange, the more reasonable assumption is that the data is wrong for those.

# ### Ticket - Categorical

# It looks like we cannot get any information from the tickets since it does not look like to have any pattern.

# In[ ]:


print('Survivor rate for passengers with numerical only tickets: {}%'
      .format(round(train[train['Ticket'].apply(lambda x: x.isdigit())]['Survived'].mean(),2)*100))
print('Percentage of passangers with numerical only tickets: {:.2f}%'.format(100*train['Ticket'].apply(lambda x: x.isdigit()).mean()))


# ### Cabin - Categorical

# In[ ]:


train['Cabin'].apply(lambda x: not pd.isna(x)).mean()


# The interesting thing here is the number of NaN values, we have only 22% not NaN values, let's see if that has any correlation with the survivor rate.

# In[ ]:


train[train['Cabin'].apply(lambda x: not pd.isna(x))]['Survived'].mean()


# In[ ]:


train[train['Cabin'].apply(lambda x: pd.isna(x))]['Survived'].mean()


# The survivor rate for all the data is approximately 38%, as we can see the ones with the Cabin label had a much higher rate, we can use this fact to improve our model.

# ### Fare - Numerical

# We can expected that the bigger the fare the better is the survivor rate.

# In[ ]:


sns.violinplot(x='Survived', y='Fare', data=train)


# In[ ]:


sns.kdeplot(train['Fare'], shade=True)


# ### Embarked - Categorical

# In[ ]:


train.pivot_table(values='Survived', index='Embarked', aggfunc=np.mean).plot.bar(color=defaultcolor);


# In[ ]:


train.groupby(['Embarked', 'Pclass']).mean()


# ### Dealing with NaN

# In[ ]:


fig, ax = plt.subplots(figsize=[15,10])
sns.heatmap(train.isna(), ax=ax, cbar=False, yticklabels=False)
ax.set_title("NaN in each label for train set");
fig2, ax2 = plt.subplots(figsize=[15,10])
sns.heatmap(test.isna(), ax=ax2, cbar=False, yticklabels=False)
ax2.set_title("NaN in each label for test set");


# ### Missing Embarked - Train

# As we have only 2 missing values here we can deal with it manually.

# In[ ]:


train[train['Embarked'].isnull()]


# Maybe the Fare and Pclass tells us something

# In[ ]:


train[train['Ticket'].str.startswith('1135')]


# In[ ]:


train.pivot_table(index='Embarked', values='Fare', aggfunc=np.mean)


# In[ ]:


train.pivot_table(index='Embarked', values='Pclass', aggfunc=np.mean)


# It looks like the highest probability is that they are from Embarked=C.

# In[ ]:


train['Embarked'].fillna('C', inplace=True)


# ### Missing Age - Train and Test

# We have several values missing in both sets, let's use the Pclass and embarked to fill the NaNs.

# In[ ]:


def fill_age_train(cols):
    age = cols[0]
    pclass = cols[1]
    embarked = cols[2]
    if pd.isna(age):
        return train[train['Pclass']==pclass][train['Embarked']==embarked]['Age'].mean()
    else:
        return age


# In[ ]:


def fill_age_test(cols):
    age = cols[0]
    pclass = cols[1]
    embarked = cols[2]
    if pd.isna(age):
        return test[test['Pclass']==pclass][test['Embarked']==embarked]['Age'].mean()
    else:
        return age


# In[ ]:


train['Age'] = train[['Age', 'Pclass','Embarked']].apply(fill_age_train, axis=1)


# In[ ]:


test['Age'] = test[['Age', 'Pclass','Embarked']].apply(fill_age_test, axis=1)


# ### Missing Fare - Test

# In[ ]:


test[test['Fare'].isna()]


# Let's fill it using Pclass and Embarked

# In[ ]:


mean = test[test['Pclass']==3][test['Embarked']=='S']['Fare'].mean()
test['Fare'].fillna(mean, inplace=True)


# ### Missing Cabin - Train and Test

# Since we have a lot of NaN we will just use the fact that the passenger has or not a Cabin label.

# ## Feature enginnering

# ### Creating new label for missing age

# In[ ]:


train['missingAge']=train['Age'].apply(lambda x: 1 if pd.isna(x) else 0)
test['missingAge']=test['Age'].apply(lambda x: 1 if pd.isna(x) else 0)


# ### Create new label for the title

# In[ ]:


train['Title']=train['Title'].apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs'] else np.nan)
test['Title']=test['Title'].apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs'] else np.nan)
trainTitledf = pd.get_dummies(data=train['Title'], prefix='Title')
testTitledf = pd.get_dummies(data=test['Title'], prefix='Title')
train = pd.concat([train, trainTitledf], axis=1)
test = pd.concat([test, testTitledf], axis=1)


# ### One-hot encoding for Sex

# In[ ]:


train = pd.concat([train,pd.get_dummies(train['Sex'], prefix='Sex', drop_first=True)], axis=1)
test = pd.concat([test,pd.get_dummies(test['Sex'], prefix='Sex', drop_first=True)], axis=1)


# ### One-hot encoding for embarked

# In[ ]:


train = pd.concat([train,pd.get_dummies(train['Embarked'], prefix='Embarked', drop_first=True)], axis=1)
test = pd.concat([test,pd.get_dummies(test['Embarked'], prefix='Embarked', drop_first=True)], axis=1)


# ### Creating new label for missing Cabin

# In[ ]:


train['missingCabin'] = train['Cabin'].apply(lambda x: 1 if not pd.isna(x) else 0)
test['missingCabin'] = test['Cabin'].apply(lambda x: 1 if not pd.isna(x) else 0)


# ## Neural net + KFold for the win

# ### Building the model

# In[ ]:


X_train = np.array(train.drop(['PassengerId', 'Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1))
X_test = np.array(test.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1))


# In[ ]:


y_train = np.array(train['Survived'])


# In[ ]:


def build_model():
    model = XGBClassifier(learning_rate=0.01, n_estimators=500, n_jobs=-1)
    return model


# ### Evaluating of the model

# In[ ]:


def KfoldAnalysis(X, y):
    models = []
    rkfold = RepeatedKFold(n_splits=5, n_repeats=10)
    for train_index, val_index in rkfold.split(X):
        model = build_model()
        model.fit(X[train_index], y[train_index])
        models.append(model)
    return models


# In[ ]:


models = KfoldAnalysis(X_train, y_train)


# ## Submitting to kaggle

# In[ ]:


y_pred = np.array([model.predict(X_test) for model in models])


# In[ ]:


sns.distplot(np.mean(y_pred, 0).reshape(-1), bins=50)
plt.vlines(0.4, 0, 10)


# In[ ]:


ans = [1 if i>0.4 else 0 for i in np.mean(y_pred, 0).reshape(-1)]


# In[ ]:


np.mean(y_train)


# In[ ]:


np.mean(ans)


# In[ ]:


df = pd.DataFrame({'PassengerId' : test['PassengerId'], 'Survived' : pd.Series(ans)})
df.to_csv("submission.csv", index=False)


# ## The End
# Thanks for reading this far, I hope you enjoyed it!
# 
# Sorry, english is not my first language, any errors feel free to correct it in the comments =).
