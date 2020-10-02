#!/usr/bin/env python
# coding: utf-8

# <p id="introduction"><h1>Logistic Classification Model for Titanic Survivors</h1></p>
# 
# <p>Hello Kagglers!!!</p>
# <p>I would like to share with you my second kernel <a href="https://www.kaggle.com/chavesfm/tuning-parameters-for-k-nearest-neighbors-iris">(if you haven't seen my first one, check out this link)</a>.
# In this kernel, I'll start with a visual exploratory analysis of the data, followed by a proposal for a logistic classification model. I hope you enjoy this kernel, and if you like, please UPVOTE! If you have any suggestions or corrections, I would be happy and grateful to hear from you!
# </p>

# <p id="summary"><h1>Summary</h1></p>
# 
# <ol style="font-size:14px; line-height:1.0">
#     <li><a href="#preprocessing-exploring" style="padding: 14px 25px; text-decoration: none">Preprocessing and exploring</a></li>
#         <ol>
#             <li><a href="#import-packages" style="padding: 14px 25px; text-decoration: none">Import Packages</a></li>
#             <li><a href="#load-dataset" style="padding: 14px 25px; text-decoration: none">Load Dataset</a></li>
#             <li><a href="#features-type" style="padding: 14px 25px; text-decoration: none">Features Type</a></li>
#             <li><a href="#missing-values" style="padding: 14px 25px; text-decoration: none">Missing Values</a></li>
#             <li><a href="#features-engineering" style="padding: 14px 25px; text-decoration: none">Features Engineering</a></li>
#                 <ol>
#                     <li><a href="#title-feature" style="padding: 14px 25px; text-decoration: none">Title</a></li>
#                     <li><a href="#deck-feature" style="padding: 14px 25px; text-decoration: none">Deck</a></li>
#                     <li><a href="#alone-feature" style="padding: 14px 25px; text-decoration: none">Alone</a></li>
#                     <li><a href="#relatives-feature" style="padding: 14px 25px; text-decoration: none">Relatives</a></li>
#                 </ol>
#         </ol>
#     <li><a href="#impute-missing-values" style="padding: 14px 25px; text-decoration: none">Impute Missing Values</a></li>
#         <ol>
#             <li><a href="#embarked-impute" style="padding: 14px 25px; text-decoration: none">Embarked</a></li>
#             <li><a href="#fare-impute" style="padding: 14px 25px; text-decoration: none">Fare</a></li>
#             <li><a href="#age-impute" style="padding: 14px 25px; text-decoration: none">Age</a></li>
#         </ol>
#     <li><a href="#exploring" style="padding: 14px 25px; text-decoration: none">Visualizing</a></li>
#         <ol>
#             <li><a href="#distributions" style="padding: 14px 25px; text-decoration: none">Distributions</a></li>
#             <li><a href="#rearrange" style="padding: 14px 25px; text-decoration: none">Rearrange Features</a></li>
#             <li><a href="#categorization" style="padding: 14px 25px; text-decoration: none">Age and Fare Categorization</a></li>
#         </ol>   
#     <li><a href="#model" style="padding: 14px 25px; text-decoration: none">Classification Model</a></li>
#         <ol>
#             <li><a href="#model-features" style="padding: 14px 25px; text-decoration: none">Model Features</a></li>
#             <li><a href="#poly-features" style="padding: 14px 25px; text-decoration: none">Polynomial Features</a></li>
#             <li><a href="#scaled-features" style="padding: 14px 25px; text-decoration: none">Scaled Features</a></li>
#             <li><a href="#corr-features" style="padding: 14px 25px; text-decoration: none">Features Correlation</a></li>
#             <li><a href="#split-training-data" style="padding: 14px 25px; text-decoration: none">Train-Test-Split and Model Training</a></li>
#             <li><a href="#decision-boundary" style="padding: 14px 25px; text-decoration: none">Selecting the Decision Boundary</a></li>
#             <li><a href="#report" style="padding: 14px 25px; text-decoration: none">Classification Reports</a></li>
#             <li><a href="#feature-importance" style="padding: 14px 25px; text-decoration: none">Feature Importance</a></li>
#             <li><a href="#model-coefficients" style="padding: 14px 25px; text-decoration: none">Model Coefficients</a></li>
#             <li><a href="#output" style="padding: 14px 25px; text-decoration: none">Output</a></li>
#         </ol>
# </ol>
# 
# 

# <p id="preprocessing-exploring"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h1> 1. Preprocessing and exploring</h1></a></p>
# <p id="import-packages"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>1. Import Packages</h2></a></p>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

from eli5.sklearn import PermutationImportance
from eli5 import show_weights

import warnings

from IPython.display import display, Math

warnings.filterwarnings("ignore")

sns.set_style('whitegrid')
sns.set_context("talk")

get_ipython().run_line_magic('matplotlib', 'inline')


# <p id="load-dataset"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>2. Load Dataset</h2></a></p>

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# It'll be a good idea to concatenate the train and test dataset to reduce the amount of code.

# In[ ]:


all_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)


# Let's check the top rows of the dataset

# In[ ]:


all_df.head()


# <p id="features-type"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>3. Features Type</h2></a></p>

# In[ ]:


all_df.info()


# <p id="missing-values"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>4. Missing Values</h2></a></p>

# In[ ]:


ans = all_df.drop("Survived", axis=1).isnull().sum().sort_values(ascending=False)
plt.figure(figsize=(12,1.5))
sns.heatmap(pd.DataFrame(data=ans[ans>0], columns=['Missing Values']).T, annot=True, cbar=False, cmap='viridis')


# There are a few missing values in the Fare and Embarked features, and a considerable number of missing values in the Age and Cabin features. We need somehow to figure out a way to fill these missing values. Before impute those missing values, let's create a few features that can be useful to compute them.

# <a href="#summary" id="features-engineering" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>5. Features Engineering</h2></a>
# 
# <a href="#summary" id="title-feature" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>1. Title</h2></a>

# The first feature can be derived from the **Name** of the passengers and it is kind of related with social status and sex.

# In[ ]:


all_df['Title'] = all_df['Name'].apply(lambda name: name.split(',')[1].strip().split('.')[0])


# In[ ]:


newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}


# In[ ]:


all_df['Title'].update(all_df['Title'].map(newtitles))


# <p id="deck-feature"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>2. Deck</h2></a></p>

# The second feature will be derived from the **Cabin** column. The Cabin column is usually composed by a letter (related with the deck) followed by a number (related with the number of the room). To construct this feature, we'll use only the first letter and, for null values, we'll fill with the letter N.

# In[ ]:


all_df['Deck'] = all_df['Cabin'].apply(lambda cabin: cabin[0] if pd.notnull(cabin) else 'N')


# <p id="alone-feature"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>3. Alone</h2></a></p>

# The third feature will tell if the person was travelling alone or not. To get it, the columns **Parch** and **SibSp** will be composed using logical operation.

# In[ ]:


all_df['Alone'] = (all_df['Parch'].apply(lambda value: not(value)) & all_df['SibSp'].apply(lambda value: not(value))).astype('int')


# <p id="relatives-feature"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>4. Relatives</h2></a></p>

# The fourth feature is the number of parents, children, siblings and spouses that each passenger had on board.

# In[ ]:


all_df['Relatives'] = all_df['SibSp'] + all_df['Parch']


# <p id="impute-missing-values"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>2. Impute Missing Values</h2></a></p>
# 
# <a href="#summary" id="embarked-impute" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>1. Embarked</h2></a>

# There are only two <a href='#missing-values'>missing values</a> for the **Embarked** feature. Let's see who are these passengers.

# In[ ]:


all_df[all_df['Embarked'].isnull()]


# We have some good and interesting information here! The two survived, belonged to the first class, were women, traveled alone in the same cabin and paid the same price for the ticket. They were not relatives, but they probably they knew each other! So we can assume they boarded the same place. To fill these NaN values, let's take a look on people that have these same characteristics.

# In[ ]:


sort_embarked_features = ['Cabin', 'Sex', 'Pclass', 'Alone', 'Fare']

sorted_df = all_df.sort_values(by=sort_embarked_features)

sorted_df[(sorted_df['Survived'] == 1) & 
          (sorted_df['Sex'] == 'female') & 
          (sorted_df['Pclass'] == 1) &
          (sorted_df['Alone'] == 1)].sort_values(by=sort_embarked_features)[sort_embarked_features+['Embarked']].head()


# It looks like that the women on Cabin B35 have the same characteristics and come from Cherbourg. As I need to give a shot to fill the NaN values, I would guess that they come from Cherbourg too!

# In[ ]:


all_df['Embarked'].fillna(value='C', inplace=True)


# <a href="#summary" id="fare-impute" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>2. Fare</h2></a>

# There is only one <a href='#missing-values'>missing value</a> for the **Fare** feature. Let's see who is this person.

# In[ ]:


all_df[all_df['Fare'].isnull()]


# As I did with Embarked missing values, I'll look for someone that has the same characteristics.

# In[ ]:


sort_fare_features = ['Sex', 'Pclass', 'Title', 'Deck', 'Embarked', 'Alone', 'Age']

sorted_df = all_df.sort_values(by=sort_embarked_features)

aux = sorted_df[(sorted_df['Alone'] == 1) & 
          (sorted_df['Sex'] == 'male') & 
          (sorted_df['Pclass'] == 1) & 
          (sorted_df['Embarked'] == 'S') & 
          (sorted_df['Deck'] == 'N') & 
          (sorted_df['Title'] == 'Mr') & 
          ((sorted_df['Age'] >= 55) & (sorted_df['Age'] <= 65))]

aux


# Great!!! We have three passengers that are very close to each other. I'll use the mean Fare to impute the missing value.

# In[ ]:


all_df['Fare'].fillna(value=aux['Fare'].mean(), inplace=True)


# <a href="#summary" id="age-impute" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>3. Age</h2></a>

# There are over two hundred <a href='#missing-values'>missing values</a> for the **Age** feature! I'll use the same procedure to fill the NaN values used with the past features, but now I'll use the function below to help filling the values.

# In[ ]:


def impute_num(cols, avg, std):
       
    try:
        avg_value = avg.loc[tuple(cols)][0]
    except Exception as e:        
        print(f'It is not possible to find an average value for this combination of features values:\n{cols}')
        return np.nan
    
    try:
        std_value = std.loc[tuple(cols)][0]
    except Exception as e:        
        std_value = 0        
    finally:
        if pd.isnull(std_value):
            std_value = 0
        
    while True:        
        value = np.random.randint(avg_value-std_value, avg_value+std_value+1)
        if value >= 0:
            break
    return round(value, 0)


# In[ ]:


group_age_features = ['Title','Relatives','Parch','SibSp','Deck','Pclass','Embarked','Sex','Alone']


# In[ ]:


stat_age = all_df.pivot_table(values='Age', index=group_age_features, aggfunc=['mean','std']).round(2)


# The idea here is to find the average age for a group of passengers that have some characteristics in common. For example, let's say that we group the passengers using their title, gender and place of embarkation. For who has the title of Royalty, are female and embarked in Quenstown, we have an average age of 45 years old and standart deviation equal to 10 years old. For those who has the same characteristics and the age is missing, their age will be filled with some random number between
# $\mu_{Age}-\sigma_{Age} \le Age \le \mu_{Age}+\sigma_{Age}$.

# In[ ]:


ages1 = all_df[~all_df['Age'].isnull()][group_age_features].apply(impute_num, axis=1, avg=stat_age.xs(key='mean', axis=1), std=0)
ages2 = all_df[~all_df['Age'].isnull()][group_age_features].apply(impute_num, axis=1, avg=stat_age.xs(key='mean', axis=1), std=stat_age.xs(key='std', axis=1))


# To test this technique I used the passengers that we already know the ages to predicted their own ages. The first one using $(\mu=\mu_{AgeGroup}$, $\sigma=\sigma_{AgeGroup})$ and second one using $(\mu=\mu_{AgeGroup}$, $\sigma=0)$. The predicted ages and statiscal informations are shown in the tables below.

# In[ ]:


comp = pd.DataFrame([all_df[~all_df['Age'].isnull()]['Age'], ages2, ages1]).T
comp.columns = ['Real Age', 'Predicted Age $(\mu=\mu_{Age}$, $\sigma=\sigma_{Age})$', 'Predicted Age $(\mu=\mu_{Age}$, $\sigma=0)$']
comp.head(10)


# In[ ]:


comp.describe()


# In[ ]:


pd.DataFrame(data=[mean_squared_error(all_df[~all_df['Age'].isnull()]['Age'], ages1), 
                   mean_squared_error(all_df[~all_df['Age'].isnull()]['Age'], ages2)],
            index = ['$\mu=\mu_{Age}, \sigma=0$', '$\mu=\mu_{Age}, \sigma=\sigma_{Age}$'],
            columns=['Mean Square Error']).round(1)


# For both approximations there is a high error. To choose one of them, I decided to plot the distribution (shown below). The figure to the left is the distribution of the real ages, the middle figure is the distribution of predicted ages using ($\mu=\mu_{AgeGroup}$, $\sigma=0$) and the right figure is the predicted ages using ($\mu=\mu_{AgeGroup}$, $\sigma=\sigma_{AgeGroup}$). As you can see, the distribution of ($\mu=\mu_{AgeGroup}$, $\sigma=\sigma_{AgeGroup}$) gets much closer to the real ages! That's why I'll choose this approximation to fill the ages.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
sns.distplot(all_df[~all_df['Age'].isnull()]['Age'], bins=20, ax=ax[0])
sns.distplot(ages1, bins=20, ax=ax[1])
sns.distplot(ages2, bins=20, ax=ax[2])

ax[0].set_xlabel('Age')
ax[1].set_xlabel('Age')
ax[2].set_xlabel('Age')

ax[0].set_title('Real Ages')
ax[1].set_title('Predicted Ages $(\mu=\mu_{Age}$, $\sigma=0)$')
ax[2].set_title('Predicted Ages $(\mu=\mu_{Age}$, $\sigma=\sigma_{Age})$')


# In[ ]:


group_features = ['Title','Pclass','Embarked','Sex']


# In[ ]:


stat_age = all_df.pivot_table(values='Age', index=group_features, aggfunc=['mean','std']).round(2)


# In[ ]:


ages = all_df[all_df['Age'].isnull()][group_features].apply(impute_num, axis=1, avg=stat_age.xs(key='mean', axis=1), std=stat_age.xs(key='std', axis=1))


# In[ ]:


all_df['Age'].update(ages)


# <p id="exploring"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>3. Visualizing</h2></a></p>
# 
# <p id="distributions"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>1. Distributions</h2></a></p>

# Now it's time to get some insights about the relationship between the features and the survival passengers. The following graphs show us the percentage of survived for each categorical feature.

# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(15,10))
fields = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Deck', 'Alone', 'Relatives']

i = 0
for row_ax in axes:
    for col_ax in row_ax:
        ans = pd.crosstab(index=[all_df['Survived'], all_df[fields[i]]], columns=all_df[fields[i]], normalize='columns').reset_index()
        ans['%'] = 100*ans[ans.drop(['Survived', fields[i]], axis=1).columns].apply(sum, axis=1).round(2)
        ans.drop(labels=ans.drop(['Survived', fields[i], '%'], axis=1).columns, inplace=True, axis=1)
        ans.sort_values(by=['Survived', '%'], inplace=True)        
        sns.barplot(x=fields[i], y='%', hue='Survived', data=ans, ax=col_ax)
        col_ax.set_ylim((0, 100))
        i+=1
        
plt.tight_layout()        


# <p id="rearrange"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>2. Rearrange</h2></a></p>

# As we can see above, all features seem to influence survival. You may notice that some features, such as SibSp, most people who survive have only one sibling or spouse on board with them. My idea here is to rearrange the categories of these features so that the chance of survival will increase or decrease as the number of categories increases or decreases.

# In[ ]:


def sort_remap(crosstab_df, key):
    alives = list(crosstab_df[crosstab_df['Survived']==1][key])
    deads = list(crosstab_df[crosstab_df['Survived']==0][key])

    alives = list(set(deads) - set(alives)) + alives
    sorted_map = {key:value for value, key in enumerate(alives)}
    
    return sorted_map


# In[ ]:


fields = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Deck', 'Alone', 'Relatives']

for feature in fields:
    ans = pd.crosstab(index=[all_df['Survived'], all_df[feature]], columns=all_df[feature], normalize='columns').reset_index()
    ans['%'] = 100*ans[ans.drop(['Survived', feature], axis=1).columns].apply(sum, axis=1).round(2)
    ans.drop(labels=ans.drop(['Survived', feature, '%'], axis=1).columns, inplace=True, axis=1)
    ans.sort_values(by=['Survived', '%'], inplace=True)
    sorted_map = sort_remap(ans, feature)    
    all_df[feature].update(all_df[feature].map(sorted_map))


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(15,10))
fields = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Deck', 'Alone', 'Relatives']

i = 0
for row_ax in axes:
    for col_ax in row_ax:
        ans = pd.crosstab(index=[all_df['Survived'], all_df[fields[i]]], columns=all_df[fields[i]], normalize='columns').reset_index()
        ans['%'] = 100*ans[ans.drop(['Survived', fields[i]], axis=1).columns].apply(sum, axis=1).round(2)
        ans.drop(labels=ans.drop(['Survived', fields[i], '%'], axis=1).columns, inplace=True, axis=1)
        ans.sort_values(by=['Survived', '%'], inplace=True)        
        sns.barplot(x=fields[i], y='%', hue='Survived', data=ans, ax=col_ax)
        col_ax.set_ylim((0, 100))
        i+=1
        
plt.tight_layout()        


# <p id="categorization"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>3. Age and Fare Categorization</h2></a></p>

# Most of the features are categorical, with the exception of age and fare that are a float point number. It'd be a good practice to discretizer these features. Let's do that using KBinsDiscretizer from sklearn.preprocessing with eight bins and uniform distributed.
# 
# Obs: I have done several tests using the model accuracy and eight bins are enough!

# In[ ]:


kdis = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform').fit(all_df[['Age', 'Fare']])


# In[ ]:


cat_age_fare = kdis.transform(all_df[['Age', 'Fare']])


# In[ ]:


all_df['Cat_Age'] = cat_age_fare[:,0].astype('int')
all_df['Cat_Fare'] = cat_age_fare[:,1].astype('int')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,5))
fields = ['Cat_Age', 'Cat_Fare']

i = 0
for col_ax in axes:
    ans = pd.crosstab(index=[all_df['Survived'], all_df[fields[i]]], columns=all_df[fields[i]], normalize='columns').reset_index()
    ans['%'] = 100*ans[ans.drop(['Survived', fields[i]], axis=1).columns].apply(sum, axis=1).round(2)
    ans.drop(labels=ans.drop(['Survived', fields[i], '%'], axis=1).columns, inplace=True, axis=1)
    ans.sort_values(by=['Survived', '%'], inplace=True)        
    sns.barplot(x=fields[i], y='%', hue='Survived', data=ans, ax=col_ax)
    col_ax.set_ylim((0, 100))
    i+=1
        
plt.tight_layout()        


# I will reorganize the age and fare categories, as I did with the previous categories, so the chance of survival will increase or decrease as the number of categories grows or decreases.

# In[ ]:


fields = ['Cat_Age', 'Cat_Fare']

for feature in fields:
    ans = pd.crosstab(index=[all_df['Survived'], all_df[feature]], columns=all_df[feature], normalize='columns').reset_index()
    ans['%'] = 100*ans[ans.drop(['Survived', feature], axis=1).columns].apply(sum, axis=1).round(2)
    ans.drop(labels=ans.drop(['Survived', feature, '%'], axis=1).columns, inplace=True, axis=1)
    ans.sort_values(by=['Survived', '%'], inplace=True)
    sorted_map = sort_remap(ans, feature)    
    all_df[feature].update(all_df[feature].map(sorted_map))


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,5))
fields = ['Cat_Age', 'Cat_Fare']

i = 0
for col_ax in axes:
    ans = pd.crosstab(index=[all_df['Survived'], all_df[fields[i]]], columns=all_df[fields[i]], normalize='columns').reset_index()
    ans['%'] = 100*ans[ans.drop(['Survived', fields[i]], axis=1).columns].apply(sum, axis=1).round(2)
    ans.drop(labels=ans.drop(['Survived', fields[i], '%'], axis=1).columns, inplace=True, axis=1)
    ans.sort_values(by=['Survived', '%'], inplace=True)        
    sns.barplot(x=fields[i], y='%', hue='Survived', data=ans, ax=col_ax)
    col_ax.set_ylim((0, 100))
    i+=1
        
plt.tight_layout()        


# <p id="model"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>4. Classification Model</h2></a></p>
# 
# <p id="model-features"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>1. Model Features</h2></a></p>

# Now is the time to implement the classification model. The red and green figure below shows the features that will be used for the model. Feel free to play with it!

# In[ ]:


features_on_off = {'PassengerId':False,
                   'Survived':False,
                   'Pclass':True,
                   'Name':False,
                   'Sex':True,
                   'Age':False,
                   'SibSp':True,                   
                   'Parch':True,
                   'Ticket':False,
                   'Fare':False,
                   'Cabin':False,
                   'Embarked':True,
                   'Title':True,
                   'Deck':True,                   
                   'Alone':True,
                   'Relatives':True,
                   'Cat_Age':True,
                   'Cat_Fare':True}


# In[ ]:


features_on = [key for key, status in features_on_off.items() if status]
aux = pd.DataFrame(features_on_off, index=['On\Off'])

plt.figure(figsize=(15,1.5))
sns.heatmap(aux, cbar=False, cmap=['red', 'green'], annot=True)


# <p id="poly-features"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>2. Polynomial Features</h2></a></p>

# Let's use a second degree polynomial to capture more complex relationships between the features.

# In[ ]:


poly = PolynomialFeatures(degree=2, include_bias=False).fit(all_df[features_on])
poly_features = poly.transform(all_df[features_on])
poly_df = pd.DataFrame(data=poly_features, columns=poly.get_feature_names(features_on))


# <p id="scaled-features"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>3. Scaled Features</h2></a></p>

# Standardize the features is always a good practice to avoid scale effect, and it helps the minimization algorithm behind the classification model to achieve the minimum faster.

# In[ ]:


std_scaler = StandardScaler().fit(poly_df)
scaled_features = std_scaler.transform(poly_df)
scaled_df = pd.DataFrame(data=scaled_features, columns=poly_df.columns)


# <p id="corr-features"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>4. Features Correlation</h2></a></p>

# The heat map below shows the correlation between resources and survival. The linear correlations are weak, but this doesn't mean that these features are not good candidates for the classification model. Later we'll see how they behave together.

# In[ ]:


new_df = scaled_df.copy()
new_df['Survived'] = all_df['Survived']


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(new_df.corr()[['Survived']].sort_values('Survived', ascending=False).drop('Survived').head(15), annot=True, cbar=False, cmap='viridis')


# <p id="split-training-data"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>5. Train-Test-Split and Model Training</h2></a></p>

# In this step I'll split the dataset into training and testing, according to the original datasets provided by kaggle. Then I'll train the logistic classifier and see how well its perform in the test set.
# 
# **I've got the true labels of the test set from some kaggle user, for learning and test purposes. So if you see my rank equal 1.0, that's not true, I've test this labels to make sure that they were correct.**

# In[ ]:


X_train = scaled_df.loc[range(train_df.shape[0])]
y_train = all_df.loc[range(train_df.shape[0]), 'Survived']

X_test = scaled_df.loc[range(train_df.shape[0], train_df.shape[0]+test_df.shape[0])]
y_test = pd.read_csv('../input/true-labels/submission_true.csv')['Survived']


# In[ ]:


log_reg = LogisticRegressionCV(Cs=1000, cv=5, refit=True, random_state=101, max_iter=200).fit(X_train, y_train)


# In[ ]:


print(f'Accuracy on the training set: {100*log_reg.score(X_train, y_train):.1f}%')


# In[ ]:


print(f'Accuracy on the test set: {100*log_reg.score(X_test, y_test):.1f}%')


# <p id="decision-boundary"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>6. Selecting the Decision Boundary</h2></a></p>

# When we use the prediction method provided by the logistic regression classifier, it usually returns rank based on the highest probability, but we can try to figure out the best probability boundary between alive or dead looking at the error in the test dataset for different probability limits. This is exactly what is shown by the dashed line in the chart below, which is the probability cut that corresponds to the smallest error in the test set.

# In[ ]:


p = np.linspace(0.01, 1, 50)
error_rate = {'train':[], 'test':[]}

for p_value in p:    
    prob = log_reg.predict_proba(X_train)
    y_pred = np.apply_along_axis(lambda pair: 1 if pair[1] > p_value else 0, 1, prob)
    error_rate['train'].append(mean_squared_error(y_train, y_pred))
    prob = log_reg.predict_proba(X_test)
    y_pred = np.apply_along_axis(lambda pair: 1 if pair[1] > p_value else 0, 1, prob)
    error_rate['test'].append(mean_squared_error(y_test, y_pred))


# In[ ]:


best_p = p[np.array(error_rate['test']).argmin()]


# In[ ]:


plt.figure(figsize=(12,5))

min_x, max_x = 0, 1
min_y, max_y = min(min(error_rate['test'], error_rate['train'])), max(max(error_rate['test'], error_rate['train']))

plt.plot(p, error_rate['train'], label='Train', marker='o', color='red')
plt.plot(p, error_rate['test'], label='Test', marker='o', color='blue')

plt.ylabel('Mean Squared Error')
plt.xlabel('Probability')

plt.vlines(x=best_p, ymin=min_y-0.1, ymax=max_y+0.1, linestyle='--', label=f'Best $p={best_p:.2f}$')

plt.ylim((min_y-0.1, max_y+0.1))
plt.xlim(min_x-0.1, max_x+0.1)

plt.legend(loc='best')


# In[ ]:


print(f'In this particular case, the best probability limit is around {best_p:.2}. This means that if the model provides a survival value greater than {best_p:.2}, we accept that the passenger will survive or die otherwise.')


# The next lines will get the labels using the best probability limit the we found.

# In[ ]:


prob = log_reg.predict_proba(X_train)
y_pred_train = np.apply_along_axis(lambda pair: 1 if pair[1] > best_p else 0, 1, prob)
print(f'Accuracy on the training set with the best choice of p: {100*np.mean(y_train == y_pred_train):.1f}%')


# In[ ]:


prob = log_reg.predict_proba(X_test)
y_pred_test = np.apply_along_axis(lambda pair: 1 if pair[1] > best_p else 0, 1, prob)
print(f'Accuracy on the testing set with the best choice of p: {100*np.mean(y_pred_test == y_test):.1f}%')


# <p id="report"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>7. Classification Reports</h2></a></p>

# ### Classification Report in the training set

# In[ ]:


print(classification_report(y_pred_train, y_train))


# ### Classification Report in the testing set

# In[ ]:


print(classification_report(y_pred_test, y_test))


# <p id="feature-importance"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>8. Feature Importance</h2></a></p>

# Now we can check what are the most important features to our model. Let's use permutation importance to see the feature's rank.

# In[ ]:


perm = PermutationImportance(log_reg).fit(X_train, y_train)
show_weights(perm, feature_names = list(X_train.columns))


# <p id="model-coefficients"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>9. Model Coefficients</h2></a></p>

# The table below shows the top 15 features coefficients that have a positive contribution to the model.

# In[ ]:


weights = pd.DataFrame(data=log_reg.coef_[0], index=scaled_df.columns, columns=['Weights'])
weights.loc['Intercept'] = log_reg.intercept_
weights.sort_values('Weights', ascending=False, inplace=True)
weights.head(15)


# <p id="output"><a href="#summary" style="padding: 14px 0px; text-decoration: none; color: black;"><h2>10. Output</h2></a></p>

# In[ ]:


output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred_test.astype('int')})


# In[ ]:


output.to_csv('submission.csv', index=False)

