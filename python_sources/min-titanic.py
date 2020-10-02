#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale = 2.5)

import missingno as msno

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Data Check

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.head()


# - The feartures in the issue that I try to deal with are Pclass, Age, SibSp, Parch, Fare, and the target label is Survived. 

# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# ## 1.1 Check Null Data

# In[ ]:


df_train.isnull().sum()


# In[ ]:


for col in df_train.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)


# - In the Train set, features of Age, Cabin and Embarked have null data. 

# In[ ]:


df_test.isnull().sum()


# In[ ]:


for col in df_test.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)


# - In the Test set, features of Age, Fare and Cabin have null data. 

# - Let's check null data with using MANO library.

# In[ ]:


msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))


# In[ ]:


msno.bar(df=df_train.iloc[:, :], figsize = (8, 8), color=(0.8, 0.5, 0.2))


# In[ ]:


msno.bar(df=df_test.iloc[:,:], figsize=(8,8), color=(0.8, 0.5, 0.2))


# ## 1.2 Check Target Label

# - Which distribution the target label has needs to be checked. 
# 
# - In case of binary clssification, the model evaluation method depends on one and zero distributions

# In[ ]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')
plt.show()


# - Sadly, many people were dead and only 38.4% survived.
# - The distribution of target label seems to be balanced. 

# # 2. Exploratory Data Analysis

# ## 2.1 Pclass

# In[ ]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()


# In[ ]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()


# In[ ]:


pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()


# - The higher Pclass survied the more.

# In[ ]:


y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7E32', '#FFDF00', '#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()


# - It seems that Pclass had a huge impact on Survived and such Pclass may be used as a feature when modeling. 

# ## 2.2 Sex

# In[ ]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()


# - As you know, the female group had a higher survival rate. 

# In[ ]:


df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')


# - Like Pclass, Sex musb be an important feature. 

# ## 2.3 Both Sex and Pclass

# - Let's check Survived with two variables, Sex and Pclass. 

# In[ ]:


sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)


# - In every Pclass category, the survival rate of the female group is higher than those of the male group. 
# - Further, regardless of Sex, the higer Pclass has the more Survied.

# In[ ]:


sns.factorplot(x='Sex', y='Survived', col='Pclass', data=df_train, satureation=.5, size=9, aspect=1)


# ## 2.4 Age

# In[ ]:


print('Max Age: {:.1f} Years'.format(df_train['Age'].max()))
print('Min Age: {:.1f} Years'.format(df_train['Age'].min()))
print('Mean Age: {:.1f} Years'.format(df_train['Age'].mean()))


# - Histogram of Age by Survived

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(9,5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Non-Survived == 0'])
plt.show()


# - The younger was the more survived.

# - Age distribution by Pclass

# In[ ]:


plt.figure(figsize=(8,6))
df_train['Age'][df_train['Pclass']==1].plot(kind='kde')
df_train['Age'][df_train['Pclass']==2].plot(kind='kde')
df_train['Age'][df_train['Pclass']==3].plot(kind='kde')
plt.xlabel('Age')
plt.title('Age Distribution by Pclass')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
plt.show()


# - The older has the higher class.

# - How will the rate of survived change as the scope of Age changes?

# In[ ]:


cummulate_survival_ratio = []
for i in range(1, 80):
    cummulate_survival_ratio.append(df_train[df_train['Age']<i]['Survived'].sum()/len(df_train[df_train['Age']<i]['Survived']))
plt.figure(figsize=(7,7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survial rate')
plt.xlabel('Range of Age(0~x)')
plt.show()


# - The younger has the higher survival rate.

# ## 2.5 Pclass, Sex, Age

# In[ ]:


f, ax = plt.subplots(1,2, figsize=(18, 8))
sns.violinplot("Pclass", "Age", hue="Survived", data=df_train, scale='count', split=True, ax=ax[0])
ax[0].legend(title='Survived', loc='rightupper')
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot("Sex", "Age", hue="Survived", data=df_train, scale='count', split=True, ax=ax[1])
ax[1].legend(title='Survived', loc='rightupper')
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
plt.show()


# ## 2.6 Embarked

# In[ ]:


f, ax = plt.subplots(1,1, figsize=(7,7))
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)


# In[ ]:


f, ax = plt.subplots(2,2, figsize=(20,15))
sns.countplot('Embarked', data=df_train, ax=ax[0, 0])
ax[0,0].set_title('(1) No. Of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0,1])
ax[0,1].set_title('Male-Females Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# ## 2.7 Family - SibSp + Parch

# In[ ]:


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1


# In[ ]:


print('Maximum size of Family: ', df_train['FamilySize'].max())
print('Minimum size of Family: ', df_train['FamilySize'].min())


# In[ ]:


f, ax = plt.subplots(1, 3, figsize=(40, 10))
sns.countplot('FamilySize', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)
sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)
df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# ## 2.8 Fare

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(8,8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness: {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')


# In[ ]:


df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()


# In[ ]:


df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(8,8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness: {:.2f}'.format(df_train['Fare'].skew(), ax=ax))
g = g.legend(loc='best')


# ## 2.9 Cabin

# - As the rate of NaNs in Cabin is around 80%, Cabin is excluded. 

# ## 2.10 Ticket

# In[ ]:


df_train['Ticket'].value_counts()


# # 3. Feature Engineering

# ## 3.1 Fill Null
# 
# ### 3.1.1 Fill Null in Age using title

# In[ ]:


df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')
df_test['Initial'] = df_test.Name.str.extract('([A-Za-z]+)\.')


# In[ ]:


pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')


# In[ ]:


df_train['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],                             ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)
df_test['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],                             ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)


# In[ ]:


df_train.groupby('Initial').mean()


# In[ ]:


df_train.groupby('Initial')['Survived'].mean().plot.bar()


# In[ ]:


df_train.groupby('Initial').mean()


# In[ ]:


df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mr'), 'Age'] = 33
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mrs'), 'Age'] = 36
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Master'), 'Age'] = 5
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Miss'), 'Age'] = 22
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Other'), 'Age'] = 46

df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Mr'), 'Age'] = 33
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Mrs'), 'Age'] = 36
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Master'), 'Age'] = 5
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Miss'), 'Age'] = 22
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Other'), 'Age'] = 46


# ### 3.1.2 Fill Null in Embarked

# In[ ]:


df_train['Embarked'].fillna('S', inplace=True)


# ## 3.2 Change Age (continuous to categorical)

# In[ ]:


df_train['Age_cat'] = 0
df_train.loc[df_train['Age'] < 10, 'Age_cat'] = 0
df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1
df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2
df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3
df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4
df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5
df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6
df_train.loc[70 <= df_train['Age'], 'Age_cat'] = 7

df_test['Age_cat'] = 0
df_test.loc[df_train['Age'] < 10, 'Age_cat'] = 0
df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1
df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2
df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3
df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4
df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5
df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6
df_test.loc[70 <= df_test['Age'], 'Age_cat'] = 7


# In[ ]:


def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7
    
df_train['Age_cat_2'] = df_train['Age'].apply(category_age)
    


# In[ ]:


print((df_train['Age_cat'] == df_train['Age_cat_2']).all())


# In[ ]:


df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)
df_test.drop(['Age'], axis=1, inplace=True)


# ## 3.3 Change Initial, Embarked and Sex (string to numerical)

# In[ ]:


df_train['Initial'] = df_train['Initial'].map({'Master':0, 'Miss': 1, 'Mr':2, 'Mrs':3, 'Other':4})
df_test['Initial'] = df_test['Initial'].map({'Master':0, 'Miss': 1, 'Mr':2, 'Mrs':3, 'Other':4})   


# In[ ]:


df_train['Embarked'].unique()


# In[ ]:


df_train['Embarked'].value_counts()


# In[ ]:


df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'Q':1, 'S':2})
df_test['Embarked'] = df_test['Embarked'].map({'C':0, 'Q':1, 'S':2})


# In[ ]:


df_train['Embarked'].isnull().any()


# In[ ]:


df_test['Embarked'].isnull().any()


# In[ ]:


df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1})
df_test['Sex'] = df_test['Sex'].map({'female':0, 'male':1})


# - Check correlations bewteen features

# In[ ]:


heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']]

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap,            linecolor='white', annot=True, annot_kws={'size':16})
del heatmap_data


# - There are some correlations between Sex, Pclass, Fare and Survived.
# - There is no feature having strong a correlation each other. 

# ## 3.4 One-hot encoding on Initial and Embarked

# In[ ]:


df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')
df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')


# In[ ]:


df_train.head()


# In[ ]:


df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix = 'Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')


# ## 3.5 Drop Columns

# In[ ]:


df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# # 4. Building Machine Learning Model and Prediction

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# ## 4.1 Preparation - Split dataset into train, valid, test set

# In[ ]:


X_train = df_train.drop('Survived', axis=1).values
target_label = df_train['Survived'].values
X_test = df_test.values


# In[ ]:


X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)


# ## 4.2 Model Generation and Prediction

# In[ ]:


model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)


# In[ ]:


print('Accuracy: {:.2f}%'.format(100 * metrics.accuracy_score(prediction, y_vld)))


# ## 4.3 Feature Importance

# In[ ]:


from pandas import Series
feature_importance = model.feature_importances_
Series_fet_imp = Series(feature_importance, index=df_test.columns)


# In[ ]:


plt.figure(figsize=(8,8))
Series_fet_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()


# ## 4.4 Prediction on Test Set

# In[ ]:


submission = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


submission.head()


# In[ ]:


prediction = model.predict(X_test)
submission['Survived'] = prediction


# In[ ]:


prediction


# In[ ]:


submission.to_csv('./my_first_submission.csv', index=False)


# In[ ]:




