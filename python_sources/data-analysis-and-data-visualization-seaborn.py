#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# Now, let's take a look at the titanic data: There are 891 rows by 12 columns observations in total. Just by the logic, I would assume that Age, Passenger Class, and Sex are major factors in the survival rate on the Titanic.

# In[ ]:


print('Test data shape: ', train.shape)
train.head()


# In[ ]:


sns.distplot(train['Sex'], bins=8) 


# In[ ]:


sns.pairplot(train)


# In[ ]:


train.describe()


# The average Age is around 30 years old which is not a surprise considering the time.

# In[ ]:


train.info()


# ## Missing values

# In[ ]:


train.isnull().sum()


# ## Treating Missing Values

# In[ ]:


import numpy as np

from pandas import Series,DataFrame
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer


# Missing Values in the column Age can be fixed by imputing values, in our case, using average, of this column. 
# For the column Embarked the missing values can be fixed by 'ffil' that implaces last valid observation in this column. At the same time, I dropped PassengerId, Name and Ticket columns.

# In[ ]:


imputer = SimpleImputer(np.nan, "mean")

train['Age'] = imputer.fit_transform(np.array(train['Age']).reshape(891, 1)) 
train.Embarked.fillna(method='ffill', inplace=True) 
train.drop(['PassengerId', 'Name','Ticket'], axis=1, inplace=True)
train.head()


# And now for the test data:

# In[ ]:


test['Age'] = imputer.fit_transform(np.array(test['Age']).reshape(418, 1))
test.Embarked.fillna(method='ffill', inplace=True)
test.Fare.fillna(method='ffill', inplace=True)
test.drop(['Name', 'Ticket'], axis=1, inplace=True)
test.head()


# ## Data Analysis
# 

# ### Survived

# Distribution of survived: 1 is for survived and 0 is for not. 

# In[ ]:


train['Survived'].value_counts()


# ### Gender

# In[ ]:


plt.figure(figsize=[10,5])
sns.countplot(x = 'Sex', hue = 'Survived', data = train)
plt.xticks(rotation = 20);


# The survival rate of women is a lot higher than men.

# #### Barplot of survival rate between men and women:

# Note: Some of the seaborn color palettes can be found here:
# https://images.app.goo.gl/a4gDzjx5ZgAvf5Kf8

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train, palette=('RdPu'));


# Percentages of who survived: The men's survival is tragically a lot lower than women's.

# In[ ]:


print('% of survived females:', train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)[1]*100)
print('% of survived males:', train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)[1]*100)


# Another way to calculate these % using groupby():

# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# The percentage of survived women is a lot higher. However, it was expected since the women and children were put on the emergency boats first.

# ### Passenger Class

# To look at the correlation between passenger class and survival statistics I would plot a countplot.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,6))
ax = sns.countplot(x = 'Survived', hue = 'Pclass', data = train, palette = 'YlOrRd')
ax.set_xlabel('Survived')
ax.set_title('Survival Rate for Passenger Classes', fontsize = 14, fontweight='bold');


# As expected, 3rd class passngers has the lowest survival rate, while the 1st class has the highest.

# At the catplot below we can see the detailed distribution of the survival rate between different passenger classes on the Titanic for men and women.  

# In[ ]:


ax = sns.catplot(x="Pclass", hue="Sex", col="Survived",
                data=train, kind="count",
                height=4, aspect=.7, palette = 'OrRd');


# Here is another representation of this correlation:

# In[ ]:


sns.countplot(x = "Pclass", hue = "Survived", data = train, palette = 'RdPu');


# And here is the rate of survival by class:

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data= train, palette = 'BuGn');


# In[ ]:


perc = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
perc*100


# Now, it's clear that 1st class had 63%, 2nd class is almost 'fifty-fifty' and 3rd class had only 24% survival rate.

# Below, another representation of the survivors and victims for each ticket class.

# In[ ]:


sns.factorplot(x='Pclass', y='Survived', hue = 'Sex', data = train, palette = 'PRGn');


# ## Age

# In[ ]:


sns.boxplot(x='Sex', y='Age', hue = 'Survived',data=train);


# Looking at the Age distribution for men and women, it's clear that the average age for both was about 30. Many men older 50 years old and children after 8 years old did not survive. However, it's opposite for women: most of the women older 50 years old survived. The middle age of men, who did not survive, was between 24 and 36 years old, for women - between 18 and 32. The oldest person on the Titanic survived who was 80 years old. 

# In[ ]:


grid = sns.FacetGrid(train, col='Survived')
grid.map(plt.hist, 'Age', bins=25, color = 'y').add_legend();
sns.set(style="ticks", color_codes=True);


# Most of the passengers are between 15 and 40 years old and many infants had survived. 

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(train['Age'], bins=24, color='g');


# Some additional statistics for both data sets: Average and Standard Deviation.

# In[ ]:


avg_age_train = train ["Age"].mean()
std_age_train = train ["Age"].std()

avg_age_test = test["Age"].mean()
std_age_test = test ["Age"].std()


# By creating Age subgroups we can examen the data even more.

# In[ ]:


bins = [0, 1, 12, 18, 21,  60, np.inf]
labels = ['Infant', 'Child', 'Teenager',' Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
sns.barplot(x="AgeGroup", y="Survived", data= train)
plt.show;


# In[ ]:


train.head()


# In[ ]:


train[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


AgeGroup_train  = pd.get_dummies(train['AgeGroup'])
AgeGroup_train.columns = ['Infant', 'Child', 'Teenager',' Young Adult', 'Adult', 'Senior']
AgeGroup_test  = pd.get_dummies(test['AgeGroup'])
AgeGroup_test.columns = ['Infant', 'Child', 'Teenager',' Young Adult', 'Adult', 'Senior']

train = train.join(AgeGroup_train)
test = test.join(AgeGroup_test)


# Instead of creating dummies like I did above, you can map values to each category. And, as you can see, these works.

# In[ ]:


age_mapping = {'Infant': 1, 'Child': 2, 'Teenager': 3, 'Young Adult': 4, 'Adult': 5, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()


# In[ ]:


train.drop(['AgeGroup', 'Age'],axis=1,inplace=True)
test.drop(['AgeGroup', 'Age'],axis=1,inplace=True)
train.head()


# ## Cabin

# In[ ]:


train["Cabin_new"] = (train["Cabin"].notnull().astype('int'))
test["Cabin_new"] = (test["Cabin"].notnull().astype('int'))
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# In[ ]:


print("% of Cabin = 1 survived:", train["Survived"][train["Cabin_new"] == 1].value_counts(normalize = True)[1]*100)
print("% of Cabin = 0 survived:", train["Survived"][train["Cabin_new"] == 0].value_counts(normalize = True)[1]*100)

sns.barplot(x="Cabin_new", y="Survived", data=train).set_title('Cabin vs No Cabin')
plt.show()


# As a metter of fact, passangers with the Cabin were more likely to survive.

# ## Family Size

# In[ ]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# To analyse the data based on the family size, first, create new column:

# In[ ]:


train['FamilySize'] = train['Parch'] + train['SibSp']
test['FamilySize'] = test['Parch'] + test['SibSp']
#train.drop(['Parch', 'SibSp'], axis=1,inplace=True)
#test.drop(['Parch', 'SibSp'], axis=1,inplace=True)
sns.barplot(x="FamilySize", y="Survived", data=train)
plt.show;


# Another way to present it:

# In[ ]:


sns.pointplot(x='FamilySize', y = 'Survived', data = train);


# ## Port of Embarkation

# In[ ]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q']);


# Looking at the survaval data for each port separetely, port Southampton has the highest number of survivors. At the same time, the largest number of victims came from port Cherbourg.

# In[ ]:


sns.countplot(x='Embarked', hue='Survived', data=train);


# In this case, I decided to use mapping each Embarked value to a numerical value instead of creating a dummies.

# In[ ]:


embark_dummies_train  = pd.get_dummies(train['Embarked'])
embark_dummies_test  = pd.get_dummies(test['Embarked'])

train = train.join(embark_dummies_train)
test = test.join(embark_dummies_test)
train.head()


# In[ ]:


train = train.drop(['Embarked'], axis = 1)
test = test.drop(['Embarked'], axis = 1)
train.head()


# Now, I map each Gender value to a numerical value:

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# And the next 3 plots are just cool kdeplots for Gender and Survival Rate 

# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True)
sns.kdeplot(train['Survived'], train['Sex'],cbar = cmap,shade=True);


# In[ ]:


g = sns.jointplot(x='Survived', y = 'Sex', data=train, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("Survived", "Gender")


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(train['Survived'], train['Sex'], ax=ax)
sns.rugplot(train['Survived'], color="g", ax=ax)
sns.rugplot(train['Sex'], vertical=True, ax=ax);


# ## Fare

# In[ ]:


# convert from float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare']= test['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = train["Fare"][train["Survived"] == 0]
fare_survived = train["Fare"][train["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
train['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50), color = 'purple')

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False, color = 'r');


# ## Correlations  and Heatmap

# In[ ]:


mask = np.zeros_like(train.corr(), dtype=np.bool)
## in order to reverse the bar replace "RdBu" with "RdBu_r"
plt.subplots(figsize = (15,12))
sns.heatmap(train.corr(), annot=True,mask = False,cmap = 'OrRd', linewidths=.7, linecolor='black',fmt='.2g',center = 0,square=True)

plt.title("Correlations", y = 1.03,fontsize = 20, fontweight = 'bold', pad = 40);

