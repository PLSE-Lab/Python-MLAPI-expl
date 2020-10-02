#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading the required libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


# ## Treating the data

# In[ ]:


# Loading the data set
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
data = pd.concat([train, test])
data.isna().sum()


# The features Age, Fare, Cabin and Embarked have missing values... Before filling the missing values I'll add 2 new features to store the values if Age and Cabin are missing... Since Embarked and Fare have just 1 and 2 missing values I'll just fill them.
# 
# After that, it's time to see how do deal with the missing values!
# 
# Since Fare and Age are numerical features I'll vizualise the using boxplot the indentify outliers. and decide the best way to fill them.
# 
# Embarked is a categorical feature so I'll fill with the value which appears the most...
# 
# Cabin has too much missing values and is also a text attribute, so it needs to be analysed more carefully, so I'll save it for later.

# In[ ]:


# Creating features has_age and has_cabin
data['age_na'] = data['Age'].isna().replace([True, False], [1, 0])
data['cabin_na'] = data['Cabin'].isna().replace([True, False], [1, 0])


# In[ ]:


# Visualizing the features with missing values

fare = data.loc[data['Fare'].notna(), 'Fare']
age = data.loc[data['Age'].notna(), 'Age']
embarked = data.loc[data['Embarked'].notna(), 'Embarked'].value_counts()

# Plotting the informations
fig, ax = plt.subplots(1,3, figsize = (12,3))
ax[0].boxplot(age, labels = ['Age'])
ax[1].boxplot(fare, notch= True, labels = ['Fare'])
ax[2].bar(embarked.index, embarked.values, color = ['r', 'g', 'b'], alpha=0.5)

plt.show()


# Fare has too many outliers, and because of this ***fill*** the missing values with the average may not be a good choice, instead of it, I'll use the median wich represents better the behavior of this feature.
# 
# On the other hand, Age doesn't have that much outliers, therefore I'll use the average to fill me missing values of this feature.
# 
# In Embarked the most frequent value is 'S', so I'll fill the missing values with it.

# In[ ]:


# Filling the missing values except the ones at the Cabin column...
data_filled = data.copy()

data_filled.loc[data['Age'].isna(), 'Age'] = round(data_filled['Age'].mean())
data_filled.loc[data['Fare'].isna(), 'Fare'] = data_filled['Fare'].median()
data_filled.loc[data['Embarked'].isna(), 'Embarked'] = 'S'


# It's time to look at the categorical features (Pclass, Sex, Embarked).
# The feature Pclass is a hierarquical feature, therefore i'll not change anything on it.
# The feature Sex will be enconded as a binary feature. 
# In the feature Embarked I can't just encode the values to numerical, since there's no hierarquical relatioship amid the categories. So, it'll be divided in 3 new binary features (one for each category).
# 
# I also will create a new feature to store the number of family members embarked by summing the features 'SibSp' and 'Parch'.

# ## EDA

# In[ ]:


survived = data_filled['Survived'] == 1
died = data_filled['Survived'] == 0
train = data_filled['Survived'].notna()
fare_100 = data_filled['Fare']<=100
bins = list(range(0,100,7))


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4), sharey=True)

sns.distplot(data_filled[train]['Fare'], hist = False, color = 'green', ax = ax[0], label= 'Total')
sns.distplot(data_filled[survived]['Fare'], hist = False, color = 'blue', ax = ax[0], label= 'Survived')
sns.distplot(data_filled[died]['Fare'], hist = False, color = 'red', ax = ax[0], label= 'Died')
ax[0].set_title('Total fare distribution')

sns.distplot(data_filled[train & fare_100]['Fare'], hist = False, color = 'green', ax = ax[1], label= 'Total')
sns.distplot(data_filled[survived & fare_100]['Fare'], hist = False, color = 'blue', ax = ax[1], label= 'Survived')
sns.distplot(data_filled[died & fare_100]['Fare'], hist = False, color = 'red', ax = ax[1], label = 'Died')
ax[1].set_title('Fare distribution (Less than 100)')

fares = data_filled[train]['Fare'].apply(lambda x: x//10).round().value_counts()
fare_suvived = data_filled[survived]['Fare'].apply(lambda x: x//10).round().value_counts().rename("Survived")
fares = pd.concat([fares, fare_suvived], axis = 1).replace(np.nan, 0)
fares['Percentage'] = 100*fares['Survived']/fares['Fare']
fig2, ax2 = plt.subplots(1, figsize=(18, 3))

ax2.bar(fares.index, fares.Percentage)
ax2.set_title("Rate of survival for each increase of 10 in Fare")


# By looking at that graph I've decided to categorize the Fare's in: less 10, 10 - 50, 50-100, more than 100

# In[ ]:


def categorize_fare(fare):
    if fare < 10:
        return 0
    elif fare < 50:
        return 1
    elif fare < 100:
        return 2
    else:
        return 3
    
data_filled['Fare_Categorical'] = data_filled['Fare'].round().apply(categorize_fare)


# In[ ]:


survived = data_filled['Survived'] == 1
died = data_filled['Survived'] == 0
men = data_filled['Sex'] == 'male'
women = data_filled['Sex'] == 'female'

# Histogram per of ages, based on the sex and the survival

bins = list(range(0,90,9))

fig, ax = plt.subplots(1, 3, figsize = (15,4))
ax[0].hist(data_filled[survived]['Age'], histtype='step', color='blue', bins = bins, label='Survived',linewidth=3)
ax[0].hist(data_filled[died]['Age'], histtype='step', color='red', bins = bins, label='Died', linewidth=3)
ax[0].legend()
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Number of passengers')
ax[0].set_title('Histogram of ages')


ax[1].hist(data_filled[women & survived]['Age'], histtype = 'step', color='red', bins = bins, label = 'Women', linewidth=3)
ax[1].hist(data_filled[men & survived]['Age'], histtype = 'step', color='blue',  bins = bins, label = 'Men', linewidth=3)
ax[1].legend()
ax[1].set_xlabel("Age")
ax[1].set_title("Survivors per age and Sex")

ax[2].hist(data_filled[women & died]['Age'], histtype = 'step', color='red', bins = bins, label = 'Woman', linewidth=3)
ax[2].hist(data_filled[men & died]['Age'], histtype = 'step', color='blue', bins = bins, label = 'Men', linewidth=3)
ax[2].legend()
ax[2].set_xlabel("Age")
ax[2].set_title("Passengers died per age and sex")


# The graph above shows that the most part of the passengers who died were man, and also, among the passengers who survived the marjority were women even though there were more men than women on board, so we can see that sex is a great parameter to decide figure out if a passenger survived or not.
# Age doesn't tell too much about it...

# In[ ]:


def categorize_age(age):
    if age < 10:   # Child
        return 0
    elif age < 20: # Mid Age
        return 1
    elif age < 55: # Adult
        return 2
    else:          # Elderly
        return 3
    
data_filled['Age_Categorical'] = data_filled['Age'].round().apply(categorize_age)


# In[ ]:


first_c = data_filled[data_filled['Pclass'] == 1]
second_c = data_filled[data_filled['Pclass'] == 2]
third_c = data_filled[data_filled['Pclass'] == 3]
print(first_c.shape, second_c.shape, third_c.shape)


# In[ ]:


# Percentage of survival of each class
first_c = 100*first_c['Survived'].value_counts()/first_c['Survived'].count()
second_c = 100*second_c['Survived'].value_counts()/second_c['Survived'].count()
third_c = 100*third_c['Survived'].value_counts()/third_c['Survived'].count()

percentages = pd.concat([first_c, second_c, third_c], axis=1, keys=['First', 'Second', 'Third']).transpose()

fig, ax = plt.subplots(figsize = (6,3))
ax.bar(percentages.index, percentages[1], label = 'Survived', color = 'blue', alpha = 0.6)
ax.set_xlabel("Class")
ax.set_ylabel("Percentage")
ax.set_title("Percentage of survival in each class")
ax.legend()


# As it was expected the people who were in the first class were more likely to survive.

# In[ ]:


embarked = data_filled[data_filled['Survived'].notna()]['Embarked'].value_counts()
emb_survived = data_filled[survived]['Embarked'].value_counts()
emb_survived_men = data_filled[survived & men]['Embarked'].value_counts()
emb_survived_women = data_filled[survived & women]['Embarked'].value_counts()
percentages = 100*emb_survived/embarked
percentages_men = 100*emb_survived_men/embarked.values
percentages_women = 100*emb_survived_women/embarked.values

fig, ax = plt.subplots(1, figsize=(6,4))
ax.bar(percentages.index, percentages.values, label='Survived')
ax.set_title('Percentage of survival per boarding place')
ax.set_ylabel('Percentage')
ax.legend()


# It seems like the boarding place did not implies whether the passenger survived or not. But still says something, since the most part of the passengers embarked in 'S', and 'S' has the lowest percentage of wurvival we can imply that the most part of the passengers who died embarked in 'S'.
# I'll encode the variable in ascending order considering the rate of survival.

# In[ ]:


data_filled['Embarked'] = data_filled['Embarked'].replace({'C': 0, 'Q': 1, 'S':2})


# In[ ]:


# Adding the 'Family' Feature
data_filled['Family'] = data_filled['SibSp'] + data_filled['Parch']

# Lets see famiily
family = data_filled[data_filled['Survived'].notna()]['Family'].value_counts()
family_survived = data_filled[survived]['Family'].value_counts().rename("family_survived")
family_all = pd.concat([family, family_survived], axis = 1).replace(np.nan, 0)

fig, ax = plt.subplots(1, figsize=(7,5))
ax.bar(family_all.index, 100*family_all.family_survived/family_all.Family)
ax.set_title('Percentage of survival due the number of family members')


# Through this graphic we can see that people with small families on board were more likely to survive. Also, just 30% of the passengers who didn't have family on board survived even though they represent the marjority on board. Therefore I'll encode it as No family (0), Small Family (1 to 3 relatives) and big family (4 or more relatives).

# In[ ]:


# 3 categories to Family (No Family, 1-3, 4 or more)

def categorize_family(members):
    if members == 0:    # No Family
        return 0
    elif members < 4:   # 1 - 3 members
        return 1
    else:               # 4 or more members
        return 2
    
data_filled['Family'] = data_filled['Family'].apply(categorize_family)


# In[ ]:


# Analyzing the feature 'Name'

cv = CountVectorizer()
count_names = cv.fit_transform(data_filled.Name)
word_count = pd.DataFrame(cv.get_feature_names(), columns = ['word'])
word_count['count'] = count_names.sum(axis=0).tolist()[0]
word_count = word_count.sort_values("count", ascending = False).reset_index(drop=True)

#word_count[0:50]
word_count[0:5]


# As we can see there are some titles in the names, like Mr., Miss., Mrs, Master, Jr., Dr., and Rev., so i'll keep this information in a new attribute.

# In[ ]:


def extract_title(name):
    name = name.lower().replace(".", "")

    titles = ['mr', 'miss', 'mrs', 'master', 'dr', 'rev']
    others = ['don', 'dona', 'sir','mme', 'mlle', 'ms', 'major', 'capt', 'lady', 'col', 'countess', 'jonkheer']
    
    for word in name.split():
        if word in titles:
            return word
        elif word in others:
            return 'other'
        
data_filled['Title'] = data_filled['Name'].apply(extract_title)
data_filled['Title'].value_counts()


# In[ ]:


titles = data_filled[data_filled['Survived'].notna()]['Title'].value_counts()
titles_survived = data_filled[survived]['Title'].value_counts().rename('titles_survived')
titles_all = pd.concat([titles, titles_survived], axis = 1).replace(np.nan, 0)
titles_all['Percentage'] = 100*titles_all.titles_survived/titles_all.Title
titles_all.sort_values(by='Percentage', inplace = True)
plt.bar(titles_all.index, titles_all.Percentage)


# In[ ]:


def categorize_title(title):
    if title == 'rev':      # 0 - Reverend
        return 0
    elif title == 'mr':     # 1 - Mister
        return 1
    elif title == 'dr':     # 2 - Doctor or Master
        return 2
    elif title == 'master': 
        return 2
    elif title == 'other':  # 3 - Other
        return 3
    elif title == 'miss':   # 4 - Miss or Misses
        return 4
    else:
        return 4
    
    
data_filled['Title'] = data_filled['Title'].apply(categorize_title)


# In[ ]:


# Sex 
data_filled['Sex'] = data_filled['Sex'].replace(['male', 'female'], [0,1])


# In[ ]:


data_filled.drop(['Name', 'Parch', 'Age', 'Fare', 'SibSp', 'Ticket', 'Cabin'], axis='columns', inplace = True)


# In[ ]:


data_filled.head()


# In[ ]:


train = data_filled[data_filled['Survived'].notna()]
test = data_filled[data_filled['Survived'].isna()]


# In[ ]:


fig, ax = plt.subplots(figsize = (9,9))
sns.heatmap(train.corr(), center = 0, annot = True, square = True, cmap = 'YlGn', ax = ax, fmt='.2g', linewidths=3)


# As we can se, the feature 'Title' is strongly correlated with the variable Sex, so I'll not use in the models.

# ## Building the models

# In[ ]:


# Required libraries
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[ ]:


X = train.loc[:, 'Pclass':'Family']
y = train.loc[:, 'Survived']


# I'll test 5 algorithms to predict the output:
# 
# * K-nearest Neighbours (Knn)
# * Logistic Regression
# * Decision Tree Classifier
# * Random Forest
# * Support Vector Machine (svm)
# 
# And use the cross validation score to pick up the greatest one to apply on the data.

# In[ ]:


knn = KNeighborsClassifier()
logreg = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svm = SVC()

classifiers = [knn, logreg, dt, rf, svm]

results = []
for classifier in classifiers:
    results.append(cross_val_score(classifier, X,y, cv=5).mean())

results


# **SUPPORT VECTOR MACHINE** got the greatest score, so I'll use it to predict the samples and submit.

# In[ ]:


X_submission = test.loc[:, 'Pclass':'Family']
submission = test.loc[:, ['PassengerId', 'Survived']]


# In[ ]:


svm.fit(X, y)
y_submission = svm.predict(X_submission).astype('int')
submission['Survived'] = y_submission
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

