#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# Lets see the data.

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


print('Shape of train data:',train_data.shape)
print('Shape of test data:',test_data.shape)


# In[ ]:


print(train_data.info())


# #### Missing in Training Data
# * Age (can be filled)
# * Cabin (with lots of values)
# * Embarked (with 2 values)

# In[ ]:


print(test_data.info())


# #### Missing in Testing Data
# * Age (can be filled)
# * Fare (only 1 value)
# * Cabin (with lots of values)

# Lets visualize our data with respect to our target variable that is 'Survival'.

# I am starting with the interesting plot. I am plotting the 'Age' feature with the 'Pclass' and seperating them according to their 'Survival' output.

# In[ ]:


g = sns.FacetGrid(train_data, col = 'Survived', row = 'Pclass', size = 4, aspect = 1.5)
g.map(plt.hist, 'Age', bins=20)
g.add_legend();


# #### Findings
# * People in 3rd Pclsas were not able to survive while People of 1st Pclass survived the most.
# * People with age from 20 to 30 were not able to Survive and they belong to 3rd Pclass. Whereas people with age 30 to 40 survived from 1st Pclass.
# * People with age 15 to 30 of 3rd Pclass survived too.
# * All children below 10 from 2nd Pclass were able to survive.
# etc...

# In[ ]:


plt.figure(figsize = (16, 9))
g = sns.FacetGrid(train_data, col = 'Survived', size = 4, aspect = 1.5)
g.map(plt.hist, 'Age', bins = 20)
plt.show()


# #### Findings
# * From the above Graph it is clear that most of the People aboard were of age ranging from 20 to 40.
# * Only 1 old person survived with aged 79 that too from 1st Pclass (from previous plot).
# etc...

# In[ ]:


g = sns.FacetGrid(train_data, row = 'Embarked', size = 4, aspect = 1.5)
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
g.add_legend()


# #### Findings
# I got alot from this graph as findings.
# * Females have higher surviving rate than males.
# * Male who embarked from Cherbourg has high surviving rate as compared to male embarked from other ports.
# etc...

# In[ ]:


g = sns.FacetGrid(train_data, row = 'Embarked', col = 'Survived', size = 4, aspect = 1.5)
g.map(sns.barplot, 'Sex', 'Fare')
g.add_legend()


# #### Findings
# * Fare from Queenstown port is lower than other ports.
# * High fare passengers have better surviving rate.
# * Fare could be the correlating feature for survival (WOW!!). High Fare could be the indication of 1st Pclass which I already explored, and they have higher surviving rate.

# Getting Survival in the seperate variable called 'target'.

# In[ ]:


target = train_data['Survived']
train_data.drop(['Survived'], axis = 1, inplace = True)


# I will combine both training and testing set for some more Exploring and then I will use this complete set for PreProcessing as well.

# In[ ]:


full_data = pd.concat([train_data, test_data], axis = 0)


# In[ ]:


print('Shape of Complete Data:',full_data.shape)


# In[ ]:


plt.figure(figsize = (16,9))
sns.distplot(full_data['Age'])
plt.title("Distribution of Passenger's Age on Titanic")
plt.show()


# So most of the passengers are of age from roughly 20 to 40. Still we have to impute values in this column as there are lots of missing values in this column and this column with no doubt plays an important role in the Survival.

# In[ ]:


plt.figure(figsize = (17,5))
plt.subplot(1,4,1)
sns.countplot(target)
plt.title('Number of Passengers Survived and Not Survived')
plt.subplot(1,4,2)
sns.countplot(full_data['Pclass'])
plt.title('Number of Pclasses in the complete Dataset')
plt.subplot(1,4,3)
sns.countplot(full_data['Sex'])
plt.title('Number of M/F in the complete Dataset')
plt.subplot(1,4,4)
sns.countplot(full_data['Embarked'])
plt.title('Boarding ports in the complete Dataset')
plt.tight_layout()


# #### Findings
# * Most of the passengers didn't survive. As only around ~330 people survived from 891.
# * Most of the passengers were from Pclass = 3.
# * Most of the passengers were Male.
# * Most of the passengers aboard from Southampton.

# In[ ]:


plt.figure(figsize = (16,5))
plt.subplot(1,2,1)
sns.countplot(full_data['SibSp'])
plt.title('Number of Siblings or Spouse Aboard')
plt.subplot(1,2,2)
sns.countplot(full_data['Parch'])
plt.title('Number of Parents or Children Aboard')
plt.show()


# #### Findings
# * Most of the people were traveling alone.
# * Siblings or spouses were on the deck.
# * There were very few who was traveling with there family.

# I will Preprocess the whole Dataset at the same time.

# In[ ]:


full_data.info()


# * I don't think Columns like 'PassengerId' and 'Ticket' plays a very important role in Survival.
# * There are lots of null values in 'Cabin' column so I think removing it, is the best. We could have used it because by this we could know the best chance of Survival due to the position of the cabin on the ship but I don't know the structure of titanic so that is why I am removing it.
# * There is only 2 null values in 'Embarked' column so that will work.
# * *'Fare' also have just one null value but I think it is correlated to 'Pclass' so removal of this feature could be done sa feature reduction.*
# * 'Name' is the magic column I think, so I will keep it.
# * Columns like 'Age' and 'Sex' will play an important role in Survival so I will keep them.
# * Columns like 'SibSp' and 'Parch' can also be used to get a new column of family size.

# I will first drop columns Which I don't think will be correlated to the target column.

# In[ ]:


full_data.drop(['PassengerId', 'Ticket', 'Cabin', 'Fare'], axis = 1, inplace = True)


# In[ ]:


full_data.head()


# Lets take care of the first column which distract me the most is 'Name'.
# This column as it is I don't think has any other information but the title before every name. So, lets take these titles out from avery name of the dataset.

# In[ ]:


# Need to install this package by typing 'pip install nameparser' on the console.
# OR
get_ipython().run_line_magic('pip', 'install nameparser')
from nameparser import HumanName
full_data['Name_Title'] = full_data['Name'].apply(lambda x: HumanName(x).title)
print(full_data.Name_Title.value_counts())
unique_title = list(full_data.Name_Title.value_counts().index)


# Most of the names are starting with Mr, Miss, Mrs and Master. I will replace all other values with 'Special'.
# The information from the 'Name' column is extracted and as there does not seem to have other information from this column so I am deleting the 'Name' column from the dataset.

# In[ ]:


full_data.drop(['Name'], axis = 1, inplace = True)


# Seperating the top titles.

# In[ ]:


top_title = ['Mr.','Miss.','Mrs.','Master.']


# Replacing the values in the title column.

# In[ ]:


full_data['Name_Title'].replace(to_replace = list(set(unique_title) - set(top_title)),
                 value = 'Special', inplace = True)
# full_data.replace({'Name_Title':{'Mr.': 0, 'Miss.': 1, 'Mrs.': 2, 'Master.': 3, 'Special': 4}}, inplace = True)


# Lets take a quick look on our data.

# In[ ]:


full_data.head()


# Next I will convert 'Sex' to the more acceptable form.

# In[ ]:


full_data.replace({'Sex': {'male': 0, 'female': 1}}, inplace = True)


# In[ ]:


full_data.head()


# Lets do some Imputation Work.

# In[ ]:


full_data.isnull().sum()[full_data.isnull().sum() > 0].sort_values(ascending = False)


# All columns are complete except 2. So need to impute values in the columns 'Age' and 'Embarked'.

# To impute Age in the dataset I will not just impute it with mena or median beacause that will make the distribution of age abnormal. Rather I will try to guess their age with the help of Gender and Pclass.
# Thanks to https://www.kaggle.com/startupsci/titanic-data-science-solutions/ notebook for the idea.

# In[ ]:


guess_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = full_data[(full_data['Sex'] == i) & (full_data['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()

        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

for i in range(0, 2):
    for j in range(0, 3):
        full_data.loc[ (full_data.Age.isnull()) & (full_data.Sex == i) & (full_data.Pclass == j+1),'Age'] = guess_ages[i,j]

full_data['Age'] = full_data['Age'].astype(int)


# In[ ]:


full_data['Embarked'].fillna(full_data['Embarked'].mode()[0], inplace = True)


# In[ ]:


full_data.isnull().sum()[full_data.isnull().sum() > 0].sort_values(ascending = False)


# Series is empty implies no missing values left in the set.

# In[ ]:


full_data.info()


# I will create a new column which reprents the size of family. It will be made with the help of 'SibSp' and 'Parch'.

# In[ ]:


full_data['Family'] = full_data['SibSp'] + full_data['Parch'] + 1


# In[ ]:


full_data.head()


# Family_size of 1 represents that he or she traveled alone on Titanic. I think alone traveller has lower chance of survival than the one with family. So I can do one thing to convert the columns 'SibSp', 'Parch' and 'Family_size' into a single column which can define whether the traveller was alone on the deck or not.

# In[ ]:


full_data.loc[full_data['Family'] == 1, 'Family'] = 'Alone'
full_data.loc[full_data['Family'] == 2, 'Family'] = 'Partner'
full_data.loc[(full_data['Family'] != 'Alone') & (full_data['Family'] != 'Partner'), 'Family'] = 'More'


# In[ ]:


# full_data.loc[full_data['Family_size'] == 1, 'Alone'] = 1
# full_data.loc[full_data['Family_size'] != 1, 'Alone'] = 0
# full_data['Alone'] = full_data['Alone'].astype(int)


# In[ ]:


full_data.drop(['SibSp', 'Parch'], axis = 1, inplace = True)
print('Types of Passengers Travelling:',full_data.Family.value_counts())
full_data.head(10)


# Next, I will divide the Age column into parts. I think it will help in computation when I will use models on this dataset. It will be easier to find the splitting value for tree algorithms.
# This division can be used as experiment, like we can use different size of groups. I am dividing Age into 8 parts and assign value 0 to 7 to these.

# In[ ]:


pd.cut(full_data.Age, 8)


# So the last line of the output matters, now the Age is simply divided into 8 groups with each group has people belonging to 10 years of span.

# In[ ]:


full_data.loc[ full_data['Age'] <= 10, 'Age'] = 0
full_data.loc[(full_data['Age'] > 10) & (full_data['Age'] <= 20), 'Age'] = 1
full_data.loc[(full_data['Age'] > 20) & (full_data['Age'] <= 30), 'Age'] = 2
full_data.loc[(full_data['Age'] > 30) & (full_data['Age'] <= 40), 'Age'] = 3
full_data.loc[(full_data['Age'] > 40) & (full_data['Age'] <= 50), 'Age'] = 4
full_data.loc[(full_data['Age'] > 50) & (full_data['Age'] <= 60), 'Age'] = 5
full_data.loc[(full_data['Age'] > 60) & (full_data['Age'] <= 70), 'Age'] = 6
full_data.loc[ full_data['Age'] > 70, 'Age'] = 7


# In[ ]:


full_data.head()


# Now, I will make use of dummy variables.

# In[ ]:


dummy_df = pd.get_dummies(full_data, columns = ['Pclass', 'Embarked', 'Name_Title', 'Family'], drop_first = True)


# In[ ]:


dummy_df.head()


# Now, as I have finished the preprocessing part so lets move into the modelling but before that I am getting my training and testing set back.

# In[ ]:


train = dummy_df[:len(train_data)]
test = dummy_df[len(train_data):]


# In[ ]:


print('Shape of Training Data',train.shape)
print('Shape of Testing Data',test.shape)


# Lets move to Modelling.

# In[ ]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators = 100)

rf_clf.fit(train, target)
rf_pred = rf_clf.predict(test)
print('Accuracy Score:',round(rf_clf.score(train, target)*100,2))


# In[ ]:


from sklearn.svm import SVC

svc_clf = SVC(C = 2, kernel = 'rbf')

svc_clf.fit(train, target)
svc_pred = svc_clf.predict(test)
print('Accuracy Score:',round(svc_clf.score(train, target)*100,2))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(n_estimators = 100)

ada_clf.fit(train, target)
ada_pred = ada_clf.predict(test)
print('Accuracy Score:',round(ada_clf.score(train, target)*100,2))


# In[ ]:


from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier(n_estimators = 400)

lgbm_clf.fit(train, target)
lgbm_pred = lgbm_clf.predict(test)
print('Accuracy Score:',round(lgbm_clf.score(train, target)*100,2))


# In[ ]:


from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators = 200)

xgb_clf.fit(train, target)
xgb_pred = xgb_clf.predict(test)
print('Accuracy Score:',round(xgb_clf.score(train, target)*100,2))


# In[ ]:


from sklearn.ensemble import StackingClassifier

base_estimators = [('lgbm',lgbm_clf),
                  ('rf',rf_clf)]

stc_clf = StackingClassifier(estimators = base_estimators, final_estimator = svc_clf)

stc_clf.fit(train, target)
stc_pred = stc_clf.predict(test)
print('Accuracy Score:',round(stc_clf.score(train, target)*100,2))


# In[ ]:


from sklearn.ensemble import VotingClassifier

estimators = [('rf',rf_clf),
             ('xgb',xgb_clf),
             ('lgbm',lgbm_clf)]

vot_clf = VotingClassifier(estimators = estimators, voting = 'soft')

vot_clf.fit(train, target)
vot_pred = vot_clf.predict(test)
print('Accuracy Score:',round(vot_clf.score(train, target)*100,2))


# In[ ]:


submit_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submit_data.head()


# In[ ]:


submit_data['Survived'] = rf_pred
submit_data.head()


# In[ ]:


submit_data.to_csv('output.csv', index = False)


# ## Upvote if you like my work.
# #### There is still alot to do in this notebook which I will keep updating.
# ### Let me know the different Findings you get from the above Visualizations??
# ### And What Algorithm do you think will work Best on this Data??
