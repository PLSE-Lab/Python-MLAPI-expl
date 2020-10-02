#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# # Import Data sets

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


train.info()


# In[ ]:


train.shape


# In[ ]:


train.isnull().sum()


# In[ ]:


test.info()


# In[ ]:


test.shape


# 'Survived' is the target variable, so test data set doesn't have it.

# In[ ]:


print("test data is "+str((test.shape[0]/(train.shape[0]+test.shape[0]))*100)+"% of total data")


# In[ ]:


train.describe()


# Now, let's look at some real data

# In[ ]:


train.head(10)


# # Exploratory Data Analysis

# Our target variable is 'Survived' (0 or 1) and the others are input variables (= features)    
# The input variables can be classified as follows:
# 
# * Continuous variables : Fare
# * Categorical variables : Pclass, Sex, Embarked
# * Ordinal variables : PassengerId, Age, SibSp, Parch
# * String variables : Name, Ticket, Cabin

# In[ ]:


plt.hist(train['Fare'], bins=50)
plt.xlabel('number of passengers')
plt.ylabel('fare')
plt.title('Distribution of Fares')
plt.show()


# In[ ]:


train.groupby('Sex')['PassengerId'].count()


# In[ ]:


sns.countplot('Sex', data=train)
plt.show()


# In[ ]:


train.groupby(['Sex','Survived'])['Survived'].count()


# In[ ]:


sns.countplot('Sex', hue='Survived', data=train)
plt.title('Sex vs Survived')
plt.show()


# To change the chart style, let's define the function:

# In[ ]:


def survival_bar(input):
    survived = train[train['Survived']==1][input].value_counts()
    dead = train[train['Survived']==0][input].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True)


# In[ ]:


survival_bar('Sex')


# We can see the same bar chart as stacked style.  
# From the above chart, we can see male are more likey to die than female. 

# In[ ]:


train.groupby('Pclass')['PassengerId'].count()


# In[ ]:


sns.countplot('Pclass', data=train)
plt.show()


# In[ ]:


train.groupby(['Pclass','Survived'])['Survived'].count()


# In[ ]:


sns.countplot('Pclass', hue='Survived', data=train)
plt.title('Pclass vs Survived')
plt.show()


# In[ ]:


survival_bar('Pclass')


# From the above chart, we can see the 1st flass passengers are more likely to survive but on the contrary, 3rd class passengers are more likely to die.

# Now, let's see our last categorical variable : Embarked

# In[ ]:


sns.countplot('Embarked', data=train)
plt.show()


# We can see the most passengers were embarked at S.

# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')


# In[ ]:


survival_bar('Embarked')


# We can see the passengers embarked at S are more likely to die.  
# To see if there is some relation between 'Embarked' and 'Pclass', let's check the correlation.

# In[ ]:


train.groupby(['Pclass','Embarked'])['Embarked'].count()


# In[ ]:


sns.countplot('Pclass', hue='Embarked', data=train)
plt.title('Pclass vs Embarked')
plt.show()


# Yes. Passengers embarked at S were mostly seated in 3rd class.   
# It explains the high fatality for S with the fact that most passengers were embarked at S.

# In[ ]:


#train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
#train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
#train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
train['Embarked'] = train['Embarked'].map(embarked_mapping)


# In[ ]:


corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# Oops, Sex is not included because it isn't numerical variable, let's convert it into integer variable:

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)


# In[ ]:


corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# From above, Fare, Plcass, Sex have meaningful correlation with Survived as we saw above.

# In[ ]:


survival_bar('SibSp')


# In[ ]:


survival_bar('Parch')


# From above two charts, sole travelers are more likely to die.

# In[ ]:


train.plot(kind="scatter", x="Age", y="Survived")


# In[ ]:


train.plot(kind="scatter", x="Age", y="Survived", figsize=(20,3))


# From the above plot, I cannot find any meaningful relation between Age and Survived.

# Let's convert Age into categorical variable and assign integers like this:
#           * Child : Age <= 10 -> 0
#           * Junior : 10 < Age <= 20 -> 1
#           * Adult : 20 < Age <= 40 -> 2
#           * Senior : 40 < Age <= 60 -> 3
#           * Old : 60 < Age -> 4

# In[ ]:


train.loc[train['Age'] <= 10, 'Age'] = 0,
train.loc[(train['Age'] > 10) & (train['Age'] <= 20), 'Age'] = 1,
train.loc[(train['Age'] > 20) & (train['Age'] <= 40), 'Age'] = 2,
train.loc[(train['Age'] > 40) & (train['Age'] <= 60), 'Age'] = 3,
train.loc[train['Age'] > 60, 'Age'] = 4


# In[ ]:


survival_bar('Age')


# After conversion, we can see children are more likely to survive.

# In[ ]:


corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# In[ ]:


train.head(10)


# # Missing value control

# Among meaningful features, Age and Embarked have missing values.  
# In the above when we analyze Embarked we simply imputed 'S' for missing values because most passengers were embarked at S.  
# For Age I decided to apply the median age for each title group which is contained in Name. 

# In[ ]:


train.info()


# In[ ]:


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train['Title'].value_counts()


# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
train['Title'] = train['Title'].map(title_mapping)


# In[ ]:


train.info()


# In[ ]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


train.isnull().sum()


# Cabin is a meaningless feature, we don't need to take care of missing values.

# Before applying prediction models, we need to discard useless features:

# In[ ]:


train_prepared = train.drop('Survived', axis=1)
train_label = train['Survived'].copy()


# In[ ]:


train_prepared = train_prepared.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[ ]:


train_prepared.info()


# In[ ]:


train_prepared.head(10)


# # Prediction Modeling

# ### Random Forest Classifier

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100,random_state = 42)
score = cross_val_score(rfc, train_prepared, train_label, cv=10, n_jobs=1, scoring='accuracy')
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# # Testing

# ### Preparing Test Data

# Before applying the prediction model, we need to prepare test data as we did for train data.

# In[ ]:


test.isnull().sum()


# Oops! There is a missing value in Fare. In training data set there was no missing value in Fare.  
# I will impute median value for each Pclass because Fare is closely related to Pclass.

# In[ ]:


test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[ ]:


test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


test['Title'] = test['Title'].map(title_mapping)


# In[ ]:


test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


test.loc[test['Age'] <= 10, 'Age'] = 0,
test.loc[(test['Age'] > 10) & (test['Age'] <= 20), 'Age'] = 1,
test.loc[(test['Age'] > 20) & (test['Age'] <= 40), 'Age'] = 2,
test.loc[(test['Age'] > 40) & (test['Age'] <= 60), 'Age'] = 3,
test.loc[test['Age'] > 60, 'Age'] = 4


# In[ ]:


sex_mapping = {"male": 0, "female": 1}
test['Sex'] = test['Sex'].map(sex_mapping)


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
test['Embarked'] = test['Embarked'].map(embarked_mapping)


# In[ ]:


test['Embarked'] = test['Embarked'].fillna('S')


# In[ ]:


test_prepared = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1).copy()


# ### Making Submission File

# In[ ]:


rfc = RandomForestClassifier(n_estimators = 100,random_state = 42)
rfc.fit(train_prepared, train_label)


# In[ ]:


prediction = rfc.predict(test_prepared)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)

