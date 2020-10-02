#!/usr/bin/env python
# coding: utf-8

# # Importing packages
# We start by importing the neede packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Creating the datasets
# Uploading and creating the training and test datasets

# In[ ]:


tr_data = pd.read_csv('/kaggle/input/titanic/train.csv')
tr_data.head()


# In[ ]:


te_data = pd.read_csv('/kaggle/input/titanic/test.csv')
te_data.head()


# # EDA

# In[ ]:


cols = tr_data.columns
print(cols)


# In[ ]:


tr_data.info()


# We realize from the analysis that 3 variables have missing info: Age, Cabin and Embarked.
# 1. Age: the age of the passengers seems to be important variable of survival in general. But has we have some missing, we shoudl try to find a proxy to replace the missing value. For instance, the price of the picket might have some correlation with the age.
# 2. Cabin: We can guess that a lot of people didn't have cabins. The next step is to see if it's correlated with the fare.
# 3. Embarked: It contains only two values missing values. This doesn't affect much the analysis

# ## 1. Tackling the Age problem

# We are using the class as a proxy for age, considering that older you are, the richer, and thus the higher class.

# ## 1.1 Do the current values look correct?

# In[ ]:


tr_data['Age'].describe()


# We see that the minimum is less than one year, and that the maximum is 80. That seems correct!

# ## 1.2 Changing the missing values

# I tried finding proxy true classes or so on, but it only decreased the final score. So I just went for the average.

# In[ ]:


mean_age = tr_data["Age"].mean()
print(mean_age)
tr_data['Age'] = tr_data['Age'].fillna(mean_age) 
te_data['Age'] = te_data['Age'].fillna(mean_age) 


# ## 1.3 Replacing age by dummies

# In order to make my RandomForest work, I need dummies. Hence here, I just split the age in 5 groups to create dummies. I created five equal layer between 0 and 80, so one per 16 years.

# In[ ]:



tr_data.loc[ tr_data['Age'] <= 16, "Age"] = 1
tr_data.loc[(tr_data['Age'] > 16) & (tr_data['Age'] <= 32), "Age"] = 2
tr_data.loc[(tr_data['Age'] > 32) & (tr_data['Age'] <= 48), "Age"] = 3
tr_data.loc[(tr_data['Age'] > 48) & (tr_data['Age'] <= 64), "Age"] = 4
tr_data.loc[ tr_data['Age'] > 64, "Age"] = 5

te_data.loc[ te_data['Age'] <= 16, "Age"] = 1
te_data.loc[(te_data['Age'] > 16) & (te_data['Age'] <= 32), "Age"] = 2
te_data.loc[(te_data['Age'] > 32) & (te_data['Age'] <= 48), "Age"] = 3
te_data.loc[(te_data['Age'] > 48) & (te_data['Age'] <= 64), "Age"] = 4
te_data.loc[ te_data['Age'] > 64, "Age"] = 5
    
tr_data["Age"].isnull().sum()
te_data["Age"].isnull().sum()


# ## 2.Analysis of the survival ratio per gender

# This is to prove that the gender has a strong predictive input and that we shoudl consider it.

# In[ ]:


women = tr_data.loc[tr_data['Sex']=='female']["Survived"]
rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)


# In[ ]:


men = tr_data.loc[tr_data['Sex']=='male']['Survived']
rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)


# ## 3. Creating a "title of nobility" column

# I am here working on the name column, and splitting it in pieces in order to isolate the title part.

# In[ ]:


name = tr_data['Name']

#for tr_data
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in tr_data["Name"]]
tr_data["Title"] = pd.Series(dataset_title)
tr_data["Title"] = tr_data["Title"].replace(["Master", "Dr", "Rev", "Col", "Major", "the Countess", "Capt", "Jonkheer", "Lady", "Sir","Don", "Dona", "Mlle", "Ms", "Mme"], 'Rare')
tr_data["Title"] = tr_data["Title"].replace(["Mr", "Miss", "Mrs"], 'Common')

#for te_data
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in te_data["Name"]]
te_data["Title"] = pd.Series(dataset_title)
te_data["Title"] = te_data["Title"].replace(["Master", "Dr", "Rev", "Col", "Major", "the Countess", "Capt", "Jonkheer", "Lady", "Sir", "Don", "Dona", "Mlle", "Ms", "Mme"], 'Rare')
te_data["Title"] = te_data["Title"].replace(["Mr", "Miss", "Mrs"], 'Common')
     
print(tr_data["Title"].value_counts())
print(te_data["Title"].value_counts())


# ## 4. Creating family size

# This whole section is rubbish and just decreased my score.

# In[ ]:


# #for train set
# tr_data['FamilySize'] = tr_data['SibSp'] + tr_data['Parch'] +1

# #for test set
# te_data['FamilySize'] = tr_data['SibSp'] + tr_data['Parch'] +1

# tr_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

From the above analysis, we see that it's not, the bigger, the better. We will thus create three groups: ideal family size, non-ideal familsy size and alone.

Edit: this code below decreased my score, I'll remove it.
# In[ ]:



# list = []

# for i in tr_data['FamilySize']:
#     if i in [4, 3, 2, 7]:
#         list.append("ideal")
#     elif i ==1:
#         list.append("alone")
#     else:
#         list.append("non_ideal")
        
# tr_data["IdealFamilySize"] = list

# #for te_data
# list2 = []

# for i in te_data['FamilySize']:
#     if i in [4, 3, 2, 7]:
#         list2.append("ideal")
#     elif i ==1:
#         list2.append("alone")
#     else:
#         list2.append("non_ideal")
        
# te_data["IdealFamilySize"] = list2
        
# print(te_data['FamilySize'].head(10))
# print(te_data['IdealFamilySize'].head(10))


# ## 5. Cabins vs no-cabins

# Also lowered my score: to be removed.

# In[ ]:


#for tr_data
cabin_list = []

tr_data['Cabin'] = tr_data['Cabin'].fillna('None')

for i in tr_data['Cabin']:
    if i == 'None':
        cabin_list.append('None')
    else:
        cabin_list.append('Cabin')

tr_data['Cabin'] = cabin_list

#for te_data
cabin_list2 = []

te_data['Cabin'] = te_data['Cabin'].fillna('None')

for i in te_data['Cabin']:
    if i == 'None':
        cabin_list2.append('None')
    else:
        cabin_list2.append('Cabin')

te_data['Cabin'] = cabin_list2

te_data['Cabin'].value_counts()


# ## 6.Fare 
# Let's try to see if fare brings something good.

# In[ ]:


#tr_data['Fare'] = pd.qcut(tr_data['Fare'], 4)
#tr_data[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True)

map(float(), tr_data['Fare'])

print(tr_data['Fare'].head())


# So we see 4 different groups that we will use to create 4 categories: Fare1, Fare2, Fare3, Fare4. This division is comes from the pd.qcut(tr_data['Fare'], 4)

# In[ ]:


#for tr_data
fare_list = []

for i in tr_data['Fare']:
    i = float(i)
    if i <= 7.91:
        fare_list.append(1)
    elif i <= 14.454:
        fare_list.append(2)
    elif i <= 31.0:
        fare_list.append(3)
    else:
        fare_list.append(4)
        
tr_data['Fare'] = fare_list

#for te_data
fare_list2 = []

for i in te_data['Fare']:
    i = float(i)
    if i <= 7.91:
        fare_list2.append(1)
    elif i <= 14.454:
        fare_list2.append(2)
    elif i <= 31.0:
        fare_list2.append(3)
    else:
        fare_list2.append(4)
        
te_data['Fare'] = fare_list2

print(tr_data['Fare'].head())
print(te_data['Fare'].head())


# # 7. Drop useless and creating the dummies

# let's drop useless variable: Name, ticket, PassengerId and Cabin

# In[ ]:


tr_data = tr_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
te_data = te_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# In[ ]:


features = ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked', "Title", "Fare"]

for i in features:
    tr_data [i] = pd.get_dummies(tr_data[i])
    te_data [i] = pd.get_dummies(te_data[i])

print(tr_data.head())


# Now let's analyze the correlations

# In[ ]:


df = pd.DataFrame(tr_data,columns=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked', "Title"])
corrMatrix = df.corr()
print (corrMatrix)


# We see that the most important variables are Sex, Pclass, Fare, Embarked, Parch, Age, SibSp. 
# On the other hand Title and Age have minor effects.
# Now we also have 3 big correlation: Age/Parch, SibSp/Parch, SibSp/Fare. we will thus create interaction term for those in 9.

# ## 8. Correlation and interaction terms

# Based on the correlation table, I thought that creating interraction terms would help. But it didn't, so I end up removing it from my final model.

# In[ ]:


# tr_data['IntAgeParch'] = tr_data['Age'] * tr_data['Parch']
# tr_data['IntSibParch'] = tr_data['SibSp'] * tr_data['Parch']
tr_data['IntSibFare'] = tr_data['Age'] * tr_data['Parch']
te_data['IntSibFare'] = te_data['Age'] * te_data['Parch']


# # Modelling

# While creating the model, I tried first the model as such, then add a GridSearch with few parameters to improve the score. But it didn't help, it jus decreased the score.

# In[ ]:


print(tr_data['Survived'].head(10))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

y = tr_data['Survived']

features = ['Pclass', 'Sex', 'Age', 'SibSp','Parch','Embarked', "Title"]
X = tr_data[features]
X_test = te_data[features]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


rfc=RandomForestClassifier(random_state=42)


# In[ ]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[ ]:


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)


# In[ ]:


CV_rfc.best_params_


# So we can see that the best parameters for the features 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked', "Title" is {'criterion': 'entropy',
#  'max_depth': 4,
#  'max_features': 'log2',
#  'n_estimators': 500}

# In[ ]:


model = RandomForestClassifier(max_depth= 4, max_features = 'auto', n_estimators = 500, criterion= 'entropy')
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': te_data.PassengerId, 'Survived':predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully save! :D")

