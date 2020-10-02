import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
print(train.info())
print(test.info())
#Use the Regular Expression to get the title from the name field.
import re
pattern = re.compile(r'.*?,(.*?)\.')
def getTitle(x):
    result = pattern.search(x)
    if result:
        return result.group(1).strip()
    else:
        return ''

train['Title'] = train['Name'].map(getTitle)
test['Title'] = test['Name'].map(getTitle)

#Check how many rows missing the Age by Title
print(train['Title'][train['Age'].isnull()].value_counts())
print(test['Title'][test['Age'].isnull()].value_counts())

#Set the missing Age of Title 'Master' 
master_age_mean = train['Age'][(train['Title']=='Master')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Master')&(train['Age'].isnull())].index, 'Age'] = master_age_mean
test.loc[test[(test['Title']=='Master')&(test['Age'].isnull())].index, 'Age'] = master_age_mean

#Set the missing Age of Title 'Mr' 
mr_age_mean = train['Age'][(train['Title']=='Mr')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Mr')&(train['Age'].isnull())].index, 'Age'] = mr_age_mean
test.loc[test[(test['Title']=='Mr')&(test['Age'].isnull())].index, 'Age'] = mr_age_mean

#Set the missing Age of Title 'Miss' or 'Ms'
miss_age_mean = train['Age'][(train['Title']=='Miss')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Miss')&(train['Age'].isnull())].index, 'Age'] = miss_age_mean
test.loc[test[((test['Title']=='Miss')|(test['Title']=='Ms'))&(test['Age'].isnull())].index, 'Age'] = miss_age_mean

#Set the missing Age of Title 'Mrs' 
mrs_age_mean = train['Age'][(train['Title']=='Mrs')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Mrs')&(train['Age'].isnull())].index, 'Age'] = mrs_age_mean
test.loc[test[(test['Title']=='Mrs')&(test['Age'].isnull())].index, 'Age'] = mrs_age_mean

#Set the missing Age of Title 'Dr' 
dr_age_mean = train['Age'][(train['Title']=='Dr')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Dr')&(train['Age'].isnull())].index, 'Age'] = dr_age_mean
test.loc[test[(test['Title']=='Mrs')&(test['Age'].isnull())].index, 'Age'] = dr_age_mean

print(train['Age'].describe())
print(test['Age'].describe())
import matplotlib.pyplot as plt
alpha = 0.6
fig = plt.figure(figsize=(8, 12))
grouped = train.groupby(['Survived'])
group0 = grouped.get_group(0)
group1 = grouped.get_group(1)

plot_rows = 5
plot_cols = 2
ax1 = plt.subplot2grid((plot_rows,plot_cols), (0,0), rowspan=1, colspan=1)
plt.hist([group0.Age, group1.Age], bins=16, range=(0,80), stacked=True, 
        label=['Not Survived', 'Survived'], alpha=alpha)
plt.legend(loc='best', fontsize='x-small')
ax1.set_title('Survival distribution by Age')

ax2 = plt.subplot2grid((plot_rows,plot_cols), (0,1), rowspan=1, colspan=1)
n, bins, patches = plt.hist([group0.Pclass, group1.Pclass], bins=5, range=(0,5), 
        stacked=True, label=['Not Survived', 'Survived'], alpha=alpha)
plt.legend(loc='best', fontsize='x-small')
ax2.set_xticks([1.5, 2.5, 3.5])
ax2.set_xticklabels(['Class1', 'Class2', 'Class3'], fontsize='small')
ax2.set_yticks([0, 150, 300, 450, 600, 750])
ax2.set_title('Survival distribution by Pclass')

ax3 = plt.subplot2grid((plot_rows,plot_cols), (1,0), rowspan=1, colspan=2)
ax3.set_title('Survival distribution by Sex')
patches, l_texts, p_texts = plt.pie(train.groupby(['Survived', 'Sex']).size(), 
        labels=['Not Survived Female', 'Not Survived Male', 'Survived Female', 'Survived Male'],
        autopct='%3.1f', labeldistance = 1.1, pctdistance = 0.6)
plt.legend(loc='upper right', fontsize='x-small')
for t in l_texts:
    t.set_size(10)
for p in p_texts:
    p.set_size(10)
#plt.legend(loc='best', fontsize='x-small')
plt.axis('equal')
sex_to_int = {'male':1, 'female':0}
train['SexInt'] = train['Sex'].map(sex_to_int)
embark_to_int = {'S': 0, 'C':1, 'Q':2}
train['EmbarkedInt'] = train['Embarked'].map(embark_to_int)
train['EmbarkedInt'] = train['EmbarkedInt'].fillna(0)
print(train.describe())
test['SexInt'] = test['Sex'].map(sex_to_int)
test['EmbarkedInt'] = test['Embarked'].map(embark_to_int)
test['EmbarkedInt'] = test['EmbarkedInt'].fillna(0)
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']
ticket = train[train['Parch']==0]
ticket = ticket.loc[ticket.Ticket.duplicated(False)]
grouped = ticket.groupby(['Ticket'])
#The Friends field indicate if the passenger has frineds/SibSp in the boat.
train['Friends'] = 0
#The below fields statistic how many are survived or not survived by sex.
train['Male_Friends_Survived'] = 0
train['Male_Friends_NotSurvived'] = 0
train['Female_Friends_Survived'] = 0
train['Female_Friends_NotSurvived'] = 0
for (k, v) in grouped.groups.items():
    for i in range(0, len(v)):
        train.loc[v[i], 'Friends'] = 1
        train.loc[v[i], 'Male_Friends_Survived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='male')&(train.Survived==1)].Survived.count()
        train.loc[v[i], 'Male_Friends_NotSurvived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='male')&(train.Survived==0)].Survived.count()
        train.loc[v[i], 'Female_Friends_Survived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='female')&(train.Survived==1)].Survived.count()
        train.loc[v[i], 'Female_Friends_NotSurvived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='female')&(train.Survived==0)].Survived.count()
test_ticket = test[test['Parch']==0]
test['Friends'] = 0
test['Male_Friends_Survived'] = 0
test['Male_Friends_NotSurvived'] = 0
test['Female_Friends_Survived'] = 0
test['Female_Friends_NotSurvived'] = 0

grouped = test_ticket.groupby(['Ticket'])
for (k, v) in grouped.groups.items():
    temp_df = train[train.Ticket==k]
    length = temp_df.shape[0]
    if temp_df.shape[0]>0:
        for i in range(0, len(v)):
            test.loc[v[i], 'Friends'] = 1
            test.loc[v[i], 'Male_Friends_Survived'] = temp_df[(temp_df.Sex=='male')&(temp_df.Survived==1)].shape[0]
            test.loc[v[i], 'Male_Friends_NotSurvived'] = temp_df[(temp_df.Sex=='male')&(temp_df.Survived==0)].shape[0]
            test.loc[v[i], 'Female_Friends_Survived'] = temp_df[(temp_df.Sex=='female')&(temp_df.Survived==1)].shape[0]
            test.loc[v[i], 'Female_Friends_NotSurvived'] = temp_df[(temp_df.Sex=='female')&(temp_df.Survived==0)].shape[0]
train['FatherOnBoard'] = 0
train['FatherSurvived'] = 0
train['MotherOnBoard'] = 0
train['MotherSurvived'] = 0
train['ChildOnBoard'] = 0
train['ChildSurvived'] = 0
train['ChildNotSurvived'] = 0
grouped = train[train.Parch>0].groupby('Ticket')
for (k, v) in grouped.groups.items():
    for i in range(0, len(v)):
        if train.loc[v[i], 'Age']<19:
            temp = train[(train.Ticket==k)&(train.Age>18)]
            if temp[temp.SexInt==1].shape[0] == 1:
                train.loc[v[i], 'FatherOnBoard'] = 1
                train.loc[v[i], 'FatherSurvived'] = temp[temp.SexInt==1].Survived.sum()
            if temp[temp.SexInt==0].shape[0] == 1:
                train.loc[v[i], 'MotherOnBoard'] = 1
                train.loc[v[i], 'MotherSurvived'] = temp[temp.SexInt==0].Survived.sum()
        else:
            temp = train[(train.Ticket==k)&(train.Age<19)]
            length = temp.shape[0]
            if length>0:
                train.loc[v[i], 'ChildOnBoard'] = 1
                train.loc[v[i], 'ChildSurvived'] = temp[temp.Survived==1].shape[0]
                train.loc[v[i], 'ChildNotSurvived'] = temp[temp.Survived==0].shape[0]
test['FatherOnBoard'] = 0
test['FatherSurvived'] = 0
test['MotherOnBoard'] = 0
test['MotherSurvived'] = 0
test['ChildOnBoard'] = 0
test['ChildSurvived'] = 0
test['ChildNotSurvived'] = 0
grouped = test[test.Parch>0].groupby('Ticket')
for (k, v) in grouped.groups.items():
    temp = train[train.Ticket==k]
    length = temp.shape[0]
    if length>0:
        for i in range(0, len(v)):
            if test.loc[v[i], 'Age']<19:
                if temp[(temp.SexInt==1)&(temp.Age>18)].shape[0] == 1:
                    test.loc[v[i], 'FatherOnBoard'] = 1
                    test.loc[v[i], 'FatherSurvived'] = temp[(temp.SexInt==1)&(temp.Age>18)].Survived.sum()
                if temp[(temp.SexInt==0)&(temp.Age>18)].shape[0] == 1:
                    test.loc[v[i], 'MotherOnBoard'] = 1
                    test.loc[v[i], 'MotherSurvived'] = temp[(temp.SexInt==0)&(temp.Age>18)].Survived.sum()
            else:
                length = temp[temp.Age<19].shape[0]
                if length>0:
                    test.loc[v[i], 'ChildOnBoard'] = 1
                    test.loc[v[i], 'ChildSurvived'] = temp[(temp.Age<19)&(temp.Survived==1)].shape[0]
                    test.loc[v[i], 'ChildNotSurvived'] = temp[(temp.Age<19)&(temp.Survived==0)].shape[0]
fig = plt.figure(figsize=(8, 1))
grouped = train.groupby(['Survived'])
group0 = grouped.get_group(0)
group1 = grouped.get_group(1)

ax1 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=2)
bottom = group0.EmbarkedInt.value_counts().index
width1 = group0.EmbarkedInt.value_counts()
plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='Not Survived', alpha=0.6)
width2 = group1.EmbarkedInt.value_counts()
plt.barh(bottom, width2, 0.8, width1, color='green', label='Survived', alpha=0.6)
plt.legend(loc='best', fontsize='x-small')
ax1.set_yticks([0.4, 1.4, 2.4])
ax1.set_yticklabels(['Southampton', 'Cherbourg', 'Queenstown'], fontsize='small')
ax1.set_title('Survival distribution by Embarked')
title_to_int = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':1, 'Dr':4, 'Rev':4, 'Mlle':2, 'Major':4, 'Col':4,
        'Ms':3, 'Lady':3, 'the Countess':4, 'Sir':4, 'Mme':3, 'Capt':4, 'Jonkheer':4, 'Don':1, 'Dona':3}
train['TitleInt'] = train['Title'].map(title_to_int)
test['TitleInt'] = test['Title'].map(title_to_int)
train.loc[train[train['Age']<13].index, 'TitleInt'] = 5
test.loc[test[test['Age']<13].index, 'TitleInt'] = 5

train['FareCat'] = pd.cut(train['Fare'], [-0.1, 50, 100, 150, 200, 300, 1000], right=True, 
        labels=[0, 1, 2, 3, 4, 5])
test['FareCat'] = pd.cut(test['Fare'], [-0.1, 50, 100, 150, 200, 300, 1000], right=True, 
        labels=[0, 1, 2, 3, 4, 5])
train['AgeCat'] = pd.cut(train['Age'], [-0.1, 12.1, 20, 30, 35, 40, 45, 50, 55, 65, 100], right=True, 
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test['AgeCat'] = pd.cut(test['Age'], [-0.1, 12.1, 20, 30, 35, 40, 45, 50, 55, 65, 100], right=True, 
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
fig = plt.figure(figsize=(8, 1))
grouped = train.groupby(['Survived'])
group0 = grouped.get_group(0)
group1 = grouped.get_group(1)

ax1 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=2)
bottom = group0.TitleInt.value_counts().index
width1 = group0.TitleInt.value_counts()
plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='Not Survived', alpha=0.6)
width2 = group1.TitleInt.value_counts()
plt.barh(bottom, width2, 0.8, width1, color='green', label='Survived', alpha=0.6)
plt.legend(loc='best', fontsize='x-small')
ax1.set_yticks([1.4, 2.4, 3.4, 4.4, 5.4])
ax1.set_yticklabels(['Mr', 'Miss', 'Mrs', 'Profession', 'Child'], fontsize='small')
ax1.set_title('Survival distribution by Title')
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']
#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeInt', 'TitleInt', 'Fare']
#columns = ['Pclass', 'FamilySize', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']
#columns = ['Pclass', 'FamilySize', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat',
#        'Friends', 'FriendsSex', 'FriendsSurvived']
#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat',
#        'Friends', 'FriendsSex', 'FriendsSurvived', 'FatherSurvived', 'MotherSurvived', 'ChildSurvived']
#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']
#        'Friends', 'FriendsSex', 'FriendsSurvived']
#columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt','Fare', 'Friends', 'FriendsSex', 'FriendsSurvived', 'FatherSurvived', 'MotherSurvived', 'ChildSurvived']
columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt','Fare', 
        'Friends', 'Male_Friends_Survived', 'Male_Friends_NotSurvived', 'Female_Friends_Survived', 'Female_Friends_NotSurvived',
        'MotherOnBoard', 'MotherSurvived', 'ChildOnBoard', 'ChildSurvived', 'ChildNotSurvived']
X_train, X_test, y_train, y_test = train_test_split(train[columns], train['Survived'], test_size=0.2, random_state=123)

#Check the features importance. 
#selected = SelectKBest(f_classif, 18)
#selected.fit(X_train, y_train)
#X_train_selected = selected.transform(X_train)
#X_test_selected = selected.transform(X_test)
#print(selected.scores_)
#print(selected.pvalues_)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [10, 50, 100, 150], 'min_samples_leaf': [1, 2, 4, 8], 
        'max_depth': [None, 5, 10, 50], 'max_features': [None, 'auto'], 'min_samples_split': [2, 4, 8]}
rfc = RandomForestClassifier(criterion='gini', min_weight_fraction_leaf=0.0, 
        max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, 
        n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None)
classifer = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1)
#classifer.fit(X_train, y_train)
#print(classifer.grid_scores_)
#print(classifer.best_params_)
#print(X_train.info())
