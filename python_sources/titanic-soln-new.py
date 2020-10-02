#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


# Import charting libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Ignore warnings thrown by Seaborn
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


# In[5]:


print(train.columns.values)
train.describe()


# In[6]:


# Understand the datatypes
print(train.dtypes)
print()
# Focus first on null values
print(train.isna().sum())


# In[7]:


#for test data 
print(test.dtypes)
print()
print(test.isna().sum())


# In[8]:


# seeing different features for their relation with survival
# Lets see the relation between Pclass and Survived
print(train[['Pclass', 'Survived']].groupby(['Pclass']).mean())
sns.catplot(x='Pclass', y='Survived',  kind='bar', data=train)


# In[9]:


#gender
print(train[['Sex', 'Survived']].groupby(['Sex']).mean())
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train)


# In[10]:


#pclass and gender together

sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass')


# In[11]:


print(train[['Embarked', 'Survived']].groupby(['Embarked']).mean())
sns.catplot(x='Embarked', y='Survived',  kind='bar', data=train)


# In[12]:


sns.catplot('Pclass', kind='count', col='Embarked', data=train)


# In[13]:


#SibSp and Parch: family size of the passengers.
print(train[['SibSp', 'Survived']].groupby(['SibSp']).mean())
sns.catplot(x='SibSp', y='Survived', data=train, kind='bar')


# In[14]:


print(train[['Parch', 'Survived']].groupby(['Parch']).mean())
sns.catplot(x='Parch', y='Survived', data=train, kind='bar')


# In[15]:


print(train[['Parch', 'Survived']].groupby(['Parch']).mean())
sns.catplot(x='Parch', y='Survived', data=train, kind='bar')


# In[16]:


train.head()


# In[17]:


#name  we will get titles
# Get the titles
for dataset in [train, test]:
    # Use split to get only the titles from the name
    dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # Check the initial list of titles.
    print(dataset['Title'].value_counts())
    print()
 


# In[18]:


sns.catplot(x='Survived', y='Title', data=train, kind ='bar')


# In[19]:


for df in [train, test]:
    print(df.shape)
    print()
    print(df.isna().sum())


# In[20]:


# Drop rows with nulls for Embarked
for df in [train, test]:
    df.dropna(subset = ['Embarked'], inplace = True)


# In[21]:


print(train[train['Fare'].isnull()])
print() 
# 1 row with null Fare in validation
print(test[test['Fare'].isnull()])
# We can deduce that Pclass should be related to Fares.
sns.catplot(x='Pclass', y='Fare', data=test, kind='point')


# In[22]:


# There is a clear relation between Pclass and Fare. We can use this information to impute the missing fare value.
# We see that the passenger is from Pclass 3. So we take a median value for all the Pclass 3 fares.
test['Fare'].fillna(test[test['Pclass'] == 3].Fare.median(), inplace = True)


# In[23]:


#age
print(train[['Age','Title']].groupby('Title').mean())
sns.catplot(x='Age', y='Title', data=train, kind ='bar')


# In[24]:


# Returns titles from the passed in series.
def getTitle(series):
    return series.str.split(',').str[1].str.split('.').str[0].str.strip()
# Prints the count of titles with nulls for the train dataframe.
print(getTitle(train[train.Age.isnull()].Name).value_counts())
# Fill Age median based on Title
mr_mask = train['Title'] == 'Mr'
miss_mask = train['Title'] == 'Miss'
mrs_mask = train['Title'] == 'Mrs'
master_mask = train['Title'] == 'Master'
dr_mask = train['Title'] == 'Dr'
train.loc[mr_mask, 'Age'] = train.loc[mr_mask, 'Age'].fillna(train[train.Title == 'Mr'].Age.mean())
train.loc[miss_mask, 'Age'] = train.loc[miss_mask, 'Age'].fillna(train[train.Title == 'Miss'].Age.mean())
train.loc[mrs_mask, 'Age'] = train.loc[mrs_mask, 'Age'].fillna(train[train.Title == 'Mrs'].Age.mean())
train.loc[master_mask, 'Age'] = train.loc[master_mask, 'Age'].fillna(train[train.Title == 'Master'].Age.mean())
train.loc[dr_mask, 'Age'] = train.loc[dr_mask, 'Age'].fillna(train[train.Title == 'Dr'].Age.mean())
# Prints the count of titles with nulls for the train dataframe. -- Should be empty this time.
print()
print(getTitle(train[train.Age.isnull()].Name).value_counts())                                                                       


# In[25]:


# Prints the count of titles with nulls for the validation dataframe.
print(getTitle(test[test.Age.isnull()].Name).value_counts())
# Fill Age median based on Title
mr_mask = test['Title'] == 'Mr'
miss_mask = test['Title'] == 'Miss'
mrs_mask = test['Title'] == 'Mrs'
master_mask = test['Title'] == 'Master'
ms_mask = test['Title'] == 'Ms'
test.loc[mr_mask, 'Age'] = test.loc[mr_mask, 'Age'].fillna(test[test.Title == 'Mr'].Age.mean())
test.loc[miss_mask, 'Age'] = test.loc[miss_mask, 'Age'].fillna(test[test.Title == 'Miss'].Age.mean())
test.loc[mrs_mask, 'Age'] = test.loc[mrs_mask, 'Age'].fillna(test[test.Title == 'Mrs'].Age.mean())
test.loc[master_mask, 'Age'] = test.loc[master_mask, 'Age'].fillna(test[test.Title == 'Master'].Age.mean())
test.loc[ms_mask, 'Age'] = test.loc[ms_mask, 'Age'].fillna(test[test.Title == 'Miss'].Age.mean())
# Prints the count of titles with nulls for the validation dataframe. -- Should be empty this time.
print(getTitle(test[test.Age.isnull()].Name).value_counts())


# In[26]:


print(train.isna().sum())
print(test.isna().sum())


# In[27]:


train.drop(columns=['PassengerId'], inplace = True)
[df.drop(columns=['Ticket'], inplace = True) for df in [train, test]]


# In[28]:


# encode all the categorical features.
[train, test] = [pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked']) for df in [train, test]]


# In[29]:


## convert the Cabin data into a flag about whether a passenger had an assigned cabin or not. Also we will use SibSp and Parch to calculate the Family Size and a flag named IsAlone
for df in [train, test]:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)


# In[30]:


[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [train,test]]


# In[31]:


train['Title'] = train['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
test['Title'] = test['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
[df.drop(columns=['Name'], inplace = True) for df in [train, test]]
[train, test] = [pd.get_dummies(data = df, columns = ['Title']) for df in [train, test]]


# In[32]:


# Check the updated dataset
print(train.columns.values)
print(test.columns.values)


# In[33]:


train.head()


# In[34]:


a = train.drop(['Survived'],axis=1)
a
a.columns.values
cols = a.columns.values
cols


# In[35]:


X = a.iloc[:, 0:18].values


# In[36]:


y = train['Survived']


# In[37]:


from sklearn.ensemble import RandomForestRegressor


# In[38]:


def feature_select(X, y, cols, cutoff):
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X, y)
    feat_imps = pd.concat([pd.DataFrame(cols, columns=['Features']),
                       pd.DataFrame(regressor.feature_importances_, columns=['Importances'])],
                     axis=1)
    feat_imps = feat_imps.sort_values(['Importances'], ascending=False)
    feat_imps['Cumulative Importances'] = feat_imps['Importances'].cumsum()
    feat_imps = feat_imps[feat_imps['Cumulative Importances'] < cutoff]
    return feat_imps['Features'].tolist()


# In[39]:


imp_cols = feature_select(X, y, cols, 0.95)
imp_cols


# In[40]:


X = train[imp_cols].values
X.shape


# In[41]:


y = train['Survived']


# In[42]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 0)


# In[43]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[44]:


models = []
acc = []
precision = []
recall = []
f1 = []


# In[45]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)
models.append('Logistic Regression')


# In[46]:


lr.predict(X_test)


# In[47]:


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', 
                                    random_state = 0)
dt.fit(X_train, y_train)
models.append('Decision Trees')


# In[48]:


# Fitting SVM to the Training set
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 0)
svc.fit(X_train, y_train)
models.append('SVM')


# In[49]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 350, criterion = 'entropy', 
                                    random_state = 0)
rf.fit(X_train, y_train)
models.append('Random Forest')


# In[50]:


# Fitting AdaBoost Classification to the Training set
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(base_estimator=dt, n_estimators=50, 
                         algorithm='SAMME.R', random_state=40)
adb.fit(X_train, y_train)
models.append('AdaBoost')


# In[51]:


# Fitting Voting Classifier Classification to the Training set
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators=[('Logistic Regression',lr),
                                   ('SVM',svc),
                                   ('Decision Tree',dt),
                                   ('Random Forest',rf),
                                   ('AdaBoost',adb)], 
                       voting='hard')
                       #flatten_transform=True)
vc.fit(X_train, y_train)
models.append('Average Ensemble')


# In[52]:


# Fitting Voting Classifier Classification to the Training set
from sklearn.ensemble import VotingClassifier
vc2 = VotingClassifier(estimators=[('Logistic Regression',lr),
                                   ('SVM',svc),
                                   ('Decision Tree',dt),
                                   ('Random Forest',rf),
                                   ('AdaBoost',adb)],
                      voting='soft',
                      flatten_transform=True, 
                      weights=[1,5,2,4,3])
vc2.fit(X_train, y_train)


# In[53]:


#evaluation
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score)


# In[54]:


print('Confusion Matrix for LR: \n',confusion_matrix(y_test, lr.predict(X_test)))
print('Accuracy for LR: \n',accuracy_score(y_test, lr.predict(X_test)))
acc.append(accuracy_score(y_test, lr.predict(X_test)))
print('Precision for LR: \n',precision_score(y_test, lr.predict(X_test)))
precision.append(precision_score(y_test, lr.predict(X_test)))
print('Recall for LR: \n',recall_score(y_test, lr.predict(X_test)))
recall.append(recall_score(y_test, lr.predict(X_test)))
print('f1_score for LR: \n',f1_score(y_test, lr.predict(X_test)))
f1.append(f1_score(y_test, lr.predict(X_test)))


# In[55]:


print('Confusion Matrix for DTrees: \n',confusion_matrix(y_test, dt.predict(X_test)))
print('Accuracy for DTrees: \n',accuracy_score(y_test, dt.predict(X_test)))
acc.append(accuracy_score(y_test, dt.predict(X_test)))
print('Precision for DTrees: \n',precision_score(y_test, dt.predict(X_test)))
precision.append(precision_score(y_test, dt.predict(X_test)))
print('Recall for DTrees: \n',recall_score(y_test, dt.predict(X_test)))
recall.append(recall_score(y_test, dt.predict(X_test)))
print('f1_score for DTrees: \n',f1_score(y_test, dt.predict(X_test)))
f1.append(f1_score(y_test, dt.predict(X_test)))


# In[56]:


print('Confusion Matrix for SVM: \n',confusion_matrix(y_test, svc.predict(X_test)))
print('Accuracy for SVM: \n',accuracy_score(y_test, svc.predict(X_test)))
acc.append(accuracy_score(y_test, svc.predict(X_test)))
print('Precision for SVM: \n',precision_score(y_test, svc.predict(X_test)))
precision.append(precision_score(y_test, svc.predict(X_test)))
print('Recall for SVM: \n',recall_score(y_test, svc.predict(X_test)))
recall.append(recall_score(y_test, svc.predict(X_test)))
print('f1_score for SVM: \n',f1_score(y_test, svc.predict(X_test)))
f1.append(f1_score(y_test, svc.predict(X_test)))


# In[57]:


print('Confusion Matrix for RF: \n',confusion_matrix(y_test, rf.predict(X_test)))
print('Accuracy for RF: \n',accuracy_score(y_test, rf.predict(X_test)))
acc.append(accuracy_score(y_test, rf.predict(X_test)))
print('Precision for RF: \n',precision_score(y_test, rf.predict(X_test)))
precision.append(precision_score(y_test, rf.predict(X_test)))
print('Recall for RF: \n',recall_score(y_test, rf.predict(X_test)))
recall.append(recall_score(y_test, rf.predict(X_test)))
print('f1_score for RF: \n',f1_score(y_test, rf.predict(X_test)))
f1.append(f1_score(y_test, rf.predict(X_test)))


# In[58]:


print('Confusion Matrix for ADB: \n',confusion_matrix(y_test, adb.predict(X_test)))
print('Accuracy for ADB: \n',accuracy_score(y_test, adb.predict(X_test)))
acc.append(accuracy_score(y_test, adb.predict(X_test)))
print('Precision for ADB: \n',precision_score(y_test, adb.predict(X_test)))
precision.append(precision_score(y_test, adb.predict(X_test)))
print('Recall for ADB: \n',recall_score(y_test, adb.predict(X_test)))
recall.append(recall_score(y_test, adb.predict(X_test)))
print('f1_score for ADB: \n',f1_score(y_test, adb.predict(X_test)))
f1.append(f1_score(y_test, adb.predict(X_test)))


# In[59]:


print('Confusion Matrix for VC: \n',confusion_matrix(y_test, vc.predict(X_test)))
print('Accuracy for VC: \n',accuracy_score(y_test, vc.predict(X_test)))
acc.append(accuracy_score(y_test, vc.predict(X_test)))
print('Precision for VC: \n',precision_score(y_test, vc.predict(X_test)))
precision.append(precision_score(y_test, vc.predict(X_test)))
print('Recall for VC: \n',recall_score(y_test, vc.predict(X_test)))
recall.append(recall_score(y_test, vc.predict(X_test)))
print('f1_score for VC: \n',f1_score(y_test, vc.predict(X_test)))
f1.append(f1_score(y_test, vc.predict(X_test)))


# In[60]:


model_dict = {'Models': models,
             'Accuracies': acc,
             'Precision': precision,
             'Recall': recall,
             'f1-score': f1}


# In[61]:


model_df = pd.DataFrame(model_dict)
model_df


# In[62]:


model_df = model_df.sort_values(['Accuracies', 'f1-score', 'Recall', 'Precision'],
                               ascending=False)


# In[63]:


model_df


# In[64]:


#Hyper parameter tuning


# In[65]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lr, 
                             X = X_train, 
                             y = y_train, 
                             cv = 10)
acMean = accuracies.mean()
acStd = accuracies.std()


# In[66]:


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {"n_estimators": [100, 200,350],
              "criterion":['gini','entropy'],
            "min_samples_split": [5,10,20,40],
              "min_samples_leaf": [1,5,15,40],
              "min_weight_fraction_leaf": [0, 0.1,0.05]
            }
grid_search = GridSearchCV(estimator = rf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)


# In[67]:


best_parameters = grid_search.best_params_


# In[68]:


# Fitting Final Model on training set
from sklearn.ensemble import RandomForestClassifier
tunedRF = RandomForestClassifier(n_estimators = best_parameters["n_estimators"],
                                 criterion = best_parameters["criterion"],
                                 min_samples_split = best_parameters["min_samples_split"],
                                 min_samples_leaf = best_parameters["min_samples_leaf"],
                                 min_weight_fraction_leaf = best_parameters["min_weight_fraction_leaf"]   )
tunedRF.fit(X_train, y_train)


# In[69]:


print('Confusion Matrix for Tuned RF: \n',confusion_matrix(y_test, tunedRF.predict(X_test)))
print('Accuracy for Tuned RF: \n',accuracy_score(y_test, tunedRF.predict(X_test)))
acc.append(accuracy_score(y_test, tunedRF.predict(X_test)))
print('Precision for Tuned RF: \n',precision_score(y_test, tunedRF.predict(X_test)))
precision.append(precision_score(y_test, tunedRF.predict(X_test)))
print('Recall for Tuned RF: \n',recall_score(y_test, tunedRF.predict(X_test)))
recall.append(recall_score(y_test, tunedRF.predict(X_test)))
print('f1_score for Tuned RF: \n',f1_score(y_test, tunedRF.predict(X_test)))
f1.append(f1_score(y_test, tunedRF.predict(X_test)))


# In[70]:


# Now we will pass the validation set provided for creating our submission
# Pick the same columns as in X_test
X_validation = test[imp_cols].values
X_validation
X_validation = sc.transform(X_validation)
# Call the predict from the created classifier
y_valid = rf.predict(X_validation)


# In[71]:


validation_pId = test.loc[:, 'PassengerId']
my_submission = pd.DataFrame(data={'PassengerId':validation_pId, 'Survived':y_valid})


# In[72]:


print(my_submission['Survived'].value_counts())


# In[73]:


my_submission.to_csv('submission1.csv', index = False)


# In[ ]:




