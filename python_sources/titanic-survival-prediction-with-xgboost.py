#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# 
# ## Looking into the training dataset
# 
# Printing first 5 rows of the train dataset.
# 

# In[ ]:




train.head()


# 
# ## Total rows and columns
# 
# We can see that there are 891 rows and 12 columns in our training dataset.
# 

# In[ ]:


train.shape


# ## Describing training dataset
# 
# describe() method can show different values like count, mean, standard deviation, etc. of numeric data types.
# 

# In[ ]:




train.describe()


# describe(include = ['O']) will show the descriptive statistics of object data types.

# In[ ]:


train.describe(include=['O'])


# We use info() method to see more information of our train dataset

# In[ ]:


train.info()


# 
# 
# We can see that Age value is missing for many rows.
# 
# Out of 891 rows, the Age value is present only in 714 rows.
# 
# Similarly, Cabin values are also missing in many rows. Only 204 out of 891 rows have Cabin values.
# 

# In[ ]:


train.isnull().sum()


# In[ ]:


test.shape


# In[ ]:


test.head()


# In[ ]:




test.info()


# There are missing entries for Age in Test dataset as well.
# 
# Out of 418 rows in Test dataset, only 332 rows have Age value.

# ## Relationship between Features and Survival
# 
# In this section, we analyze relationship between different features with respect to Survival. We see how different feature values show different survival chance. We also plot different kinds of diagrams to visualize our data and findings.

# In[ ]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))


# 
# ## Pclass vs. Survival
# 
# Higher class passengers have better survival chance.
# 

# In[ ]:


train.Pclass.value_counts()


# In[ ]:


train.groupby('Pclass').Survived.value_counts()


# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)


# 
# ## Sex vs. Survival
# 
# Females have better survival chance.
# 

# In[ ]:


train.Sex.value_counts()


# In[ ]:


train.groupby('Sex').Survived.value_counts()


# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)


# 
# ## Pclass & Sex vs. Survival
# 
# Below, we just find out how many males and females are there in each Pclass. We then plot a stacked bar diagram with that information. We found that there are more males among the 3rd Pclass passengers.
# 

# In[ ]:


tab = pd.crosstab(train['Pclass'], train['Sex'])
print (tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')


# In[ ]:


sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)


# 
# 
# From the above plot, it can be seen that:
# 
# *     Women from 1st and 2nd Pclass have almost 100% survival chance.
# *     Men from 2nd and 3rd Pclass have only around 10% survival chance.
# 
# 

# ## Pclass, Sex & Embarked vs. Survival

# In[ ]:


sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)


# 
# 
# From the above plot, it can be seen that:
# 
# *     Almost all females from Pclass 1 and 2 survived.
# *     Females dying were mostly from 3rd Pclass.
# *     Males from Pclass 1 only have slightly higher survival chance than Pclass 2 and 3.
# 
# 

# ## Embarked vs. Survived

# In[ ]:


train.Embarked.value_counts()


# In[ ]:


train.groupby('Embarked').Survived.value_counts()


# In[ ]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=train)


# ## Parch vs. Survival

# In[ ]:


train.Parch.value_counts()


# In[ ]:


train.groupby('Parch').Survived.value_counts()


# In[ ]:


train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()


# In[ ]:


sns.barplot(x='Parch', y='Survived', ci=None, data=train) # ci=None will hide the error bar


# ## SibSp vs. Survival

# In[ ]:


train.SibSp.value_counts()


# In[ ]:


train.groupby('SibSp').Survived.value_counts()


# In[ ]:


train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()


# In[ ]:


sns.barplot(x='SibSp', y='Survived', ci=None, data=train) # ci=None will hide the error bar


# ## Age vs. Survival

# In[ ]:


train.Age.value_counts()


# In[ ]:


train.groupby('Age')['Survived'].mean()


# In[ ]:


total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]


# 
# ### Correlating Features
# 
# Heatmap of Correlation between different features:
# 
#     Positive numbers = Positive correlation, i.e. increase in one feature will increase the other feature & vice-versa.
# 
#     Negative numbers = Negative correlation, i.e. increase in one feature will decrease the other feature & vice-versa.
# 
# In our case, we focus on which features have strong positive or negative correlation with the Survived feature.
# 

# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)


# 
# ## Feature Extraction
# 
# In this section, we select the appropriate features to train our classifier. Here, we create new features based on existing features. We also convert categorical features into numeric form.
# Name Feature
# 
# Let's first extract titles from Name column.
# 

# In[ ]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')


# In[ ]:




train.head()


# As you can see above, we have added a new column named Title in the Train dataset with the Title present in the particular passenger name

# In[ ]:


pd.crosstab(train['Title'], train['Sex'])


# The number of passengers with each Title is shown above.
# 
# We now replace some less common titles with the name "Other".

# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',  	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# After that, we convert the categorical Title values into numeric form.

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[ ]:




train.head()


# ## Sex Feature
# 
# We convert the categorical value of Sex into numeric. We represent 0 as female and 1 as male.

# In[ ]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


train.head()


# ## Embarked Feature
# 
# There are empty values for some rows for Embarked column. The empty values are represented as "nan" in below list.

# In[ ]:


train.Embarked.unique()


# In[ ]:


train.Embarked.value_counts()


# We find that category "S" has maximum passengers. Hence, we replace "nan" values with "S".

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


train.head()


# We now convert the categorical value of Embarked into numeric. We represent 0 as S, 1 as C and 2 as Q

# In[ ]:


for dataset in train_test_data:
    
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


train.head()


# ## Age Feature
# 
# We first fill the NULL values of Age with a random number between (mean_age - std_age) and (mean_age + std_age).
# 
# We then create a new column named AgeBand. This categorizes age into 5 different age range.

# In[ ]:


for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())


# In[ ]:


train.head()


# Now, we map Age according to AgeBand.

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# In[ ]:


train.head()


# 
# ## Fare Feature
# 
# Replace missing Fare values with the median of Fare.
# 

# In[ ]:


for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# Create FareBand. We divide the Fare into 4 category range.

# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())


# In[ ]:


train.head()


# Map Fare according to FareBand

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


train.head()


# ## SibSp & Parch Feature
# 
# Combining SibSp & Parch feature, we create a new feature named FamilySize.

# In[ ]:


for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# 
# 
# About data shows that:
# 
# *     Having FamilySize upto 4 (from 2 to 4) has better survival chance.
# *     FamilySize = 1, i.e. travelling alone has less survival chance.
# *     Large FamilySize (size of 5 and above) also have less survival chance

# Let's create a new feature named IsAlone. This feature is used to check how is the survival chance while travelling alone as compared to travelling with family.

# In[ ]:


for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# This shows that travelling alone has only 30% survival chance.

# In[ ]:


train.head(1)


# In[ ]:


test.head(1)


# 
# ## Feature Selection
# 
# We drop unnecessary columns/features and keep only the useful ones for our experiment. Column PassengerId is only dropped from Train set because we need PassengerId in Test set while creating Submission file to Kaggle.
# 

# In[ ]:


features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# We are done with Feature Selection/Engineering. Now, we are ready to train a classifier with our feature set.

# ## Classification & Accuracy

# In[ ]:


# Defining training and testing set

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape


# 
# 
# There are many classifying algorithms present. Among them, we choose the following Classification algorithms for our problem:
# 
#     Logistic Regression
#     Support Vector Machines (SVC)
#     Linear SVC
#     k-Nearest Neighbor (KNN)
#     Decision Tree
#     Random Forest
#     Naive Bayes (GaussianNB)
#     Perceptron
#     Stochastic Gradient Descent (SGD)
#     XgBoost
# 
# Here's the training and testing procedure:
# 
#     First, we train these classifiers with our training data.
# 
#     After that, using the trained classifier, we predict the Survival outcome of test data.
# 
#     Finally, we calculate the accuracy score (in percentange) of the trained classifier.
# 
# Please note: that the accuracy score is generated based on our training dataset.
# 
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
import xgboost as xgb


# 
# ## **Logistic Regression**

# In[ ]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')


# ## Support Vector Machine (SVM)

# In[ ]:


clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_svc)


# ## Linear SVM

# In[ ]:


clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_linear_svc)


# ## k-Nearest Neighbors

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn)


# ## Decision Tree

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)


# ## Random Forest

# In[ ]:


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)


# ## Gaussian Naive Bayes

# In[ ]:


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb)


# ## Perceptron

# In[ ]:


clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
print (acc_perceptron)


# ## Stochastic Gradient Descent (SGD)

# In[ ]:


clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print (acc_sgd)


# ## Xgboost

# In[ ]:


xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
acc_xgb = round(xgb_classifier.score(X_train, y_train) * 100, 2)
print (acc_xgb)


# ## Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)


# 
# ## Comparing Models
# 
# Let's compare the accuracy score of all the classifier models used above.
# 

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent', 'XgBoost'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_perceptron, acc_sgd, acc_xgb]
    })

models.sort_values(by='Score', ascending=False)


# ## Create Submission File to Kaggle

# In[ ]:


test.head()


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_random_forest
    })

submission.to_csv('gender_submission.csv', index=False)


# In[ ]:




